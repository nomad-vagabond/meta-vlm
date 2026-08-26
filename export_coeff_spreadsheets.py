"""
Bundle normalized spanwise coefficient tables (cla*, clg*, cdi*, plus the
alpha_zero_lift/alpha_min_drag/cd_min fits needed to restore them) into
Excel workbooks: one workbook per (airfoil, aspect ratio), with one sheet
per taper ratio.

Reads from, and writes into, the output/meta/{airfoil}/AR={AR}/TR={TR}/
layout written by ``normalize_coefficients.export_normalized_coefficients()``
-- each workbook lands alongside the per-TR CSVs and k_alpha table it was
built from.

Requirements:
    pip install pandas openpyxl
"""

from pathlib import Path

import pandas as pd

# Root directory normalized-coefficient CSVs are read from, and workbooks
# are written into (see normalize_coefficients.OUTPUT_DIR).
META_DIR = Path(__file__).resolve().parent / "output" / "meta"


def _dir_values(parent_dir, prefix):
    """
    Scan `parent_dir` for immediate subdirectories named "{prefix}{value}"
    (e.g. prefix="AR=" matches "AR=8"), returning {value: path}.
    """
    values = {}
    for entry in parent_dir.iterdir():
        if entry.is_dir() and entry.name.startswith(prefix):
            values[float(entry.name[len(prefix):])] = entry

    return values


def _find_norm_csv(tr_dir):
    matches = list(tr_dir.glob("*_norm.csv"))
    if not matches:
        raise FileNotFoundError(f"No '*_norm.csv' file found in {tr_dir}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple '*_norm.csv' files found in {tr_dir}: {matches}. "
            f"Expected exactly one (normalization was run for more than one alpha?)."
        )

    return matches[0]


def export_normalized_spreadsheets(airfoil_name, meta_dir=META_DIR):
    """
    For one airfoil, build an Excel workbook per aspect ratio, with one
    sheet per taper ratio found on disk.

    Sheets are named "TR={value}" and hold the normalized-coefficient
    table (columns "cla*", "clg*", "cdi*", "alpha_zero_lift",
    "alpha_min_drag", "cd_min", indexed by eta) as-is. Workbooks are
    placed at `meta_dir/{airfoil_name}/AR={AR}/{airfoil_name}_AR{AR}_norm.xlsx`,
    alongside the per-TR CSVs and k_alpha table they were built from.

    Parameters
    ----------
    airfoil_name : str
    meta_dir : str or Path
        Root directory the normalized-coefficient CSVs were exported into,
        and workbooks are written into.

    Returns
    -------
    out_paths : list of Path
        Paths to the written workbooks.
    """
    airfoil_dir = Path(meta_dir) / airfoil_name
    if not airfoil_dir.is_dir():
        raise FileNotFoundError(f"No normalized results found for airfoil {airfoil_name!r} at {airfoil_dir}")

    ar_dirs = _dir_values(airfoil_dir, "AR=")
    out_paths = []

    for aspect_ratio, ar_dir in sorted(ar_dirs.items()):
        tr_dirs = _dir_values(ar_dir, "TR=")
        out_path = ar_dir / f"{airfoil_name}_AR{aspect_ratio:g}_norm.xlsx"

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for taper_ratio, tr_dir in sorted(tr_dirs.items()):
                csv_path = _find_norm_csv(tr_dir)
                table = pd.read_csv(csv_path).set_index("eta")
                table.to_excel(writer, sheet_name=f"TR={taper_ratio:g}")

        out_paths.append(out_path)

    return out_paths


if __name__ == "__main__":

    AIRFOILS = ["naca2412", "naca0009"]

    for airfoil in AIRFOILS:
        paths = export_normalized_spreadsheets(airfoil)
        print(f"[{airfoil}] wrote {len(paths)} workbook(s):")
        for path in paths:
            print(f"  {path}")
