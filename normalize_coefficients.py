"""
Normalize spanwise VLM aerodynamic coefficients (cla, clg, cdi) against
angle of attack, for symmetric and cambered airfoils.

Ported from "Normalize Coefficients.ipynb" (sections 1-3, excluding the
experimental 3.8 "Cd_min = 0" model). See the notebook for the derivation
of the normalization models and validation plots.

Requirements:
    pip install scipy pandas numpy
"""

import re
from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq, curve_fit, minimize_scalar

# Root directory VLM sweep results are read from (see wing_vlm_analysis.OUTPUT_DIR
# / run_vlm_analysis.WingAnalysisManager.export, which write into this layout).
VLM_DIR = Path(__file__).resolve().parent / "output" / "vlm"

# Root directory normalized-coefficient CSVs are exported into.
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "meta"

DEFAULT_ALPHA_DEG = 6
DEFAULT_R2_THRESHOLD = 0.995  # below this, the polynomial fit isn't trusted -> spline fallback
DEFAULT_LOW_DROP_IND = 3
DEFAULT_HIGH_DROP_IND = -3


def _poly2(x, a, b, c):
    return a * x**2 + b * x + c


def _poly4(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def _r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


# -----------------------------------------------------------------------
# 1. Loading data
# -----------------------------------------------------------------------

def parse_alpha_values(table):
    """
    Parse the angle of attack encoded in each column header of a VLM
    sweep table, e.g. "alpha=6" -> 6.0.

    Returns
    -------
    alpha_values : dict
        Maps each column name to its alpha value [deg].
    """
    alpha_values = {}
    for col in table.columns:
        match = re.search(r"alpha\s*=\s*(-?\d+(?:\.\d+)?)", col)
        if not match:
            raise ValueError(f"Could not parse alpha value from column header: {col!r}")
        alpha_values[col] = float(match.group(1))

    return alpha_values


@dataclass
class WingCoeffTables:
    """Spanwise cl, cd, clg vs. alpha sweep tables for one wing/airfoil."""
    cl_table: pd.DataFrame
    cd_table: pd.DataFrame
    clg_table: pd.DataFrame
    alpha_values: dict
    alphas: np.ndarray
    etas: np.ndarray


def _vlm_sweep_path(vlm_dir, airfoil_name, aspect_ratio, taper_ratio, sweep_c4_deg, coeff_name):
    return (
        Path(vlm_dir)
        / airfoil_name
        / f"AR={aspect_ratio:g}"
        / f"TR={taper_ratio:g}"
        / f"{airfoil_name}_AR{aspect_ratio:g}_taper{taper_ratio:g}_sweep{sweep_c4_deg:g}_{coeff_name}(eta, alpha).csv"
    )


def load_wing_coeff_tables(airfoil_name, aspect_ratio, taper_ratio, sweep_c4_deg,
                            vlm_dir=VLM_DIR, low_drop_ind=DEFAULT_LOW_DROP_IND,
                            high_drop_ind=DEFAULT_HIGH_DROP_IND):
    """
    Load the cl(eta, alpha), cd(eta, alpha) and clg(eta, alpha) sweep CSVs
    written by ``WingAnalysisManager.export()`` for one airfoil/wing
    geometry, dropping untrusted spanwise edge points.

    Parameters
    ----------
    airfoil_name : str
    aspect_ratio, taper_ratio, sweep_c4_deg : float
        Wing geometry the sweep was run at.
    vlm_dir : str or Path
        Root directory the sweep CSVs were exported into.
    low_drop_ind, high_drop_ind : int
        Row-index bounds (``.iloc[low_drop_ind:high_drop_ind]``) applied
        to every table to drop untrusted spanwise edge points.

    Returns
    -------
    tables : WingCoeffTables
    """
    cl_table = pd.read_csv(
        _vlm_sweep_path(vlm_dir, airfoil_name, aspect_ratio, taper_ratio, sweep_c4_deg, "cl")
    ).set_index("eta")
    cd_table = pd.read_csv(
        _vlm_sweep_path(vlm_dir, airfoil_name, aspect_ratio, taper_ratio, sweep_c4_deg, "cd")
    ).set_index("eta")
    clg_table = pd.read_csv(
        _vlm_sweep_path(vlm_dir, airfoil_name, aspect_ratio, taper_ratio, sweep_c4_deg, "clg")
    ).set_index("eta")

    cl_table = cl_table.iloc[low_drop_ind:high_drop_ind]
    cd_table = cd_table.iloc[low_drop_ind:high_drop_ind]
    clg_table = clg_table.iloc[low_drop_ind:high_drop_ind]

    alpha_values = parse_alpha_values(cl_table)
    alphas = np.array(list(alpha_values.values()))
    etas = cl_table.index.to_numpy(dtype=float)

    return WingCoeffTables(cl_table, cd_table, clg_table, alpha_values, alphas, etas)


def _is_symmetric_airfoil(airfoil_name):
    """
    NACA 4-digit airfoils with "00" camber/position digits (e.g.
    "naca0009") are symmetric; everything else (e.g. "naca2412") is
    treated as cambered.
    """
    match = re.match(r"naca(\d{2})\d{2}$", airfoil_name.strip().lower())
    if not match:
        raise ValueError(
            f"Cannot determine airfoil type from name {airfoil_name!r}; "
            f"expected a NACA 4-digit name, e.g. 'naca2412' or 'naca0009'."
        )
    return match.group(1) == "00"


def _alpha_column(tables, alpha_deg):
    col = f"alpha={alpha_deg:g}"
    if col not in tables.cl_table.columns:
        available = sorted(tables.alpha_values.values())
        raise ValueError(f"alpha_deg={alpha_deg} not among the swept values: {available}")
    return col


# -----------------------------------------------------------------------
# 2. Normalizing coefficients
# -----------------------------------------------------------------------

def _normalize_cla_all_alphas(cl_table, alpha_values, alpha_zero_lift, k_alpha):
    """
    Trial-normalize cla(eta, alpha) across all swept alphas for a given
    k_alpha; used by `find_optimal_k_alpha`'s objective only (final
    output uses `normalize_symmetric`/`normalize_cambered` instead).

    `alpha_zero_lift` may be a scalar (symmetric airfoil, uniform across
    eta) or a per-eta array (cambered airfoil's alpha_zero_lift_circ(eta)).
    Columns landing on (or very near) the shifted zero-lift angle are
    masked out to avoid dividing by ~0.
    """
    alpha_zero_lift = np.broadcast_to(alpha_zero_lift, len(cl_table)).astype(float)
    norm_table = pd.DataFrame(index=cl_table.index)
    for col, alpha_deg in alpha_values.items():
        cl_vals = cl_table[col].to_numpy(dtype=float)
        sin_alpha = np.sin(np.radians(alpha_deg - alpha_zero_lift))
        cos_alpha = np.cos(np.radians(alpha_deg))
        with np.errstate(divide="ignore", invalid="ignore"):
            values = (cl_vals / (2 * pi * sin_alpha)) * np.sqrt(2 / (1 + k_alpha * cos_alpha))
        norm_table[col] = np.where(np.abs(sin_alpha) > 1e-6, values, np.nan)

    return norm_table


def find_optimal_k_alpha(cl_table, alpha_values, alpha_zero_lift, k_bounds=(0.0, 20.0)):
    """
    Fit the k_alpha constant of the finite-wing cl_a normalization model
    by minimizing the median-relative spread of the normalized
    cl_a(eta)* curves across all swept angles of attack, fit directly
    against `cl_table` alone (no separate reference airfoil needed).

    Parameters
    ----------
    cl_table : pd.DataFrame
        The airfoil's own cl(eta, alpha) sweep table (`WingCoeffTables.cl_table`).
    alpha_values : dict
        As returned in `WingCoeffTables.alpha_values` for `cl_table`.
    alpha_zero_lift : float or np.ndarray
        Zero-lift angle of attack [deg]: a single scalar for a symmetric
        airfoil (0 by construction), or the per-eta alpha_zero_lift_circ(eta)
        array (from `fit_alpha_zero_lift`) for a cambered airfoil.
    k_bounds : tuple of float
        Search bounds for k_alpha.

    Returns
    -------
    k_alpha : float
    """
    def objective(k_alpha):
        norm_table = _normalize_cla_all_alphas(cl_table, alpha_values, alpha_zero_lift, k_alpha)
        row_median = norm_table.median(axis=1, skipna=True)
        rel_dev = norm_table.sub(row_median, axis=0).abs().div(row_median.abs(), axis=0)
        return np.nanmean(rel_dev.values)

    result = minimize_scalar(objective, bounds=k_bounds, method="bounded", options={"xatol": 1e-6})
    return round(result.x, 2)


def _find_zero_lift(alphas, coeff_row, poly_func, popt):
    sign = np.sign(coeff_row)
    cross_idx = np.where(np.diff(sign) > 0)[0]
    if len(cross_idx) == 0:
        return np.nan

    i = cross_idx[0]
    a, b = alphas[i], alphas[i + 1]

    f = lambda x: poly_func(x, *popt)
    if f(a) * f(b) > 0:
        pad = 0.5 * (b - a)
        a, b = a - pad, b + pad

    return brentq(f, a, b)


def fit_alpha_zero_lift(cl_table, alphas, r2_threshold=DEFAULT_R2_THRESHOLD):
    """
    Fit alpha_zero_lift(eta): the angle of attack at which each spanwise
    station's cl(alpha) row crosses zero, by fitting a quadratic
    (falling back to a quartic) polynomial across `alphas` and rooting it.

    Parameters
    ----------
    cl_table : pd.DataFrame
        A cl(eta, alpha) or clg(eta, alpha) sweep table.
    alphas : np.ndarray
        Swept alpha values [deg], aligned with `cl_table`'s columns.
    r2_threshold : float
        Minimum R² for a polynomial fit to be trusted.

    Returns
    -------
    alpha_zero_lift : np.ndarray
        One value per row of `cl_table`.
    """
    alpha_zero_lift = np.empty(len(cl_table))

    for i, (_, row) in enumerate(cl_table.iterrows()):
        coeff = row.to_numpy(dtype=float)

        popt2, _ = curve_fit(_poly2, alphas, coeff, p0=[0.0, 0.1, 0.0])
        r2_2 = _r_squared(coeff, _poly2(alphas, *popt2))

        popt4, _ = curve_fit(_poly4, alphas, coeff, p0=[0.0, 0.0, 0.0, 0.1, 0.0], maxfev=10000)
        r2_4 = _r_squared(coeff, _poly4(alphas, *popt4))

        if r2_4 >= r2_threshold:
            popt, poly_func = popt4, _poly4
        elif r2_2 >= r2_threshold:
            popt, poly_func = popt2, _poly2
        else:
            raise ValueError(
                f"Failed to fit a polynomial at eta={cl_table.index[i]} "
                f"within r2_threshold={r2_threshold}."
            )

        alpha_zero_lift[i] = _find_zero_lift(alphas, coeff, poly_func, popt)

    return alpha_zero_lift


def _find_min_from_poly(coeffs, x_range):
    poly = np.poly1d(coeffs)
    dpoly = poly.deriv()
    ddpoly = dpoly.deriv()
    roots = dpoly.r
    real_roots = roots[np.abs(roots.imag) < 1e-8].real
    candidates = [x for x in real_roots if x_range[0] <= x <= x_range[1] and ddpoly(x) > 0]
    if candidates:
        x_min = min(candidates, key=lambda x: poly(x))
        return x_min, poly(x_min)

    res = minimize_scalar(poly, bounds=x_range, method="bounded")
    return res.x, res.fun


def fit_alpha_min_drag(cd_table, alphas, r2_threshold=DEFAULT_R2_THRESHOLD):
    """
    Fit alpha_min_drag(eta) and cd_min(eta): the angle of attack and
    coefficient value at the minimum of each spanwise station's
    cd(alpha) row, by fitting a quadratic (falling back to a quartic,
    then a smoothing spline) across `alphas`.

    Parameters
    ----------
    cd_table : pd.DataFrame
        A cd(eta, alpha) sweep table.
    alphas : np.ndarray
        Swept alpha values [deg], aligned with `cd_table`'s columns.
    r2_threshold : float
        Minimum R² for a polynomial fit to be trusted before falling
        back to a smoothing spline.

    Returns
    -------
    alpha_min_drag, cd_min : np.ndarray
        One value each per row of `cd_table`.
    """
    x_range = (alphas.min(), alphas.max())
    alpha_min_drag = np.empty(len(cd_table))
    cd_min = np.empty(len(cd_table))

    for i, (_, row) in enumerate(cd_table.iterrows()):
        cd_vals = row.to_numpy(dtype=float)

        popt2, _ = curve_fit(_poly2, alphas, cd_vals, p0=[1.0, 0.0, cd_vals.min()])
        r2_2 = _r_squared(cd_vals, _poly2(alphas, *popt2))

        popt4, _ = curve_fit(_poly4, alphas, cd_vals, p0=[0.0, 0.0, 1.0, 0.0, cd_vals.min()], maxfev=10000)
        r2_4 = _r_squared(cd_vals, _poly4(alphas, *popt4))

        if r2_4 >= r2_threshold:
            a_min, c_min = _find_min_from_poly(popt4, x_range)
        elif r2_2 >= r2_threshold:
            a_min, c_min = _find_min_from_poly(popt2, x_range)
        else:
            spline = UnivariateSpline(alphas, cd_vals, k=4, s=len(alphas) * 1e-7)
            res = minimize_scalar(spline, bounds=x_range, method="bounded")
            a_min, c_min = res.x, float(res.fun)

        alpha_min_drag[i] = a_min
        cd_min[i] = c_min

    return alpha_min_drag, cd_min


def normalize_symmetric(tables, alpha_zero_lift, k_alpha, alpha_deg=DEFAULT_ALPHA_DEG):
    """
    Normalize cla*(eta), clg*(eta), cdi*(eta) for a symmetric airfoil at
    a single angle of attack `alpha_deg` (must be one of the swept alpha
    columns in `tables`).

    Assumes zero minimum drag at zero lift (cd_min=0, alpha_min_drag=0),
    valid by symmetry.

    In addition to the normalized coefficients, the returned table
    carries alpha_zero_lift(eta), alpha_min_drag(eta) and cd_min(eta) —
    constant across eta here, but included so the output has the same
    shape as `normalize_cambered`'s — needed to restore cla, clg, cdi
    from cla*, clg*, cdi* (together with k_alpha, stored separately by
    `export_k_alpha`).

    Returns
    -------
    norm_df : pd.DataFrame
        Indexed by eta, with columns "cla*", "clg*", "cdi*",
        "alpha_zero_lift", "alpha_min_drag", "cd_min".
    """
    if alpha_deg == int(alpha_zero_lift):
        raise ValueError(
            f"alpha_deg={alpha_deg} is the (near) zero-lift angle; normalization is singular there."
        )
    col = _alpha_column(tables, alpha_deg)

    sin_alpha = np.sin(np.radians(alpha_deg))
    cos_alpha = np.cos(np.radians(alpha_deg))

    cla = tables.cl_table[col].to_numpy(dtype=float)
    clg = tables.clg_table[col].to_numpy(dtype=float)
    cdi = tables.cd_table[col].to_numpy(dtype=float)

    cla_star = (cla / (2 * pi * sin_alpha)) * np.sqrt(2 / (1 + k_alpha * cos_alpha))
    clg_star = clg / (2 * pi * sin_alpha)
    cdi_star = cdi / (cos_alpha * sin_alpha**2)

    n = len(tables.etas)
    return pd.DataFrame(
        {
            "cla*": cla_star,
            "clg*": clg_star,
            "cdi*": cdi_star,
            "alpha_zero_lift": np.full(n, alpha_zero_lift),
            "alpha_min_drag": np.zeros(n),
            "cd_min": np.zeros(n),
        },
        index=pd.Index(tables.etas, name="eta"),
    )


def normalize_cambered(tables, k_alpha, alpha_deg=DEFAULT_ALPHA_DEG,
                        alpha_zero_lift_circ=None, alpha_min_drag=None, cd_min=None,
                        r2_threshold=DEFAULT_R2_THRESHOLD):
    """
    Normalize cla*(eta), clg*(eta), cdi*(eta) for a cambered airfoil at a
    single angle of attack `alpha_deg` (must be one of the swept alpha
    columns in `tables`).

    `alpha_zero_lift_circ`, `alpha_min_drag`, `cd_min` are the per-eta
    fits from `fit_alpha_zero_lift(tables.clg_table, tables.alphas)` and
    `fit_alpha_min_drag(tables.cd_table, tables.alphas)`; computed
    automatically if not supplied, so callers normalizing at several
    alphas can fit once and reuse.

    In addition to the normalized coefficients, the returned table
    carries these per-eta alpha_zero_lift(eta), alpha_min_drag(eta) and
    cd_min(eta) values — needed to restore cla, clg, cdi from cla*,
    clg*, cdi* (together with k_alpha, stored separately by
    `export_k_alpha`).

    Returns
    -------
    norm_df : pd.DataFrame
        Indexed by eta, with columns "cla*", "clg*", "cdi*",
        "alpha_zero_lift", "alpha_min_drag", "cd_min".
    """
    col = _alpha_column(tables, alpha_deg)

    if alpha_zero_lift_circ is None:
        alpha_zero_lift_circ = fit_alpha_zero_lift(tables.clg_table, tables.alphas, r2_threshold)
    if alpha_min_drag is None or cd_min is None:
        alpha_min_drag, cd_min = fit_alpha_min_drag(tables.cd_table, tables.alphas, r2_threshold)

    sin_alpha = np.sin(np.radians(alpha_deg))
    cos_alpha = np.cos(np.radians(alpha_deg))
    sin_alpha_shifted = np.sin(np.radians(alpha_deg - alpha_zero_lift_circ))

    cla = tables.cl_table[col].to_numpy(dtype=float)
    clg = tables.clg_table[col].to_numpy(dtype=float)
    cdi = tables.cd_table[col].to_numpy(dtype=float)

    cla_star = (cla / (2 * pi * sin_alpha_shifted)) * np.sqrt(2 / (1 + k_alpha * cos_alpha))
    clg_star = clg / (2 * pi * sin_alpha_shifted)

    cos_alpha_md = np.cos(np.radians(alpha_deg - alpha_min_drag))
    sin_alpha_md = np.sin(np.radians(alpha_min_drag))
    cdi_star = (cdi - cd_min) / (cos_alpha_md * (sin_alpha - sin_alpha_md) ** 2)

    return pd.DataFrame(
        {
            "cla*": cla_star,
            "clg*": clg_star,
            "cdi*": cdi_star,
            "alpha_zero_lift": alpha_zero_lift_circ,
            "alpha_min_drag": alpha_min_drag,
            "cd_min": cd_min,
        },
        index=pd.Index(tables.etas, name="eta"),
    )


def normalize_wing_coefficients(
    airfoil_name, aspect_ratio, taper_ratio, sweep_c4_deg,
    alpha_deg=DEFAULT_ALPHA_DEG, vlm_dir=VLM_DIR,
    low_drop_ind=DEFAULT_LOW_DROP_IND, high_drop_ind=DEFAULT_HIGH_DROP_IND,
    r2_threshold=DEFAULT_R2_THRESHOLD,
):
    """
    Load `airfoil_name`'s own VLM sweep and normalize cla*, clg*,
    cdi*(eta) at `alpha_deg`. The normalization model — and whether
    alpha_zero_lift(eta)/alpha_min_drag(eta)/cd_min(eta) are fit at all,
    or assumed zero by symmetry — is chosen by parsing `airfoil_name`
    (see `_is_symmetric_airfoil`). Every fit, including k_alpha, uses
    only this airfoil's own sweep; no separate reference airfoil is
    needed for either case.

    Returns
    -------
    norm_df : pd.DataFrame
        Indexed by eta, with columns "cla*", "clg*", "cdi*",
        "alpha_zero_lift", "alpha_min_drag", "cd_min".
    k_alpha : float
        Store separately with `export_k_alpha` (it's a per-wing scalar,
        not a per-eta value).
    """
    tables = load_wing_coeff_tables(
        airfoil_name, aspect_ratio, taper_ratio, sweep_c4_deg, vlm_dir, low_drop_ind, high_drop_ind
    )

    if _is_symmetric_airfoil(airfoil_name):
        alpha_zero_lift = 0.0
        k_alpha = find_optimal_k_alpha(tables.cl_table, tables.alpha_values, alpha_zero_lift)
        norm_df = normalize_symmetric(tables, alpha_zero_lift, k_alpha, alpha_deg)
    else:
        alpha_zero_lift_circ = fit_alpha_zero_lift(tables.clg_table, tables.alphas, r2_threshold)
        k_alpha = find_optimal_k_alpha(tables.cl_table, tables.alpha_values, alpha_zero_lift_circ)
        alpha_min_drag, cd_min = fit_alpha_min_drag(tables.cd_table, tables.alphas, r2_threshold)
        norm_df = normalize_cambered(
            tables, k_alpha, alpha_deg, alpha_zero_lift_circ, alpha_min_drag, cd_min, r2_threshold
        )

    return norm_df, k_alpha


# -----------------------------------------------------------------------
# 3. Storing results
# -----------------------------------------------------------------------

def export_normalized_coefficients(norm_df, airfoil_name, aspect_ratio, taper_ratio, sweep_c4_deg,
                                    alpha_deg, output_dir=OUTPUT_DIR):
    """
    Export a normalized-coefficient DataFrame (as returned by
    `normalize_symmetric` / `normalize_cambered` and friends) to a CSV
    file, indexed by eta with columns "cla*", "clg*", "cdi*",
    "alpha_zero_lift", "alpha_min_drag", "cd_min" — everything needed to
    restore cla, clg, cdi except k_alpha, which is a per-wing scalar
    stored separately by `export_k_alpha`.

    Files are grouped under `output_dir` by [airfoil name] -> [aspect
    ratio] -> [taper ratio], matching the vlm exports, e.g.
    "naca2412/AR=8/TR=0.6/naca2412_AR8_taper0.6_sweep0_alpha6_norm.csv".

    Parameters
    ----------
    norm_df : pd.DataFrame
    airfoil_name : str
    aspect_ratio, taper_ratio, sweep_c4_deg, alpha_deg : float
    output_dir : str or Path
        Root directory to write into (created if it doesn't exist).

    Returns
    -------
    out_path : Path
        Path to the written CSV file.
    """
    output_dir = (
        Path(output_dir)
        / airfoil_name
        / f"AR={aspect_ratio:g}"
        / f"TR={taper_ratio:g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{airfoil_name}"
        f"_AR{aspect_ratio:g}"
        f"_taper{taper_ratio:g}"
        f"_sweep{sweep_c4_deg:g}"
        f"_alpha{alpha_deg:g}_norm.csv"
    )
    out_path = output_dir / filename
    norm_df.to_csv(out_path, index=True, float_format="%.6f")

    return out_path


def export_k_alpha(k_alpha, airfoil_name, aspect_ratio, taper_ratio, output_dir=OUTPUT_DIR):
    """
    Upsert a wing's k_alpha constant into a per-(airfoil, aspect ratio)
    table indexed by taper ratio, one level up from the per-taper-ratio
    directory used by `export_normalized_coefficients`, e.g.
    "naca2412/AR=8/naca2412_AR=8_k_alpha.csv". Running normalization for
    several taper ratios accumulates one row each into the same file;
    re-running for a taper ratio already present replaces its row.

    Parameters
    ----------
    k_alpha : float
    airfoil_name : str
    aspect_ratio, taper_ratio : float
    output_dir : str or Path
        Root directory to write into (created if it doesn't exist).

    Returns
    -------
    out_path : Path
        Path to the written CSV file.
    """
    ar_dir = Path(output_dir) / airfoil_name / f"AR={aspect_ratio:g}"
    ar_dir.mkdir(parents=True, exist_ok=True)
    out_path = ar_dir / f"{airfoil_name}_AR={aspect_ratio:g}_k_alpha.csv"

    if out_path.exists():
        table = pd.read_csv(out_path).set_index("TR")
    else:
        table = pd.DataFrame(columns=["k_alpha"])
        table.index.name = "TR"

    table.loc[taper_ratio, "k_alpha"] = k_alpha
    table = table.sort_index()
    table.to_csv(out_path, float_format="%.6f")

    return out_path


if __name__ == "__main__":

    AIRFOILS = ["naca2412", "naca0009"]  # each processed independently, from its own VLM sweep
    AR, TR, SW = 8, 0.2, 0
    ALPHA_DEG = 6

    for airfoil in AIRFOILS:
        norm_df, k_alpha = normalize_wing_coefficients(airfoil, AR, TR, SW, alpha_deg=ALPHA_DEG)
        out_path = export_normalized_coefficients(norm_df, airfoil, AR, TR, SW, ALPHA_DEG)
        k_alpha_path = export_k_alpha(k_alpha, airfoil, AR, TR)

        print(f"[{airfoil}] normalized coefficients exported to {out_path}")
        print(f"[{airfoil}] k_alpha exported to {k_alpha_path}")
