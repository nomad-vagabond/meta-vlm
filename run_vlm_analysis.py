from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd

from wing_vlm_analysis import WingAnalysis, OUTPUT_DIR, N_SPAN, N_CHORD


SWEEP_PARAMS = {
    "aspect_ratio": "AR",
    "taper_ratio": "taper",
    "sweep_c4_deg": "sweep",
    "alpha_deg": "alpha",
}


class WingAnalysisManager:
    """
    Setup and run single-variable parameter sweeps over ``WingAnalysis``
    configurations, then export and plot grouped results.

    Parameters
    ----------
    aspect_ratio, taper_ratio, sweep_c4_deg, alpha_deg : float
        Baseline wing parameters. The ``sweep()`` method overrides one of
        these while holding the rest constant.
    root_chord : float
        Root chord [m] (default 1.0).
    velocity, rho : float
        Freestream conditions passed to ``WingAnalysis.run()``.
    n_span, n_chord : int
        VLM resolution passed to ``WingAnalysis.run()``.
    """

    def __init__(self, airfoil_name, aspect_ratio, taper_ratio, sweep_c4_deg, alpha_deg,
                 root_chord=1.0, velocity=50.0, rho=1.225, n_span=N_SPAN, n_chord=N_CHORD):
        self.baseline = dict(
            aspect_ratio=aspect_ratio,
            taper_ratio=taper_ratio,
            sweep_c4_deg=sweep_c4_deg,
            alpha_deg=alpha_deg,
            root_chord=root_chord,
        )
        self.airfoil_name = airfoil_name
        self.velocity = velocity
        self.rho = rho
        self.n_span = n_span
        self.n_chord = n_chord

        self.sweep_variable = None
        self.sweep_values = None
        self.analyses = []

    def sweep(self, variable, values):
        """
        Run a parameter sweep over a single variable.

        Parameters
        ----------
        variable : str
            Name of the variable to sweep: ``"aspect_ratio"``,
            ``"taper_ratio"``, ``"sweep_c4_deg"``, or ``"alpha_deg"``.
        values : list of float
            Values to sweep through.

        Returns
        -------
        analyses : list of WingAnalysis
        """
        if variable not in SWEEP_PARAMS:
            raise ValueError(
                f"Unknown sweep variable '{variable}'. "
                f"Choose from: {list(SWEEP_PARAMS)}"
            )

        self.sweep_variable = variable
        self.sweep_values = list(values)
        self.analyses = []

        for val in values:
            params = dict(self.baseline)
            params[variable] = val
            wa = WingAnalysis(self.airfoil_name, **params)
            wa.run(velocity=self.velocity, rho=self.rho,
                   n_span=self.n_span, n_chord=self.n_chord)
            self.analyses.append(wa)
            print(
                f"{wa.label:45s}  span = {wa.span:.2f} m  "
                f"CL = {wa.aero['CL']:.4f}   CD = {wa.aero['CD']:.4f}"
            )

        return self.analyses

    def _export_coeff_sweep(self, dist_attr, span_key, coeff_name, fixed_str, output_dir, half_span_only):
        """
        Build and write one sweep CSV for a single coefficient (cl or cd).

        Parameters
        ----------
        dist_attr : str
            Name of the per-analysis distribution dict attribute to read,
            e.g. ``"cl_dist"`` or ``"cd_dist"``.
        span_key : str
            Key within that dict holding the per-station coefficient array,
            e.g. ``"cl_span"`` or ``"cd_span"``.
        coeff_name : str
            Short coefficient name used in the output filename, e.g. ``"cl"``.
        fixed_str : str
            Underscore-joined fixed (non-swept) parameters, for the filename.
        output_dir : Path
            Directory to write into.
        half_span_only : bool
            If True, keep only the eta >= 0 half.

        Returns
        -------
        out_path : Path
            Path to the written CSV file.
        """
        ref = self.analyses[0]
        ref_dist = getattr(ref, dist_attr)
        data = {"eta": ref_dist["y_span"] / (ref_dist["span"] / 2)}

        short = SWEEP_PARAMS[self.sweep_variable]
        for wa, val in zip(self.analyses, self.sweep_values):
            data[f"{short}={val:g}"] = getattr(wa, dist_attr)[span_key]

        df = pd.DataFrame(data)
        if half_span_only:
            df = df[df["eta"] >= 0].reset_index(drop=True)

        filename = f"{ref.airfoil_name}_{fixed_str}_{coeff_name}(eta, {short}).csv"
        out_path = output_dir / filename
        df.to_csv(out_path, index=False, float_format="%.6f")

        return out_path

    def export(self, output_dir=OUTPUT_DIR, half_span_only=True):
        """
        Export sweep results into CSV files, one each for Cl, Cd, Γ and clg.

        Columns in each: ``eta``, then one column per sweep value, labelled
        ``<variable>=<value>`` (e.g. ``alpha=-6``, ``taper=0.4``). The
        filenames encode the airfoil name and the fixed (non-swept)
        parameters, e.g. "naca2412_taper1_sweep0_cl(eta, alpha).csv",
        "..._cd(eta, alpha).csv", "..._gamma(eta, alpha).csv" and
        "..._clg(eta, alpha).csv".

        Files are grouped under `output_dir` by [airfoil name] -> [aspect
        ratio] -> [taper ratio], matching ``WingAnalysis`` exports, e.g.
        "naca2412/AR=8/TR=0.6/...cl(eta, alpha).csv". Whichever of aspect
        ratio / taper ratio is the swept variable has no single fixed
        value, so its folder level is omitted.

        Parameters
        ----------
        output_dir : str or Path
            Directory to write into (created if it doesn't exist).
        half_span_only : bool
            If True, export only the eta >= 0 half.

        Returns
        -------
        cl_path, cd_path, gamma_path, clg_path : Path
            Paths to the written Cl, Cd, Γ and clg CSV files.
        """
        if not self.analyses:
            raise RuntimeError("Call `sweep()` before exporting.")

        output_dir = Path(output_dir) / self.airfoil_name
        if self.sweep_variable != "aspect_ratio":
            output_dir /= f"AR={self.baseline['aspect_ratio']:g}"
        if self.sweep_variable != "taper_ratio":
            output_dir /= f"TR={self.baseline['taper_ratio']:g}"
        output_dir.mkdir(parents=True, exist_ok=True)

        fixed_parts = []
        for param, abbrev in SWEEP_PARAMS.items():
            if param != self.sweep_variable:
                fixed_parts.append(f"{abbrev}{self.baseline[param]:g}")
        fixed_str = "_".join(fixed_parts)

        cl_path = self._export_coeff_sweep(
            "cl_dist", "cl_span", "cl", fixed_str, output_dir, half_span_only
        )
        cd_path = self._export_coeff_sweep(
            "cd_dist", "cd_span", "cd", fixed_str, output_dir, half_span_only
        )
        gamma_path = self._export_coeff_sweep(
            "gamma_dist", "gamma_span", "gamma", fixed_str, output_dir, half_span_only
        )
        clg_path = self._export_coeff_sweep(
            "clg_dist", "clg_span", "clg", fixed_str, output_dir, half_span_only
        )

        return cl_path, cd_path, gamma_path, clg_path

    def plot(self, ax=None, savepath=None, cd_ax=None, cd_savepath=None,
             gamma_ax=None, gamma_savepath=None, clg_ax=None, clg_savepath=None):
        """
        Plot all sweep results overlaid, for Cl, Cd, Γ and clg.

        `ax`/`savepath` apply to the Cl plot, `cd_ax`/`cd_savepath` to the
        Cd plot, `gamma_ax`/`gamma_savepath` to the Γ plot, and
        `clg_ax`/`clg_savepath` to the clg plot; each defaults to its own
        new figure if not given.

        Returns
        -------
        ax_cl, ax_cd, ax_gamma, ax_clg : matplotlib.axes.Axes
        """
        if not self.analyses:
            raise RuntimeError("Call `sweep()` before plotting.")
        ax_cl = WingAnalysis.plot_comparison(self.analyses, ax=ax, savepath=savepath)
        ax_cd = WingAnalysis.plot_cd_comparison(self.analyses, ax=cd_ax, savepath=cd_savepath)
        ax_gamma = WingAnalysis.plot_gamma_comparison(self.analyses, ax=gamma_ax, savepath=gamma_savepath)
        ax_clg = WingAnalysis.plot_clg_comparison(self.analyses, ax=clg_ax, savepath=clg_savepath)
        return ax_cl, ax_cd, ax_gamma, ax_clg


# =============================================================================
# Example: run several geometries and overlay their spanwise Cl distributions
# =============================================================================

if __name__ == "__main__":

    mgr = WingAnalysisManager("naca2412",
        aspect_ratio=8.0, taper_ratio=0.4, sweep_c4_deg=0.0, alpha_deg=3.0,
        velocity=50.0, rho=1.225, n_span=80, n_chord=24,
    )

    alphas = np.arange(-20, 22, 2)
    mgr.sweep("alpha_deg", alphas)

    cl_path, cd_path, gamma_path, clg_path = mgr.export()
    print(f"Cl results exported to {cl_path}")
    print(f"Cd results exported to {cd_path}")
    print(f"Γ results exported to {gamma_path}")
    print(f"clg results exported to {clg_path}")

    mgr.plot()
