"""
Spanwise Cl distribution — NACA 2412 wing, parametric geometry
Using AeroSandbox VortexLatticeMethod (VLM)

Requirements:
    pip install aerosandbox matplotlib
"""

from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Root directory for exported analysis results, inside the project's `output/` folder
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "vlm"

N_SPAN = 30
N_CHORD = 8


# Maps velocity_component names accepted by plot_velocity_field()
# to their column index in a geometry-axes [u, v, w] velocity vector.
VELOCITY_COMPONENTS = {"x": 0, "u": 0, "y": 1, "v": 1, "z": 2, "w": 2}
VELOCITY_COMPONENT_LETTERS = {0: "u", 1: "v", 2: "w"}

# Maps the `plane` argument of plot_velocity_field() to the (horizontal,
# vertical) column indices of `points` / velocity vectors it plots, plus
# axis labels.
PLANE_AXES = {
    "yz": (1, 2, "y (spanwise) [m]", "z [m]"),
    "xz": (0, 2, "x (chordwise) [m]", "z [m]"),
}


class WingAnalysis:
    """
    Build and analyze a tapered/swept wing with AeroSandbox VLM.

    On construction, builds the wing geometry (`self.wing`).
    Call `run()` to solve the VLM and compute the spanwise Cl, Cd
    distribution, then `plot_cl_distribution()`, `plot_cd_distribution()` /
    `plot_comparison()` to visualize results.

    Parameters
    ----------
    aspect_ratio : float
        Wing aspect ratio, AR = b^2 / S
    taper_ratio : float
        tip_chord / root_chord
    sweep_c4_deg : float
        Quarter-chord sweep angle [deg]
    root_chord : float
        Root chord [m] (default 1.0)
    label : str, optional
        Legend label for comparison plots. Defaults to a description
        derived from the geometry parameters.
    """

    def __init__(self, airfoil_name, aspect_ratio, taper_ratio, sweep_c4_deg, alpha_deg, root_chord=1.0):
        self.label = f"AR={aspect_ratio} (λ={taper_ratio}, Λ={sweep_c4_deg}°, α={alpha_deg}°)"

        self.aspect_ratio = aspect_ratio
        self.taper_ratio = taper_ratio
        self.sweep_c4_deg = sweep_c4_deg
        self.alpha_deg = alpha_deg
        self.root_chord = root_chord
        self.tip_chord = root_chord * taper_ratio
        self.span = aspect_ratio * root_chord * (1 + taper_ratio) / 2

        self.airfoil_name = airfoil_name
        self.airfoil = asb.Airfoil(self.airfoil_name)
        self.wing = self._build_wing(self.airfoil, self.root_chord, self.tip_chord, self.span, self.sweep_c4_deg)

        # Operating conditions from the most recent `run()`
        self.velocity = None
        self.rho = None

        # Results from the most recent `run()`
        self.vlm = None
        self.aero = None
        self.cl_dist = None
        self.cd_dist = None
        self.cs_dist = None
        self.gamma_dist = None
        self.clg_dist = None

    # -------------------------------------------------------------------
    # Geometry / solver
    # -------------------------------------------------------------------

    @staticmethod
    def _build_wing(airfoil, root_chord, tip_chord, span, sweep_c4_deg):
        """
        Build a single-section tapered/swept Wing (symmetric half-wing definition).

        Returns
        -------
        airplane : asb.Airplane
        """
        tip_x_offset = (span / 2) * np.tan(np.radians(sweep_c4_deg)) + 0.25 * (root_chord - tip_chord)

        root_sect = asb.WingXSec(xyz_le=[0, 0, 0], chord=root_chord, airfoil=airfoil)
        tip_sect = asb.WingXSec(xyz_le=[tip_x_offset, span / 2, 0], chord=tip_chord, airfoil=airfoil)
        wing = asb.Wing(name="Wing", symmetric=True, xsecs=[root_sect, tip_sect])

        return asb.Airplane(wings=[wing])

    def _local_chord_le(self, y):
        """
        Local leading-edge x-position and chord at a spanwise station,
        by linear interpolation between root and tip (matches the
        straight-taper/sweep geometry built by `_build_wing`).

        Parameters
        ----------
        y : float
            Spanwise station [m].

        Returns
        -------
        x_le, chord : float
            Local leading-edge x-position [m] and local chord [m].
        """
        tip_x_offset = (self.span / 2) * np.tan(np.radians(self.sweep_c4_deg)) + 0.25 * (self.root_chord - self.tip_chord)
        eta = abs(y) / (self.span / 2)
        x_le = eta * tip_x_offset
        chord = self.root_chord + eta * (self.tip_chord - self.root_chord)
        return x_le, chord

    @staticmethod
    def _run_vlm(airplane, velocity, alpha_deg, n_span=30, n_chord=8):
        """
        Run AeroSandbox VLM and return the (vlm, aero) results.

        Returns
        -------
        vlm : asb.VortexLatticeMethod
        aero : dict
        """
        op_point = asb.OperatingPoint(velocity=velocity, alpha=alpha_deg)

        half_cosine = lambda start, end, n: start + (end - start) * np.sin(np.pi / 2 * np.linspace(0, 1, n))

        vlm = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=op_point,
            spanwise_resolution=n_span,
            chordwise_resolution=n_chord,
            # spanwise_spacing_function=half_cosine,
            align_trailing_vortices_with_wind=False
        )

        aero = vlm.run()
        return vlm, aero

    @staticmethod
    def _wind_axis_forces(vlm):
        """
        Per-panel lift and drag forces in wind axes
        """
        alpha_rad = np.radians(vlm.op_point.alpha)
        Fx = vlm.forces_geometry[:, 0]
        Fy = vlm.forces_geometry[:, 1]
        Fz = vlm.forces_geometry[:, 2]

        Fxw, Fyw, Fzw = vlm.op_point.convert_axes(
            Fx, Fy, Fz,
            from_axes="geometry", to_axes="wind",
        )

        # wind axes: x upstream, z down  ->  negate for positive L, D
        L = -Fzw
        D = -Fxw
        Y =  Fyw

        return L, D, Y

    @staticmethod
    def _wind_axis_lift(vlm):
        """
        Per-panel lift in wind axes (rotation about y by −α, valid for β=0):
            L = Fz·cos α − Fx·sin α
        """
        alpha_rad = np.radians(vlm.op_point.alpha)
        Fx = vlm.forces_geometry[:, 0]
        Fz = vlm.forces_geometry[:, 2]
        return Fz * np.cos(alpha_rad) - Fx * np.sin(alpha_rad)

    @staticmethod
    def _wind_axis_drag(vlm):
        """
        Per-panel drag in wind axes (positive = opposing motion):
            D = Fx·cos α + Fz·sin α
        """
        alpha_rad = np.radians(vlm.op_point.alpha)
        Fx = vlm.forces_geometry[:, 0]
        Fz = vlm.forces_geometry[:, 2]
        return Fx * np.cos(alpha_rad) + Fz * np.sin(alpha_rad)

    def _compute_spanwise_coeff_distributions(self, rho=1.225):
        """
        Compute the spanwise local-Cl, Cd, Cs and circulation Γ distribution
        from the solved `self.vlm`.

        Aggregates chordwise panels into spanwise strips: for each unique
        spanwise (y) station, sums lift force and planform area across all
        chordwise panels, then divides by dynamic pressure to get local coefficient.

        The circulation Γ(y) is recovered from the strip's sectional lift
        via the Kutta-Joukowski theorem, l'(y) = rho * V * Γ(y), i.e.
        Γ(y) = l'(y) / (rho * V), where l'(y) is lift-per-unit-span
        (strip_lift / strip_width).

        Parameters
        ----------
        rho : float
            Freestream density [kg/m^3], used for dynamic pressure.

        Returns
        -------
        y_span : np.ndarray
            Spanwise coordinates of each strip [m] (both half-wings, sorted).
        cl_span : np.ndarray
            Local section Cl at each spanwise station.
        cd_span : np.ndarray
            Local section Cd at each spanwise station.
        cs_span : np.ndarray
            Local section Cs (coefficient of side force) at each spanwise station.
        gamma_span : np.ndarray
            Local circulation Γ at each spanwise station [m^2/s].
        clg_span : np.ndarray
            Local section Cl recovered from circulation via the thin-airfoil/
            lifting-line relation Cl = 2Γ/(V·c), at each spanwise station.
        """
        vlm = self.vlm
        velocity = vlm.op_point.velocity
        alpha_rad = np.radians(vlm.op_point.alpha)
        q_inf = 0.5 * rho * velocity**2

        cp = vlm.collocation_points          # (n_panels, 3)
        areas = vlm.areas                     # panel area [m^2]
        Lift, Drag, SideForce = self._wind_axis_forces(vlm)

        Gamma = vlm.vortex_strengths

        y_all = cp[:, 1]
        y_strips = np.unique(y_all)

        cl_list = []
        clgamma_list = []
        cd_list = []
        cs_list = []
        gamma_list = []
        for y in y_strips:
            mask = y_all == y

            strip_lift = Lift[mask].sum()
            strip_drag = Drag[mask].sum()
            strip_sideforce = SideForce[mask].sum()

            gamma_strip = Gamma[mask].sum()
            strip_area = areas[mask].sum()
            panel_idx = np.nonzero(mask)[0][0]
            strip_width = abs(vlm.front_right_vertices[panel_idx, 1] - vlm.front_left_vertices[panel_idx, 1])
            chord = strip_area/strip_width

            _cl = strip_lift / (q_inf * strip_area)
            _cd = strip_drag / (q_inf * strip_area)
            _cs = strip_sideforce / (q_inf * strip_area)
            _clgamma = 2 * gamma_strip / (velocity * chord)

            cl_list.append(_cl)
            cd_list.append(_cd)
            cs_list.append(_cs)
            gamma_list.append(gamma_strip)
            clgamma_list.append(_clgamma)

        cl_span = np.array(cl_list)
        cd_span = np.array(cd_list)
        cs_span = np.array(cs_list)
        gamma_span = np.array(gamma_list)
        clg_span = np.array(clgamma_list)

        return y_strips, cl_span, cd_span, cs_span, gamma_span, clg_span

    def run(self, velocity=50.0, rho=1.225, n_span=N_SPAN, n_chord=N_CHORD, normalize_by_cl_inf=False):
        """
        Solve the VLM for this wing (and, if requested, the infinite-wing
        reference) and compute the spanwise Cl distribution.

        The infinite-wing reference is solved with ``3 * n_span`` panels
        for higher fidelity near the root.

        Populates `self.vlm`, `self.aero`, `self.cl_dist`, `self.cd_dist`, 
        `self.gamma_dist`, and `self.clg_dist`.

        Returns
        -------
        cl_dist, cd_dist, gamma_dist, clg_dist : dict
            Spanwise Cl, Cd, Γ and circulation-derived Cl distributions, see
            `plot_cl_distribution`, `plot_cd_distribution`,
            `plot_gamma_distribution` and `plot_clg_distribution` for keys.
        """
        self.velocity = velocity
        self.rho = rho

        self.vlm, self.aero = self._run_vlm(self.wing, velocity, self.alpha_deg, n_span, n_chord)

        y_span, cl_span, cd_span, cs_span, gamma_span, clg_span = self._compute_spanwise_coeff_distributions(rho=rho)

        self.cl_dist = dict(y_span=y_span, cl_span=cl_span, span=self.span, label=self.label, CL=self.aero["CL"])
        self.cd_dist = dict(y_span=y_span, cd_span=cd_span, span=self.span, label=self.label, CD=self.aero["CD"])
        self.gamma_dist = dict(y_span=y_span, gamma_span=gamma_span, span=self.span, label=self.label)
        self.clg_dist = dict(y_span=y_span, clg_span=clg_span, span=self.span, label=self.label)

        return self.cl_dist, self.cd_dist, self.gamma_dist, self.clg_dist

    def _export_distribution(self, dist, span_key, name, output_dir, half_span_only):
        """
        Export a spanwise distribution dict to a CSV file. Requires `run()` first.

        Files are grouped under `output_dir` by [airfoil name] -> [aspect
        ratio] -> [taper ratio], e.g.
        "naca2412/AR=8/TR=0.6/naca2412_AR8_taper0.6_sweep10_alpha5_cl.csv".

        Parameters
        ----------
        dist : dict
            Distribution dict as populated by `run()` (e.g. `self.cl_dist`),
            with keys "y_span", "span", and `span_key`.
        span_key : str
            Key within `dist` holding the per-station values, e.g. "cl_span".
        name : str
            Short coefficient name, used as both the CSV column name and
            the filename suffix, e.g. "cl", "cd", "gamma", "clg".
        output_dir : str or Path
            Directory to write the CSV file into (created if it doesn't exist).
        half_span_only : bool
            If True, export only the y >= 0 half of the (symmetric)
            distribution.

        Returns
        -------
        out_path : Path
            Path to the written CSV file.
        """
        if dist is None:
            raise RuntimeError("Call `run()` before exporting results.")

        output_dir = (
            Path(output_dir)
            / self.airfoil_name
            / f"AR={self.aspect_ratio:g}"
            / f"TR={self.taper_ratio:g}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{self.airfoil_name}"
            f"_AR{self.aspect_ratio:g}"
            f"_taper{self.taper_ratio:g}"
            f"_sweep{self.sweep_c4_deg:g}"
            f"_alpha{self.alpha_deg:g}_{name}.csv"
        )
        out_path = output_dir / filename

        df = pd.DataFrame({
            "eta": dist["y_span"] / (dist["span"] / 2),
            name: dist[span_key],
        })
        if half_span_only:
            df = df[df["eta"] >= 0].reset_index(drop=True)

        df.to_csv(out_path, index=False, float_format="%.6f")

        return out_path

    def export_cl_distribution(self, output_dir=OUTPUT_DIR, half_span_only=True):
        """Export this wing's spanwise Cl distribution to a CSV file. Requires `run()` first."""
        return self._export_distribution(self.cl_dist, "cl_span", "cl", output_dir, half_span_only)

    def export_cd_distribution(self, output_dir=OUTPUT_DIR, half_span_only=True):
        """Export this wing's spanwise Cd distribution to a CSV file. Requires `run()` first."""
        return self._export_distribution(self.cd_dist, "cd_span", "cd", output_dir, half_span_only)

    def export_gamma_distribution(self, output_dir=OUTPUT_DIR, half_span_only=True):
        """Export this wing's spanwise circulation Γ distribution to a CSV file. Requires `run()` first."""
        return self._export_distribution(self.gamma_dist, "gamma_span", "gamma", output_dir, half_span_only)

    def export_clg_distribution(self, output_dir=OUTPUT_DIR, half_span_only=True):
        """Export this wing's spanwise circulation-derived Cl (clg) distribution to a CSV file. Requires `run()` first."""
        return self._export_distribution(self.clg_dist, "clg_span", "clg", output_dir, half_span_only)

    # -------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------

    @staticmethod
    def _plot_coeff_results(results, coeff_key, coeff_label, overall_key, overall_label,
                             unit="", ax=None, savepath=None, airfoil_name=""):
        """
        Plot one or more spanwise coefficient distributions on the same axes.

        Parameters
        ----------
        results : list of dict
            Each dict must have keys:
                "y_span"    : np.ndarray  spanwise coordinates [m]
                coeff_key   : np.ndarray  local coefficient values
                "span"      : float       full span [m] (for eta normalization)
                "label"     : str         legend label
            Optional key `overall_key` (overall wing coefficient) is shown
            in the label if present.
        coeff_key : str
            Key holding the per-station coefficient array, e.g. "cl_span" or "cd_span".
        coeff_label : str
            Axis/title label for the local coefficient, e.g. "$C_l$" or "$C_d$".
        overall_key : str or None
            Key holding the overall wing coefficient, e.g. "CL" or "CD". If
            None (or absent from a given result dict), no overall value is
            appended to that result's legend label.
        overall_label : str or None
            Legend label for the overall wing coefficient, e.g. "$C_L$" or "$C_D$".
        unit : str, optional
            Unit suffix appended to the y-axis label only, e.g. " [m²/s]".
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure/axes is created.
        savepath : str, optional
            If given, saves the figure to this path.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))

        for res in results:
            eta = res["y_span"] / (res["span"] / 2)
            label = res["label"]
            if overall_key in res:
                label += f"  ({overall_label}={res[overall_key]:.3f})"
            ax.plot(eta, res[coeff_key], lw=2, label=label)

        ax.set_xlabel("η = 2y/b  (spanwise position)", fontsize=12)
        ax.set_ylabel(f"Local {coeff_label}{unit}", fontsize=12)
        ax.set_title(f"Spanwise {coeff_label} Distribution for {airfoil_name} airfoil", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.set_xlim(-1, 1)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()
        if savepath:
            plt.savefig(savepath, dpi=150)

        return ax

    # Per-coefficient differences fed into `_plot_coeff_results`; keys match
    # the `name` argument of `_plot_distribution` / `_plot_comparison` below.
    _COEFF_SPECS = {
        "cl":    dict(coeff_key="cl_span", coeff_label="$C_l$",
                       overall_key="CL", overall_label="$C_L$", unit=""),
        "cd":    dict(coeff_key="cd_span", coeff_label="$C_d$",
                       overall_key="CD", overall_label="$C_D$", unit=""),
        "gamma": dict(coeff_key="gamma_span", coeff_label="$\\Gamma$",
                       overall_key=None, overall_label=None, unit=" [m²/s]"),
        "clg":   dict(coeff_key="clg_span", coeff_label="$C_{l,\\Gamma}$",
                       overall_key=None, overall_label=None, unit=""),
    }

    def _plot_distribution(self, dist_attr, name, ax=None, savepath=None):
        """Plot this wing's spanwise `name` distribution (`dist_attr` attribute). Requires `run()` first."""
        dist = getattr(self, dist_attr)
        if dist is None:
            raise RuntimeError("Call `run()` before plotting results.")
        return self._plot_coeff_results(
            [dist], ax=ax, savepath=savepath, airfoil_name=self.airfoil_name, **self._COEFF_SPECS[name]
        )

    @staticmethod
    def _plot_comparison(analyses, dist_attr, name, ax=None, savepath=None):
        """Overlay the spanwise `name` distribution (`dist_attr` attribute) for multiple `WingAnalysis` instances."""
        results = []
        for wa in analyses:
            dist = getattr(wa, dist_attr)
            if dist is None:
                raise RuntimeError(f"Call `run()` on '{wa.label}' before plotting.")
            results.append(dist)
        airfoil_name = analyses[0].airfoil_name if analyses else ""
        return WingAnalysis._plot_coeff_results(
            results, ax=ax, savepath=savepath, airfoil_name=airfoil_name, **WingAnalysis._COEFF_SPECS[name]
        )

    def plot_cl_distribution(self, ax=None, savepath=None):
        """Plot this wing's spanwise Cl distribution. Requires `run()` first."""
        return self._plot_distribution("cl_dist", "cl", ax=ax, savepath=savepath)

    def plot_cd_distribution(self, ax=None, savepath=None):
        """Plot this wing's spanwise Cd distribution. Requires `run()` first."""
        return self._plot_distribution("cd_dist", "cd", ax=ax, savepath=savepath)

    def plot_gamma_distribution(self, ax=None, savepath=None):
        """Plot this wing's spanwise circulation Γ distribution. Requires `run()` first."""
        return self._plot_distribution("gamma_dist", "gamma", ax=ax, savepath=savepath)

    def plot_clg_distribution(self, ax=None, savepath=None):
        """Plot this wing's spanwise circulation-derived Cl (clg) distribution. Requires `run()` first."""
        return self._plot_distribution("clg_dist", "clg", ax=ax, savepath=savepath)

    @staticmethod
    def plot_comparison(analyses, ax=None, savepath=None):
        """Overlay spanwise Cl distributions for multiple `WingAnalysis` instances."""
        return WingAnalysis._plot_comparison(analyses, "cl_dist", "cl", ax=ax, savepath=savepath)

    @staticmethod
    def plot_cd_comparison(analyses, ax=None, savepath=None):
        """Overlay spanwise Cd distributions for multiple `WingAnalysis` instances."""
        return WingAnalysis._plot_comparison(analyses, "cd_dist", "cd", ax=ax, savepath=savepath)

    @staticmethod
    def plot_gamma_comparison(analyses, ax=None, savepath=None):
        """Overlay spanwise Γ distributions for multiple `WingAnalysis` instances."""
        return WingAnalysis._plot_comparison(analyses, "gamma_dist", "gamma", ax=ax, savepath=savepath)

    @staticmethod
    def plot_clg_comparison(analyses, ax=None, savepath=None):
        """Overlay spanwise circulation-derived Cl (clg) distributions for multiple `WingAnalysis` instances."""
        return WingAnalysis._plot_comparison(analyses, "clg_dist", "clg", ax=ax, savepath=savepath)

    def plot_panels(self, ax=None, savepath=None, show_collocation_points=True, elev=25, azim=-50):
        """
        Visualize the VLM panel mesh (quads) and collocation points for
        this wing using matplotlib (no extra dependencies beyond matplotlib).

        Requires `run()` first.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.Axes3D, optional
            3D axes to plot on. If None, a new figure/axes is created.
        savepath : str, optional
            If given, saves the figure to this path.
        show_collocation_points : bool
            If True, overlays collocation points as red markers.
        elev, azim : float
            Initial 3D view angles [deg].

        Returns
        -------
        ax : mpl_toolkits.mplot3d.Axes3D
        """
        if self.vlm is None:
            raise RuntimeError("Call `run()` before plotting the panel mesh.")

        vlm = self.vlm
        fl = vlm.front_left_vertices
        fr = vlm.front_right_vertices
        br = vlm.back_right_vertices
        bl = vlm.back_left_vertices
        cp = vlm.collocation_points

        # Each panel = quad [front-left, front-right, back-right, back-left]
        quads = np.stack([fl, fr, br, bl], axis=1)  # (n_panels, 4, 3)

        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="3d")

        poly = Poly3DCollection(
            quads, facecolor="lightblue", edgecolor="k", linewidths=0.5, alpha=0.6,
        )
        ax.add_collection3d(poly)

        if show_collocation_points:
            ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2],
                        color="red", s=10, label="Collocation points")
            ax.legend()

        ax.set_xlabel("x (chordwise)")
        ax.set_ylabel("y (spanwise)")
        ax.set_zlabel("z")
        ax.set_title("VLM Panel Mesh")

        # Equal aspect ratio across all axes
        all_pts = np.vstack([fl, fr, br, bl, cp])
        max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2
        mid = all_pts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=150)

        return ax

    def plot_velocity_field(self, points, scale=1.0, ax=None, savepath=None,
                                     arrow_color="tab:blue", linewidth=1.2, head_size=6,
                                     velocity_component=None, velocity="induced", plane="yz"):
        """
        Plot the velocity field induced by the wing (bound + trailing
        vortices) at a set of field points, as arrows in a Y-Z or X-Z
        plane.

        Arrows are drawn with `FancyArrowPatch` rather than `quiver`:
        `quiver` builds each arrowhead as a rotated polygon mixing
        shaft-width and data-length units and auto-shrinks the head for
        short vectors, which distorts/skews heads across a field of
        varying magnitudes. `FancyArrowPatch` instead places a
        constant-size head (`mutation_scale`, a display-space size in
        points) at the tip of a straight path, so head size and shaft
        linewidth stay identical for every arrow and only the path
        length encodes magnitude.

        Requires `run()` first.

        Parameters
        ----------
        points : np.ndarray
            Nx3 array of field points [m] (geometry axes) at which to
            evaluate the velocity, e.g. built from a Y-Z meshgrid at a
            fixed x (`plane="yz"`), or an X-Z meshgrid at a fixed
            spanwise station y (`plane="xz"`).
        scale : float
            Multiplier applied to the velocity vectors before drawing
            them as arrows. Increase to make arrows longer, decrease to
            make them shorter.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure/axes is created.
        savepath : str, optional
            If given, saves the figure to this path.
        arrow_color : str
            Color of the arrows.
        linewidth : float
            Shaft linewidth [pt], constant for all arrows.
        head_size : float
            Arrowhead size (`mutation_scale`) [pt], constant for all arrows.
        velocity_component : {None, "x", "u", "y", "v", "z", "w"}, optional
            If None (default), plot the full velocity vector (its
            in-plane projection, since arrows are drawn in a 2D plane).
            If given, the other two components are zeroed out first so
            only that component is plotted, e.g. "z"/"w" isolates the
            vertical (downwash/upwash) component. Raises `ValueError`
            if the requested component is normal to `plane` (e.g.
            "x"/"u" with `plane="yz"`), since it can't be drawn as an
            in-plane arrow.
        velocity : {"induced", "total"}
            "induced" (default) plots only the wing-induced velocity
            (`vlm.get_induced_velocity_at_points`); "total" plots
            freestream + induced (`vlm.get_velocity_at_points`).
        plane : {"yz", "xz"}
            Plane to plot arrows in. "yz" (default) plots spanwise (y)
            vs. vertical (z) at a fixed x. "xz" plots chordwise (x) vs.
            vertical (z) at a fixed spanwise station y — `points` should
            share a common y value in this case.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if self.vlm is None:
            raise RuntimeError("Call `run()` before computing induced velocities.")

        if plane not in PLANE_AXES:
            raise ValueError(f"Unknown plane '{plane}'. Choose from {list(PLANE_AXES)}.")
        h_idx, v_idx, h_label, v_label = PLANE_AXES[plane]

        points = np.array(points)

        if velocity == 'induced':
            V = self.vlm.get_induced_velocity_at_points(points)
        else:
            V = self.vlm.get_velocity_at_points(points)

        component = velocity_component.lower() if velocity_component is not None else None
        if component is not None:
            if component not in VELOCITY_COMPONENTS:
                raise ValueError(
                    f"Unknown velocity_component '{velocity_component}'. "
                    f"Choose from 'x'/'u', 'y'/'v', 'z'/'w', or None for the full vector."
                )
            component_idx = VELOCITY_COMPONENTS[component]
            if component_idx not in (h_idx, v_idx):
                normal_names = "/".join(k for k, i in VELOCITY_COMPONENTS.items() if i == component_idx)
                raise ValueError(
                    f"velocity_component '{velocity_component}' ({normal_names}) is normal to "
                    f"this plot's '{plane}' plane and can't be drawn as an in-plane arrow."
                )
            mask = np.zeros(3)
            mask[component_idx] = 1.0
            V = V * mask

        h = points[:, h_idx]
        vert = points[:, v_idx]
        vh = V[:, h_idx] * scale
        vv = V[:, v_idx] * scale

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))

        for hi, zi, vhi, vvi in zip(h, vert, vh, vv):
            arrow = FancyArrowPatch(
                (hi, zi), (hi + vhi, zi + vvi),
                arrowstyle="-|>", mutation_scale=head_size,
                linewidth=linewidth, color=arrow_color, shrinkA=0, shrinkB=0,
            )
            ax.add_patch(arrow)

        if plane == "yz":
            half_span = self.span / 2
            ax.plot([-half_span, half_span], [0, 0], color="k", lw=3, alpha=0.8, label="Wing")
        else:
            y_station = points[0, 1]
            x_le, chord = self._local_chord_le(y_station)
            ax.plot([x_le, x_le + chord], [0, 0], color="k", lw=3, alpha=0.8, label=f"Chord (y={y_station:.2f} m)")

        all_h = np.concatenate([h, h + vh])
        all_v = np.concatenate([vert, vert + vv])
        margin_h = 0.05 * (all_h.max() - all_h.min())
        margin_v = 0.05 * (all_v.max() - all_v.min())
        ax.set_xlim(all_h.min() - margin_h, all_h.max() + margin_h)
        ax.set_ylim(all_v.min() - margin_v, all_v.max() + margin_v)

        component_label = VELOCITY_COMPONENT_LETTERS.get(component_idx) if component is not None else None
        title_suffix = f" ({component_label}-component)" if component_label else ""
        velocity_label = "Induced" if velocity == "induced" else "Total"

        ax.set_xlabel(h_label, fontsize=12)
        ax.set_ylabel(v_label, fontsize=12)
        ax.set_title(f"{velocity_label} Velocity Field{title_suffix} — {self.label}", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=10)

        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=150)
        plt.show()

        return ax

    def show_panels_plotly(self, show=True, **draw_kwargs):
        """
        Visualize the VLM panel mesh using AeroSandbox's built-in interactive
        Plotly viewer.

        Shows panels, collocation points, vortex filaments, and (after a run)
        pressure/Cp coloring. Requires `run()` first.

        Parameters
        ----------
        show : bool
            If True, opens the interactive plot (in browser or notebook).
        **draw_kwargs :
            Extra keyword arguments passed through to `vlm.draw()`,
            e.g. `c=vlm.forces_geometry[:, 2]` to color panels by lift force.

        Notes
        -----
        Requires `plotly` (pip install plotly).
        """
        if self.vlm is None:
            raise RuntimeError("Call `run()` before viewing the panel mesh.")
        return self.vlm.draw(show=show, backend="plotly", **draw_kwargs)



