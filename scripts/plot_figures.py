"""Rebuild figures 3, 7, 8 and 9 from the portable processed-data folder.

Figures 7--9 retain the manuscript's data and visual conventions. Figure 3 is
a new rendering of recovered vector-marker coordinates, not instrument data.
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image

from common import ROOT, load_parameters, read_csv

WIDTH = 6.5
INK = "#171c21"
PALETTE = ["#0B6E6E", "#2E9AA6", "#7CC8C7", "#B9E4C9", "#DDEFBA", "#C6B6E9", "#A785D6", "#7A5FB3"]
STATES = [PALETTE[k] for k in (3, 2, 1, 5, 6, 7)]
CURVES = [PALETTE[k] for k in (0, 1, 2, 3, 5, 6, 7)]
SEVERITY = LinearSegmentedColormap.from_list("severity", ["#f7fafc", "#b6d7eb", "#398fbd", "#194d83", "#262137"])


def typography(requested):
    system_fonts = Path("C:/Windows/Fonts")
    if system_fonts.exists():
        for name in ("times.ttf", "timesi.ttf", "timesbd.ttf", "timesbi.ttf"):
            fm.fontManager.addfont(str(system_fonts / name))
    try:
        fm.findfont(requested, fallback_to_default=False)
        font = requested
    except ValueError:
        font = "DejaVu Serif"
        warnings.warn(f"{requested} is unavailable; using {font}. Numerical content is unchanged.")
    plt.rcParams.update({"font.family": font, "font.size": 11,
        "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.it": font + ":italic",
        "mathtext.bf": font + ":bold", "mathtext.bfit": font + ":bold:italic",
        "mathtext.sf": font, "mathtext.tt": font, "mathtext.cal": font + ":italic",
        "mathtext.fallback": None, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.labelsize": 11, "axes.titlesize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.3, "lines.markersize": 4.4, "xtick.direction": "out", "ytick.direction": "out",
        "figure.facecolor": "white", "savefig.facecolor": "white"})
    return font


def box(fig, x, y, width, height):
    return fig.add_axes([x / WIDTH, y / fig.get_figheight(), width / WIDTH, height / fig.get_figheight()])


def panel(ax, key, title):
    ax.set_title(f"({key}) {title}", loc="left", pad=9, fontsize=12)


def format_map(ax, rows):
    energy = sorted({float(r["energy_mJ"]) for r in rows})
    concentration = sorted({float(r["concentration_molL"]) for r in rows})
    assert min(energy) > 0 and min(concentration) > 0
    labels = [r"$10^{-5}$", r"$3.16\times10^{-5}$", r"$10^{-4}$", r"$3.16\times10^{-4}$",
              r"$10^{-3}$", r"$3.16\times10^{-3}$", r"$10^{-2}$"]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(4.8, 210); ax.set_ylim(6e-6, 1.55e-2)
    ax.set_xticks(energy); ax.set_xticklabels([f"{v:g}" for v in energy], rotation=90, ha="center")
    ax.set_yticks(concentration); ax.set_yticklabels(labels)
    ax.minorticks_off(); ax.grid(color="#dce3e8", lw=.5, zorder=0)
    ax.set_xlabel(r"$E_p$ (mJ)", labelpad=4)
    ax.set_ylabel(r"Concentration $c$ (mol L$^{-1}$)", labelpad=6)


def save(fig, out, name, groups, audit=False):
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    if audit:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        axes = [ax for group in groups for ax in group]
        ids = {ax: str(i) for i, ax in enumerate(axes)}
        require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=ids,
            row_groups=[[ids[ax] for ax in group] for group in groups],
            json_out=out / f"{name}.alignment.json", strict=True)
    fig.savefig(out / f"{name}.pdf", dpi=400)
    fig.savefig(out / f"{name}.png", dpi=190)
    plt.close(fig)


def states(root, out, audit):
    rows = read_csv(root / "examples/representatives.csv")
    lookup = {r["state_label"]: r for r in rows}
    fig = plt.figure(figsize=(WIDTH, 3.8), dpi=130)
    axes = []
    for k in range(6):
        with Image.open(root / lookup[f"S{k}"]["raw_path"]) as source:
            image = source.convert("RGB").crop((55, 360, 480, 1320))
        ax = box(fig, .10 + k * 1.055, .80, .97, 2.19)
        axes.append(ax); ax.imshow(image); ax.axis("off")
        fig.text((.10 + k * 1.055 + .97 / 2) / WIDTH, .49 / 3.8,
                 f"S{k}", ha="center", fontweight="bold", fontsize=12, color=INK)
        if k == 0:
            ax.plot([24, 24 + 1000 / 5.63847153682089], [54, 54], lw=1.7, color="black")
            ax.text(24, 105, "1 mm", color="black", fontsize=11)
    fig.text(.10 / WIDTH, 3.37 / 3.8, r"Observation delay: $15\ \mu\mathrm{s}$", fontsize=12, color=INK)
    arrow = box(fig, .10, .09, 1.1, .24)
    arrow.set_xlim(0, 1); arrow.set_ylim(0, 1); arrow.axis("off")
    arrow.annotate("", xy=(.94, .5), xytext=(.06, .5), arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#008544"})
    fig.text(1.28 / WIDTH, .115 / 3.8, r"Incident laser, $+y$", fontsize=11, color=INK)
    save(fig, out, "fig07_states", [axes], audit)


def state_map(root, out, audit):
    rows = read_csv(root / "data/per_condition.csv")
    fig = plt.figure(figsize=(WIDTH, 4.67), dpi=130)
    a, b = box(fig, 1.01, 1.25, 1.86, 2.72), box(fig, 3.76, 1.25, 2.41, 2.72)
    for ax in (a, b):
        format_map(ax, rows)
    b.set_ylabel(""); b.tick_params(axis="y", labelleft=False)
    panel(a, "a", "Mean state"); panel(b, "b", "State composition")
    im = a.scatter([float(r["energy_mJ"]) for r in rows], [float(r["concentration_molL"]) for r in rows],
                   c=[float(r["S_bar"]) for r in rows], cmap=SEVERITY, vmin=0, vmax=5, s=93,
                   edgecolor="#35556e", lw=.5, zorder=3)
    cb = fig.colorbar(im, cax=box(fig, 3.03, 1.25, .10, 2.72))
    cb.ax.set_title(r"$\overline{S}$", fontsize=11, pad=8)
    refs = read_csv(root / "data/thermal_reference_curves.csv")
    for key, col, ls in (("energy_boiling_peak_mJ", PALETTE[0], "--"), ("energy_highT_peak_mJ", PALETTE[7], "-.")):
        a.plot([float(r[key]) for r in refs], [float(r["concentration_molL"]) for r in refs], ls=ls, color=col, lw=1.45, zorder=2)
    for row in rows:
        energy, c = float(row["energy_mJ"]), float(row["concentration_molL"])
        start = np.log10(energy) - .067
        for k in range(6):
            end = start + .134 * int(row[f"S{k}_count"]) / int(row["n_frames"])
            if end > start:
                b.add_patch(Rectangle((10**start, c / 1.17), 10**end - 10**start, c * (1.17 - 1/1.17), fc=STATES[k], ec="none", zorder=3))
            start = end
        b.add_patch(Rectangle((energy * 10**-.067, c/1.17), energy * (10**.067 - 10**-.067),
                             c * (1.17 - 1/1.17), fill=False, ec="#344b54", lw=.65, zorder=4))
    degree = chr(176)
    fig.legend(handles=[Line2D([], [], color=PALETTE[0], ls="--", lw=1.5, label=r"Local $T_{\max}=78.4$ " + degree + "C"),
                        Line2D([], [], color=PALETTE[7], ls="-.", lw=1.5, label=r"Local $T_{\max}=190$ " + degree + "C")],
               loc="lower left", bbox_to_anchor=(.10, .025), ncol=1, frameon=False, fontsize=10.5, handlelength=2.8, labelspacing=.4)
    fig.legend(handles=[Patch(fc=col, ec="#344b54", lw=.5, label=f"S{k}") for k, col in enumerate(STATES)],
               loc="lower center", bbox_to_anchor=(.78, .025), ncol=3, frameon=False, fontsize=11,
               handlelength=1.1, columnspacing=.85, handletextpad=.3, labelspacing=.4)
    save(fig, out, "fig08_state_map", [[a, b]], audit)


def pca_bias(root, out, audit):
    rows = read_csv(root / "data/per_condition.csv")
    frames = read_csv(root / "data/per_frame.csv")
    parameters = load_parameters(root)
    fig = plt.figure(figsize=(WIDTH, 7.35), dpi=130)
    a = box(fig, 1.00, 4.40, 1.95, 2.40)
    format_map(a, rows); panel(a, "a", "Mean PC1")
    a.set_ylabel(r"$c$ (mol L$^{-1}$)", labelpad=4)
    sc = a.scatter([float(r["energy_mJ"]) for r in rows], [float(r["concentration_molL"]) for r in rows],
                   c=[float(r["PC1_mean"]) for r in rows], cmap=SEVERITY, s=78, ec="#35556e", lw=.5, zorder=3)
    cb = fig.colorbar(sc, cax=box(fig, 3.10, 4.40, .11, 2.40)); cb.set_label("PC1", labelpad=3)
    b = box(fig, 4.51, 4.40, 1.72, 2.40); panel(b, "b", "PC1 loadings")
    b.barh(range(4), parameters["loadings"], color="#1764a0", height=.45)
    b.set_yticks(range(4)); b.set_yticklabels(["Area", "Lateral span", "Segments", "Components"], fontsize=10.5)
    b.invert_yaxis(); b.set_xlim(0, .69); b.set_xticks([0, .25, .5]); b.set_xlabel("Loading")
    for k, value in enumerate(parameters["loadings"]):
        b.text(value + .018, k, f"{value:.3f}", va="center", fontsize=10)
    c = box(fig, .72, 1.50, 2.26, 1.90); panel(c, "c", "PC1 by state")
    values = [[float(r["PC1"]) for r in frames if r["state_label"] == f"S{k}"] for k in range(6)]
    bp = c.boxplot(values, tick_labels=[f"S{k}" for k in range(6)], whis=[5,95], showfliers=False,
                   patch_artist=True, widths=.55, medianprops={"color": INK, "lw":1.1}, whiskerprops={"lw":.9}, capprops={"lw":.9})
    for patch, color in zip(bp["boxes"], STATES):
        patch.set(fc=color, ec=INK, lw=.7, alpha=1)
    c.set_ylabel("PC1"); c.set_yticks([-1,1,3,5,7]); c.tick_params(axis="x", labelsize=10)
    c.grid(axis="y", color="#dce3e8", lw=.5)
    d = box(fig, 3.84, 1.50, 2.39, 1.90); panel(d, "d", "Transmitted-side area")
    for i, level in enumerate(sorted({float(r["concentration_molL"]) for r in rows})):
        group = sorted([r for r in rows if float(r["concentration_molL"]) == level and r["Ba_mean"] and int(r["n_active"]) >= 10], key=lambda r: float(r["energy_mJ"]))
        if not group:
            continue
        v = np.array([float(r["Ba_mean"]) for r in group])
        lo = np.array([float(r["Ba_ci95_lower"]) for r in group])
        hi = np.array([float(r["Ba_ci95_upper"]) for r in group])
        d.errorbar([float(r["energy_mJ"]) for r in group], v, yerr=np.vstack((v-lo,hi-v)),
                   fmt=["o","s","^","D","v","P","X"][i], color=CURVES[i], ms=4, capsize=2.2,
                   elinewidth=1., label=rf"$10^{{{np.log10(level):g}}}$")
    d.axhline(.5, color=INK, ls="--", lw=1.1, label=r"Equal areas: $B_a=0.5$")
    d.set_xscale("log"); d.set_xticks([13,31.5,56,84,168]); d.set_xticklabels(["13","31.5","56","84","168"], rotation=45, ha="right")
    d.set_ylim(-.02,1.02); d.set_yticks([0,.5,1]); d.set_ylabel(r"$B_a$"); d.set_xlabel(r"$E_p$ (mJ)")
    handles, labels = d.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52,.014), ncol=4,
               title=r"Panel (d): $c$ (mol L$^{-1}$)", frameon=False, fontsize=10,
               handlelength=1.7, handletextpad=.3, columnspacing=.8, labelspacing=.25)
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_marker() not in ("None","none","",None):
                line.set_markeredgecolor("#314b52"); line.set_markeredgewidth(.55)
    save(fig, out, "fig09_pca_bias", [[a,b],[c,d]], audit)


def transmittance(root, out, audit):
    rows = read_csv(root / "data/transmittance_recovered.csv")
    fig = plt.figure(figsize=(WIDTH, 5.25), dpi=130)
    axes = []
    for i, letter in enumerate("abc"):
        ax = box(fig, .60 + 2.08 * i, 2.55, 1.57, 2.07); axes.append(ax)
        group = [r for r in rows if r["panel"] == letter]
        series_key = "path_length_mm" if letter == "a" else "energy_mJ"
        x_key = "energy_mJ" if letter == "a" else "path_length_mm"
        for k, setting in enumerate(sorted({float(r[series_key]) for r in group})):
            points = sorted([r for r in group if float(r[series_key]) == setting], key=lambda r: float(r[x_key]))
            ax.plot([float(r[x_key]) for r in points], [float(r["transmittance"]) for r in points],
                    marker=["o","s","^","D","v","P","X","<",">"][k], ms=3.8, lw=1.05,
                    color=(CURVES + [PALETTE[4], "#505050"])[k],
                    label=f"{setting:g} " + ("mm" if letter == "a" else "mJ"))
        panel(ax, letter, rf"$10^{{{-3-i}}}$ mol L$^{{-1}}$")
        ax.set_ylabel(r"$\eta$", labelpad=3)
        ax.set_xlabel(r"$E_p$ (mJ)" if letter == "a" else r"$l$ (mm)")
        ax.set_xlim(0,24 if letter == "a" else 1.8)
        ax.set_ylim(0,.18 if letter == "a" else 1.05)
        ax.set_xticks([0,10,20] if letter == "a" else [0,.6,1.2,1.8])
        ax.set_yticks([0,.06,.12,.18] if letter == "a" else [0,.5,1.])
        ax.grid(axis="y", color="#dce3e8", lw=.5)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0,-.31), fontsize=10,
                  handlelength=1.3, labelspacing=.4, borderaxespad=0)
    save(fig, out, "fig03_transmittance_recovered", [axes], audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/reproduced")
    parser.add_argument("--font", default="Times New Roman")
    parser.add_argument("--figures", nargs="+", choices=["3","7","8","9"], default=["3","7","8","9"])
    parser.add_argument("--audit", action="store_true", help="Optional external audit_panel_alignment helper; not needed for normal use.")
    args = parser.parse_args()
    font = typography(args.font)
    funcs = {"3":transmittance, "7":states, "8":state_map, "9":pca_bias}
    for number in args.figures:
        funcs[number](args.root, args.output_dir, args.audit)
    print(json.dumps({"figures":args.figures, "font":font, "data_source":"processed CSV tables; original raw representative frames"}))
