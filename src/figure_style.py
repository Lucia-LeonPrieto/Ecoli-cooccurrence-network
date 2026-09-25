"""
figure_style  (vendored into this repository — src/figure_style.py)
==================================================================
Universal matplotlib style for LaTeX thesis and journal figures.
Works identically on Windows, macOS and Linux.
Arial is installed automatically if not present (Linux / CI).

IMPORT IN EVERY SCRIPT
-----------------------
    from figure_style import *
    init_style()

QUICK EXAMPLE
-------------
    init_style()

    fig, ax = plt.subplots(figsize=figsize(W_70, H_STD), layout='constrained')
    ax.plot(x, y, label="signal")
    ax.legend()
    style(ax, xlabel="Time (s)", ylabel="Amplitude")
    save(fig, "figures/my_figure")        # → my_figure.pdf

    # Three panels for LaTeX subfigure{0.32\\textwidth}
    for label, data in panels.items():
        fig, ax = plt.subplots(figsize=figsize(W_THIRD, H_STD), layout='constrained')
        ax.scatter(data["x"], data["y"])
        style(ax, xlabel="X", ylabel="Y")
        save(fig, f"figures/panel_{label}")
        plt.close(fig)

LATEX INSERTION
---------------
The figsize already encodes the physical size, so always insert at
\\textwidth or \\linewidth — never with arbitrary percentages:

    \\includegraphics[width=0.70\\textwidth]{figures/my_figure}
    \\includegraphics[width=\\linewidth]{figures/panel_A}  % inside subfigure
"""

import os
import platform
import subprocess
import importlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
# Measure your \\textwidth with: \\the\\textwidth  (compile and read the output)
# Convert: LaTeX pt × 0.3515 = mm
# This document: 472.03 pt × 0.3515 = 165.9 mm

_MM = 1 / 25.4          # mm → inches (matplotlib unit)

TW_MM   = 165.9         # ← your \\textwidth. Update if you change template.

W_FULL  = TW_MM         # \\includegraphics[width=\\textwidth]
W_70    = 0.70 * TW_MM  # \\includegraphics[width=0.70\\textwidth]
W_HALF  = 0.48 * TW_MM  # subfigure{0.48\\textwidth}  — 2-panel row
W_HALF_XS = 0.4 * TW_MM
W_THIRD = 0.32 * TW_MM  # subfigure{0.32\\textwidth}  — 3-panel row
W_QRTR  = 0.23 * TW_MM  # subfigure{0.23\\textwidth}  — 4-panel row

H_STD  = 65.0           # general purpose height (mm)
H_WIDE = 50.0           # short wide panel, e.g. barplot
H_TALL = 90.0           # tall panel, e.g. heatmap


# ─────────────────────────────────────────────────────────────────────────────
# TYPOGRAPHY  —  absolute pt values in the final PDF
# ─────────────────────────────────────────────────────────────────────────────
# Body text in the thesis: ~11 pt.
# Figures at 8–9 pt: clearly smaller, still legible on printed A4.
# These meet Frontiers' minimum (8 pt) and match high-impact journal style.

TICK        = 6     # tick labels
LABEL       = 7     # xlabel, ylabel
LEGEND      = 6     # legend entries
TITLE       = 7     # panel title (prefer LaTeX subfigure captions)
ANNOT       = 6     # in-plot text / stat markers

SPINE_LW    = 0.8   # spine linewidth (pt)
TICK_W      = 0.8   # tick mark width (pt)
TICK_LEN    = 3.5   # tick mark length (pt)
GRID_LW     = 0.5   # grid line width
GRID_ALPHA  = 0.45  # grid opacity


# ─────────────────────────────────────────────────────────────────────────────
# FONT DETECTION AND INSTALLATION
# ─────────────────────────────────────────────────────────────────────────────
def _get_arial():
    """
    Returns the best available sans-serif font name, preferring Arial.

    On Windows and macOS, Arial is always present.
    On Linux (including Colab), attempts to install the Microsoft core fonts
    package (which includes Arial). Falls back to Liberation Sans
    (metrically identical to Arial, open-source) if installation fails.

    Called automatically by init_style(). You do not need to call this directly.
    """
    def _available_fonts():
        return {f.name for f in fm.fontManager.ttflist}

    # Arial already available — nothing to do
    if "Arial" in _available_fonts():
        return "Arial"

    # Linux / Colab: try to install Microsoft core fonts
    if platform.system() == "Linux":
        print("  [figure_style] Arial not found. Attempting installation...")
        try:
            # Accept the EULA non-interactively
            subprocess.run(
                ["bash", "-c",
                 "echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula"
                 " select true | debconf-set-selections"],
                capture_output=True, timeout=15
            )
            result = subprocess.run(
                ["apt-get", "install", "-y", "ttf-mscorefonts-installer"],
                capture_output=True, timeout=120
            )
            if result.returncode == 0:
                # Rebuild matplotlib font cache so it picks up the new fonts
                cache_dir = mpl.get_cachedir()
                for f in os.listdir(cache_dir):
                    if f.startswith("fontlist"):
                        os.remove(os.path.join(cache_dir, f))
                importlib.reload(fm)
                if "Arial" in _available_fonts():
                    print("  [figure_style] Arial installed successfully.")
                    return "Arial"
        except Exception as e:
            print(f"  [figure_style] Installation failed: {e}")

        # Fallback: Liberation Sans (open-source, metrically identical to Arial)
        if "Liberation Sans" in _available_fonts():
            print("  [figure_style] Using Liberation Sans (Arial substitute).")
            return "Liberation Sans"

    # Last resort: matplotlib default
    print("  [figure_style] Using DejaVu Sans (default fallback).")
    return "DejaVu Sans"


# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION  —  call once at the top of every script or notebook
# ─────────────────────────────────────────────────────────────────────────────
def init_style():
    """
    Sets global matplotlib rcParams for publication-quality figures.

    - Detects and installs Arial automatically (Linux / Colab).
    - Sets all font sizes as absolute pt values.
    - Configures spines, ticks, and legend defaults.
    - Sets savefig defaults (300 dpi, tight bbox).

    Call once per script/notebook, before any plt.subplots() call.
    """
    font = _get_arial()
    sns.set_style("white")

    mpl.rcParams.update({
        # ── Font ──────────────────────────────────────────────────────────────
        "font.family":          "sans-serif",
        "font.sans-serif":      [font, "Helvetica Neue", "Helvetica",
                                 "Liberation Sans", "DejaVu Sans"],
        "font.size":            LABEL,
        "pdf.fonttype":         42,     # TrueType embedding — required by journals
        "ps.fonttype":          42,     # avoids Type 3 bitmap fonts in PDF

        # ── Axes ──────────────────────────────────────────────────────────────
        "axes.labelsize":       LABEL,
        "axes.titlesize":       TITLE,
        "axes.linewidth":       SPINE_LW,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            False,
        "axes.labelpad":        4,

        # ── Ticks ─────────────────────────────────────────────────────────────
        "xtick.labelsize":      TICK,
        "ytick.labelsize":      TICK,
        "xtick.major.width":    TICK_W,
        "ytick.major.width":    TICK_W,
        "xtick.major.size":     TICK_LEN,
        "ytick.major.size":     TICK_LEN,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.color":          "black",
        "ytick.color":          "black",

        # ── Legend ────────────────────────────────────────────────────────────
        "legend.fontsize":      LEGEND,
        "legend.frameon":       False,
        "legend.handlelength":  1.2,
        "legend.handletextpad": 0.5,
        "legend.labelspacing":  0.4,
        "legend.borderpad":     0.3,

        # ── Lines ─────────────────────────────────────────────────────────────
        "lines.linewidth":      1.2,

        # ── Saving ────────────────────────────────────────────────────────────
        "figure.dpi":           150,    # screen preview only
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",  # applied globally via rcParams — sin esto,
                                           # save() para fmt="png"/"tiff" (que no pasan
                                           # bbox_inches explícito, a diferencia de "pdf")
                                           # no recortaba nunca, y podía cortar labels
                                           # rotados 90° que sobresalen del área del eje.
        "savefig.pad_inches":   0.01,    # 0.25 mm safety border — distortion < 0.5%
    })

    print(f"  [figure_style] Ready. Font: {font}. "
          f"Labels: {LABEL} pt, Ticks: {TICK} pt.")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE SIZE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def figsize(width_mm, height_mm=None):
    """
    Returns (width_in, height_in) for plt.subplots(figsize=...).

    Parameters
    ----------
    width_mm  : float — use a W_* constant or any custom mm value
    height_mm : float — use an H_* constant or any custom mm value.
                        None → square panel.

    Examples
    --------
    fig, ax = plt.subplots(figsize=figsize(W_70, H_STD), layout='constrained')
    fig, ax = plt.subplots(figsize=figsize(W_THIRD, 50), layout='constrained')
    fig, ax = plt.subplots(figsize=figsize(W_HALF), layout='constrained')
    """
    w = width_mm * _MM
    h = w if height_mm is None else height_mm * _MM
    return (w, h)


# ─────────────────────────────────────────────────────────────────────────────
# STYLE FUNCTION  —  one function for every panel size
# ─────────────────────────────────────────────────────────────────────────────
def style(ax, xlabel="", ylabel="", title="", grid=False, grid_axis="both"):
    """
    Applies publication style to an axes object.

    Font sizes are absolute pt values and do not depend on panel size.
    Use the same call for single panels and subfigure panels alike.

    Parameters
    ----------
    ax        : matplotlib Axes
    xlabel    : str  — x-axis label
    ylabel    : str  — y-axis label
    title     : str  — panel title (prefer LaTeX subfigure captions)
    grid      : bool — adds a light grid; False by default (Nature / Cell style)
    grid_axis : str  — "both" | "x" | "y"
    """
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(SPINE_LW)

    ax.tick_params(axis="both", which="both", colors="black")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if grid:
        ax.grid(True, axis=grid_axis, linestyle="-",
                linewidth=GRID_LW, color="#CCCCCC",
                alpha=GRID_ALPHA, zorder=0)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, path, fmt="pdf", dpi=300):
    """
    Saves a figure at publication quality.

    Parameters
    ----------
    fig  : matplotlib Figure
    path : str — output path WITHOUT extension
    fmt  : "pdf"  → vector, best for LaTeX thesis
           "tiff" → lossless raster, required by Frontiers for submission
           "png"  → acceptable for LaTeX, lighter than tiff
    dpi  : int — 300 minimum; 600 for figures with very fine lines

    bbox_inches='tight' and pad_inches=0.01 are set globally in rcParams
    by init_style(), so save() stays clean. The 0.01 in (0.25 mm) safety
    border prevents any label clipping and introduces < 0.5% size distortion
    — well below the threshold of visual perception.
    Use layout='constrained' in plt.subplots() for complex figures with
    rotated labels, colorbars, or external legends.
    """
    kwargs = {"dpi": dpi}
    if fmt == "pdf":
        kwargs["bbox_inches"] = "tight"
        kwargs["pad_inches"]  = 0.02
    if fmt == "tiff":
        kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}

    output = f"{path}.{fmt}"
    fig.savefig(output, format=fmt, **kwargs)
    w, h = fig.get_size_inches()
    print(f"  Saved → {output}  [{w*25.4:.1f} × {h*25.4:.1f} mm @ {dpi} dpi]")


# ─────────────────────────────────────────────────────────────────────────────
# GRID SHORTCUTS
# ─────────────────────────────────────────────────────────────────────────────
def soft_grid_y(ax):
    """Horizontal grid for barplots and boxplots. Renders below the bars."""
    ax.grid(True, axis="y", linestyle="-", linewidth=GRID_LW,
            color="#CCCCCC", alpha=GRID_ALPHA, zorder=0)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def info():
    """Prints all constants. Useful as a reminder."""
    print(f"\n  Document geometry (TW_MM = {TW_MM} mm):")
    for name, val in [("W_FULL", W_FULL), ("W_70", W_70),
                      ("W_HALF", W_HALF), ("W_THIRD", W_THIRD),
                      ("W_QRTR", W_QRTR)]:
        print(f"    {name:8s} = {val:.1f} mm")
    print(f"\n  Typography:")
    print(f"    Tick labels : {TICK} pt")
    print(f"    Axis labels : {LABEL} pt")
    print(f"    Legend      : {LEGEND} pt")
    print(f"    Annotations : {ANNOT} pt")