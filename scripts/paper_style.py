"""Shared figure style for the Bio+MedVis Challenge submission.

One place for the rules every figure obeys, so the five figures read as one set:
print-sized, every axis named with units, ticks outward, no chrome, colourblind-safe
hues. Sizes are the VGTC conference column metrics (3.33 in single, 7.0 in double).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

COL1, COL2 = 3.33, 7.0          # VGTC single- and double-column widths, inches

# ---- palette -------------------------------------------------------------
# Systems: the free peptide is the neutral reference, the two drugs take the
# blue/orange pair (the most robust dichromat-safe contrast available).
SYS_COL = {"free": "#4d545c", "TMP": "#2a78d6", "SPA": "#d1552a"}
SYS_LAB = {"free": u"Aβ42 (free)", "TMP": u"Aβ42 + TMP", "SPA": u"Aβ42 + SPA"}
SYS_KEYS = ["free", "TMP", "SPA"]
# Markov states keep the tool's hues so figures and system agree
STATE_COL = ["#2a78d6", "#1a9e6f", "#d99400", "#7a5cf0"]
# Abeta42 regions that matter for aggregation (1-indexed, inclusive)
REGIONS = [("N-terminus", 1, 15, "#9aa3b2"),
           ("KLVFF / CHC", 16, 21, "#d1332e"),
           ("D23-K28 turn", 22, 28, "#7a5cf0"),
           ("C-terminus", 30, 42, "#1a9e6f")]
BASIC = set("RK")           # Arg / Lys, the residues a dianion can salt-bridge
ACIDIC = set("DE")

SEQ = LinearSegmentedColormap.from_list(
    "seqblue", ["#f7fafe", "#dbe9fb", "#b3d1f6", "#7fb0ec", "#4a8ce0", "#2a6cbf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "divbr", ["#a11d1a", "#d1332e", "#eda49f", "#f2f1ee", "#9ec5f4", "#2a78d6", "#12518f"])
CONTACT = LinearSegmentedColormap.from_list(
    "contact", ["#d7dade", "#e9b6ab", "#dd7f66", "#c9472f", "#9e1f13"])
ELEM_COL = {"C": "#4d545c", "N": "#2a78d6", "O": "#d1332e", "S": "#d9a300", "H": "#a8b0b8"}


def use_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.labelsize": 7.6,
        "axes.titlesize": 7.8,
        "axes.titleweight": "bold",
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#3a3f45",
        "axes.labelcolor": "#15181b",
        "text.color": "#15181b",
        "xtick.color": "#3a3f45", "ytick.color": "#3a3f45",
        "xtick.labelcolor": "#15181b", "ytick.labelcolor": "#15181b",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.2,
        "grid.color": "#e3e6e9",
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed real TrueType, not paths
        # Arial has no subscript/superscript glyphs, so chemical formulae go through
        # mathtext; point mathtext at Arial so they still match the surrounding text.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.default": "regular",
    })


def load():
    j = lambda f: json.load(open(os.path.join(DATA, f), encoding="utf-8"))
    return j("track1.json"), j("track1_struct.json"), j("track1_struct2.json"), j("track1_pocket.json")


def panel_label(ax, letter, dx=-0.085, dy=1.06, size=9):
    """Bold panel letter in figure convention, placed outside the axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="top", ha="left")


def region_bands(ax, y0=None, y1=None, alpha=0.10, label_y=None, fontsize=5.8, show_labels=True):
    """Shade the aggregation-relevant sequence regions behind a residue axis."""
    lo, hi = ax.get_ylim() if y0 is None else (y0, y1)
    for name, a, b, c in REGIONS:
        ax.axvspan(a - 0.5, b + 0.5, color=c, alpha=alpha, lw=0, zorder=0)
        if show_labels:
            ax.text((a + b) / 2, label_y if label_y is not None else hi,
                    name, ha="center", va="bottom", fontsize=fontsize, color=c)


def pca_basis(P):
    """Two principal axes of a point set, so a 3D trace is shown at maximum spread."""
    P = np.asarray(P, float)
    m = P.mean(0)
    C = np.cov((P - m).T)
    w, v = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    return m, v[:, order[0]], v[:, order[1]]


def project(P, basis):
    m, e1, e2 = basis
    D = np.asarray(P, float) - m
    return np.column_stack([D @ e1, D @ e2])


def scalebar(ax, length=10.0, label=u"10 Å", loc=(0.62, 0.04), lw=1.1, fontsize=6.2):
    """Explicit length reference for the structure panels, which have no axes."""
    x0, y0 = ax.transLimits.inverted().transform(loc)
    ax.plot([x0, x0 + length], [y0, y0], color="#3a3f45", lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=8)
    for xx in (x0, x0 + length):
        ax.plot([xx, xx], [y0 - length * 0.05, y0 + length * 0.05], color="#3a3f45",
                lw=lw, clip_on=False, zorder=8)
    ax.text(x0 + length / 2, y0 + length * 0.10, label, ha="center", va="bottom",
            fontsize=fontsize, color="#3a3f45", zorder=8)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    plt.close(fig)
    p = os.path.join(FIGS, name + ".png")
    print(f"  {name}: {os.path.getsize(p)/1024:.0f} KB png + pdf")
