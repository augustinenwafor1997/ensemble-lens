"""Render the five submission figures at publication quality.

These are print renderings of the same encodings the interactive Ensemble Lens uses;
the tool itself is shown in the abstract teaser. Every panel carries named, ticked
axes with units. Output: figures/figN_*.pdf (vector) and .png (600 dpi).

Run:  py -3.10 make_paper_figures.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker

from paper_style import (use_style, load, save, panel_label, region_bands, pca_basis,
                         project, scalebar, COL1, COL2, SYS_COL, SYS_LAB, SYS_KEYS,
                         STATE_COL, REGIONS, BASIC, ACIDIC, SEQ, DIV, CONTACT, ELEM_COL)

use_style()
T1, ST, S2, PK = load()
K = "3"                                   # the 3-state model is the one we report
NRES = 42
SEQ42 = S2["systems"][0]["seq"]
IDX = {"free": 0, "TMP": 1, "SPA": 2}     # index into T1/S2 systems
PKIDX = {"TMP": 0, "SPA": 1}              # pocket file holds only the two drug systems


def occ(sysname):
    return np.array(T1["systems"][IDX[sysname]]["params"][K]["occupancy"])


def occci(sysname):
    return np.array(T1["systems"][IDX[sysname]]["params"][K]["occCI"])


def states2(sysname):
    return S2["systems"][IDX[sysname]]["params"][K]["states"]


def specific(sysname, state):
    """Contact above the concentration background: observed minus decorrelated null."""
    v = states2(sysname)[str(state)]
    return np.maximum(0.0, np.array(v["drug"]) - np.array(v["drugNull"]))


def ens_specific(sysname):
    """Population-weighted specific contact over the whole ensemble."""
    o, P = occ(sysname), states2(sysname)
    out = np.zeros(NRES)
    for k, v in P.items():
        out += o[int(k)] * specific(sysname, int(k))
    return out


RES_TICKS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 42]


def residue_axis(ax, label=True):
    ax.set_xlim(0.5, NRES + 0.5)
    ax.set_xticks(RES_TICKS)
    if label:
        ax.set_xlabel(u"Aβ42 residue number")
    ax.tick_params(axis="x", labelsize=6.4)


# =====================================================================
# Figure 1 - what the drugs do to the conformational ensemble
# =====================================================================
def fig1():
    fig = plt.figure(figsize=(COL2, 3.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.62, wspace=0.28,
                          left=0.075, right=0.995, top=0.90, bottom=0.10)
    ax = fig.add_subplot(gs[0, :])
    nS = 3
    w = 0.26
    x = np.arange(nS)
    for i, s in enumerate(SYS_KEYS):
        o, ci = occ(s) * 100, occci(s) * 100
        err = np.abs(np.column_stack([o - ci[:, 0], ci[:, 1] - o]).T)
        ax.bar(x + (i - 1) * w, o, w * 0.88, color=SYS_COL[s], label=SYS_LAB[s],
               zorder=3, linewidth=0)
        ax.errorbar(x + (i - 1) * w, o, yerr=err, fmt="none", ecolor="#15181b",
                    elinewidth=0.7, capsize=1.9, capthick=0.7, zorder=4)
        # label above the CI, not above the bar, or the cap collides with the number
        for xx, vv, hi in zip(x + (i - 1) * w, o, ci[:, 1]):
            ax.text(xx, hi + 2.0, f"{vv:.0f}", ha="center", va="bottom", fontsize=6.2,
                    color=SYS_COL[s], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"State {k}" for k in range(nS)])
    ax.set_ylabel("Population\n(% of simulated time)")
    ax.set_xlabel("Markov state (matched across systems)")
    ax.set_ylim(0, 78)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.16))
    panel_label(ax, "a", dx=-0.062, dy=1.30)

    # the same three states, drawn, so a state number means a shape
    P = states2("free")
    for k in range(nS):
        axs = fig.add_subplot(gs[1, k])
        v = P[str(k)]
        ens = np.array(v["ensemble"], float)
        basis = pca_basis(ens[0])
        for fi, fr in enumerate(ens[:5]):
            xy = project(fr, basis)
            axs.plot(xy[:, 0], xy[:, 1], color=STATE_COL[k], lw=1.5 if fi == 0 else 0.65,
                     alpha=1.0 if fi == 0 else 0.32, solid_capstyle="round",
                     solid_joinstyle="round", zorder=3 if fi == 0 else 2)
        axs.set_aspect("equal")
        axs.axis("off")
        axs.margins(0.10)
        axs.set_title(f"State {k}   Rg = {v['rgMean']:.1f} " + u"Å", color=STATE_COL[k],
                      fontsize=7.0, pad=2)
        if k == 0:
            scalebar(axs, 10.0, u"10 Å", loc=(0.02, -0.06))
            panel_label(axs, "b", dx=-0.02, dy=1.24)
    fig.text(0.5, 0.005, u"Cα backbone, five superposed frames per state (free peptide); "
             u"states are matched across systems by CoVAMPnet",
             ha="center", va="bottom", fontsize=6.2, color="#4d545c")
    save(fig, "fig2_ensemble_populations")


# =====================================================================
# Figure 2 - the ligand in its binding site
# =====================================================================
def ligand_groups(elems, xyz):
    """Identify the charged groups so the figure can name the chemistry it depends on."""
    elems = list(elems)
    xyz = np.asarray(xyz, float)
    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    bond = (D < 2.05) & (D > 0.1)
    S = elems.index("S") if "S" in elems else None
    sulf_O = [i for i in range(len(elems)) if elems[i] == "O" and S is not None and bond[S, i]]
    other_O = [i for i in range(len(elems)) if elems[i] == "O" and i not in sulf_O]
    N = elems.index("N") if "N" in elems else None
    carboxyl_C = None
    for i, e in enumerate(elems):
        if e == "C" and sum(1 for o in other_O if bond[i, o]) >= 2:
            carboxyl_C = i
    return dict(S=S, sulf_O=sulf_O, other_O=other_O, N=N, carboxyl_C=carboxyl_C)


def draw_ligand(ax, elems, xyz2, xyz3, ms=52, lw=2.0):
    for i in range(len(xyz3)):
        for j in range(i + 1, len(xyz3)):
            if np.linalg.norm(xyz3[i] - xyz3[j]) < 2.05:
                ax.plot(xyz2[[i, j], 0], xyz2[[i, j], 1], color="#2b3036", lw=lw,
                        zorder=4, solid_capstyle="round")
    for i, e in enumerate(elems):
        ax.scatter(xyz2[i, 0], xyz2[i, 1], s=ms, c=ELEM_COL.get(e, "#4d545c"),
                   edgecolors="white", linewidths=0.7, zorder=5)


def fig2():
    drugs = ["TMP", "SPA"]
    fig = plt.figure(figsize=(COL2, 5.15))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.055],
                          height_ratios=[1, 1, 0.60], hspace=0.10, wspace=0.06,
                          left=0.055, right=0.925, top=0.885, bottom=0.055)
    vmax = 0.0
    for d in drugs:
        for k, v in PK["systems"][PKIDX[d]]["params"][K]["states"].items():
            vmax = max(vmax, max(v["contactFreq"]))
    for c in range(3):
        fig.text(0.055 + (0.925 - 0.055) * ((c + 0.5) / 3.06), 0.905, f"State {c}",
                 ha="center", va="bottom", fontsize=7.6, fontweight="bold",
                 color=STATE_COL[c])
    for r, d in enumerate(drugs):
        ps = PK["systems"][PKIDX[d]]
        P = ps["params"][K]["states"]
        for c, k in enumerate(sorted(P, key=int)):
            v = P[k]
            ax = fig.add_subplot(gs[r, c])
            pose = v["poses"][0]
            ca = np.array(pose["ca"], float)
            dg = np.array(pose["drug"], float)
            pocket = np.array(pose["pocket"], float)
            basis = pca_basis(ca)
            xy, dxy, pxy = project(ca, basis), project(dg, basis), project(pocket, basis)
            cf = np.array(v["contactFreq"], float)
            # pocket lining recedes
            ax.scatter(pxy[:, 0], pxy[:, 1], s=1.1, c="#c9ced4", lw=0, zorder=1)
            # backbone: width and colour both carry contact frequency
            segs = np.stack([xy[:-1], xy[1:]], axis=1)
            seg_v = (cf[:-1] + cf[1:]) / 2
            lc = LineCollection(segs, cmap=CONTACT, norm=plt.Normalize(0, vmax),
                                linewidths=1.0 + 3.6 * seg_v / max(vmax, 1e-9),
                                capstyle="round", joinstyle="round", zorder=3)
            lc.set_array(seg_v)
            ax.add_collection(lc)
            draw_ligand(ax, ps["drugElems"], dxy, dg, ms=24, lw=1.5)
            # name the residues that hold the ligand most often, fanned so they do not stack
            picks = list(np.argsort(-cf)[:3])
            for pi, i in enumerate(picks):
                off = [(5, 4), (5, -7), (-6, 5)][pi % 3]
                ax.annotate(f"{ps['seq'][i]}{i+1}", xy[i], textcoords="offset points",
                            xytext=off, fontsize=6.2, color="#9e1f13", fontweight="bold",
                            zorder=6, ha="left" if off[0] > 0 else "right")
            ax.annotate("N", xy[0], textcoords="offset points", xytext=(0, 5),
                        fontsize=6.2, color="#3a3f45", ha="center", zorder=6)
            ax.annotate("C", xy[-1], textcoords="offset points", xytext=(0, 5),
                        fontsize=6.2, color="#3a3f45", ha="center", zorder=6)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.margins(0.14)
            if c == 0:
                ax.text(-0.04, 0.5, SYS_LAB[d], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=7.4, fontweight="bold",
                        color=SYS_COL[d])
                panel_label(ax, "a" if r == 0 else "b", dx=-0.115, dy=1.06)
            if r == 1 and c == 0:
                scalebar(ax, 10.0, u"10 Å", loc=(0.02, 0.01))

    # (c) the chemistry that explains the whole result
    for i, d in enumerate(drugs):
        ax = fig.add_subplot(gs[2, i])
        ps = PK["systems"][PKIDX[d]]
        pose = ps["params"][K]["states"]["2"]["poses"][0]
        dg = np.array(pose["drug"], float)
        el = ps["drugElems"]
        b = pca_basis(dg)
        xy = project(dg, b)
        draw_ligand(ax, el, xy, dg, ms=62, lw=2.4)
        g = ligand_groups(el, dg)
        cen = xy.mean(0)

        def label_out(idx, text, colour, extra=1.0):
            """Push the group label radially away from the molecule so it clears the atoms."""
            d = xy[idx] - cen
            n = np.linalg.norm(d)
            u = d / n if n > 1e-6 else np.array([0.0, 1.0])
            ax.annotate(text, xy[idx], textcoords="offset points",
                        xytext=(u[0] * 20 * extra, u[1] * 20 * extra),
                        ha="center", va="center", fontsize=6.8, color=colour,
                        fontweight="bold", zorder=7)
        if g["S"] is not None:
            label_out(g["S"], r"SO$_3^-$", "#8a6a00")
        if g["N"] is not None:
            label_out(g["N"], r"NH$_3^+$", "#2a78d6")
        if g["carboxyl_C"] is not None:
            label_out(g["carboxyl_C"], r"COO$^-$", "#d1332e")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0.34)
        chem = ("tramiprosate, " + r"C$_3$H$_9$NO$_3$S" + "\nzwitterion: one cationic amine"
                if d == "TMP" else
                "SPA, " + r"C$_3$H$_4$O$_5$S" + "\ndianion: no cationic group")
        ax.set_title(chem, fontsize=6.6, fontweight="normal", color=SYS_COL[d], pad=2)
        if i == 0:
            panel_label(ax, "c", dx=-0.115, dy=1.16)
    axn = fig.add_subplot(gs[2, 2])
    axn.axis("off")
    axn.text(0.0, 0.72, "Why the two ligands pick different residues", fontsize=6.9,
             fontweight="bold", va="top", ha="left", transform=axn.transAxes)
    axn.text(0.0, 0.55, "TMP carries a positive amine as well as a sulfonate, so it\n"
             "pairs with acidic side chains (Asp, Glu). SPA has no cationic\n"
             "group, so it can only salt-bridge to basic side chains\n"
             "(Arg, Lys). Fig. 3 shows exactly that split.", fontsize=6.4, va="top",
             ha="left", color="#4d545c", transform=axn.transAxes, linespacing=1.55)
    cax = fig.add_subplot(gs[0:2, 3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax * 100), cmap=CONTACT),
                      cax=cax)
    cb.set_label("Residue-ligand contact frequency\n(% of ligand-bound frames)", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.4, width=0.6, size=2.2)
    cb.outline.set_linewidth(0.5)
    lg = [plt.Line2D([], [], marker="o", ls="", ms=4.2, mfc=ELEM_COL[e], mec="white",
                     mew=0.5, label=e) for e in ("C", "N", "O", "S")]
    fig.legend(handles=lg, ncol=4, loc="upper right", bbox_to_anchor=(0.928, 1.005),
               title="Ligand atoms", title_fontsize=6.6, fontsize=6.6)
    fig.text(0.055, 0.995, "Best-contacting ligand pose per state; backbone width and colour\n"
             "give each residue's contact frequency over all bound frames",
             ha="left", va="top", fontsize=6.6, color="#4d545c", linespacing=1.5)
    save(fig, "fig3_binding_pocket")


# =====================================================================
# Figure 3 - which residues each drug actually engages
# =====================================================================
def fig3():
    fig = plt.figure(figsize=(COL2, 4.15))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.017], height_ratios=[1, 1, 1.5],
                          hspace=0.52, wspace=0.02, left=0.085, right=0.955,
                          top=0.92, bottom=0.095)
    mats = {}
    vmax = 0.0
    for d in ("TMP", "SPA"):
        M = np.array([specific(d, k) for k in range(3)]) * 100
        mats[d] = M
        vmax = max(vmax, M.max())
    for r, d in enumerate(("TMP", "SPA")):
        ax = fig.add_subplot(gs[r, 0])
        im = ax.imshow(mats[d], aspect="auto", cmap=SEQ, vmin=0, vmax=vmax,
                       extent=[0.5, NRES + 0.5, 2.5, -0.5], interpolation="nearest")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([f"State {k}" for k in range(3)])
        for t, k in zip(ax.get_yticklabels(), range(3)):
            t.set_color(STATE_COL[k])
        # no "Markov state" axis title: the tick labels already name the rows, and the
        # left margin belongs to the system name
        ax.set_xticks(RES_TICKS)
        ax.set_xlim(0.5, NRES + 0.5)
        ax.tick_params(axis="x", labelsize=6.4)
        # system name sits left of the rows, keeping the top clear for the region ruler
        ax.text(-0.075, 0.5, SYS_LAB[d], transform=ax.transAxes, rotation=90,
                va="center", ha="center", fontsize=7.4, fontweight="bold",
                color=SYS_COL[d])
        for name, a, b, c in REGIONS:
            ax.add_patch(Rectangle((a - 0.5, -0.5 - 0.40), b - a + 1, 0.28, color=c,
                                   clip_on=False, lw=0))
            if r == 0:
                ax.text((a + b) / 2, -0.5 - 0.52, name, ha="center", va="bottom",
                        fontsize=5.9, color=c)
        if r == 1:
            ax.set_xlabel(u"Aβ42 residue number")
        panel_label(ax, "a" if r == 0 else "b", dx=-0.105, dy=1.30 if r == 0 else 1.16)
    cax = fig.add_subplot(gs[0:2, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Specific contact above chance\n(% of frames)", fontsize=6.6)
    cb.ax.tick_params(labelsize=6.4, width=0.6, size=2.2)
    cb.outline.set_linewidth(0.5)

    # the head-to-head that carries the finding
    ax = fig.add_subplot(gs[2, 0])
    xs = np.arange(1, NRES + 1)
    tm, sp = ens_specific("TMP") * 100, ens_specific("SPA") * 100
    ax.bar(xs - 0.2, tm, 0.4, color=SYS_COL["TMP"], label=SYS_LAB["TMP"], lw=0, zorder=3)
    ax.bar(xs + 0.2, sp, 0.4, color=SYS_COL["SPA"], label=SYS_LAB["SPA"], lw=0, zorder=3)
    ax.set_ylabel("Specific contact,\npopulation-weighted (% of frames)")
    residue_axis(ax)
    ax.set_ylim(0, max(tm.max(), sp.max()) * 1.34)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper right")
    for i in range(NRES):
        aa = SEQ42[i]
        if aa in BASIC and sp[i] > 1.5:
            ax.annotate(f"{aa}{i+1}", (i + 1.2, sp[i]), textcoords="offset points",
                        xytext=(0, 2.5), ha="center", fontsize=6.0,
                        color=SYS_COL["SPA"], fontweight="bold")
        if aa in ACIDIC and tm[i] > 18:
            ax.annotate(f"{aa}{i+1}", (i + 0.8, tm[i]), textcoords="offset points",
                        xytext=(0, 2.5), ha="center", fontsize=6.0,
                        color=SYS_COL["TMP"], fontweight="bold")
    # mark side-chain charge under the sequence axis
    for i in range(NRES):
        aa = SEQ42[i]
        if aa in BASIC or aa in ACIDIC:
            ax.plot(i + 1, -max(tm.max(), sp.max()) * 0.055,
                    marker="+" if aa in BASIC else "_",
                    color="#2a78d6" if aa in BASIC else "#d1332e",
                    ms=3.4, mew=0.9, clip_on=False, zorder=5)
    ax.text(0.0, -0.30, "+ basic (Arg, Lys)     " + u"−" + " acidic (Asp, Glu)",
            transform=ax.transAxes, fontsize=6.2, color="#4d545c")
    panel_label(ax, "c", dx=-0.075, dy=1.16)
    save(fig, "fig4_drug_binding")


# =====================================================================
# Figure 4 - the aggregation readout
# =====================================================================
def fig4():
    fig = plt.figure(figsize=(COL2, 2.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.36], wspace=0.26,
                          left=0.072, right=0.985, top=0.86, bottom=0.155)
    ax = fig.add_subplot(gs[0, 0])
    curves = {}
    for s in SYS_KEYS:
        o, P = occ(s), states2(s)
        b = np.zeros(NRES)
        for k, v in P.items():
            b += o[int(k)] * np.array(v["beta"])
        curves[s] = b * 100
    top = max(c.max() for c in curves.values())
    for name, a, bb, c in REGIONS:
        ax.axvspan(a - 0.5, bb + 0.5, color=c, alpha=0.10, lw=0, zorder=0)
        ax.text((a + bb) / 2, top * 1.14, name, ha="center", va="bottom",
                fontsize=5.9, color=c)
    for s in SYS_KEYS:
        ax.plot(np.arange(1, NRES + 1), curves[s], color=SYS_COL[s], label=SYS_LAB[s],
                lw=1.35, zorder=3, solid_capstyle="round")
    ax.set_ylabel(u"β-strand propensity\n(% of frames, population-weighted)")
    residue_axis(ax)
    ax.set_ylim(0, top * 1.13)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(2.5))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.30))
    panel_label(ax, "a", dx=-0.068, dy=1.30)

    ax2 = fig.add_subplot(gs[0, 1])
    sel = [("KLVFF / CHC", 16, 21), ("C-terminus", 30, 42)]
    x = np.arange(len(sel))
    w = 0.26
    for i, s in enumerate(SYS_KEYS):
        vals = [curves[s][a - 1:b].mean() for _, a, b in sel]
        ax2.bar(x + (i - 1) * w, vals, w * 0.88, color=SYS_COL[s], lw=0, zorder=3)
        for xx, vv in zip(x + (i - 1) * w, vals):
            ax2.text(xx, vv + 0.22, f"{vv:.1f}", ha="center", va="bottom", fontsize=6.0,
                     color=SYS_COL[s], fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n for n, _, _ in sel], fontsize=6.6)
    ax2.set_ylabel(u"Mean β-strand (%)")
    ax2.set_xlabel("Aggregation-driving region")
    ax2.set_ylim(0, max(curves[s][29:42].mean() for s in SYS_KEYS) * 1.45)
    ax2.grid(axis="y", zorder=0)
    ax2.set_axisbelow(True)
    panel_label(ax2, "b", dx=-0.22, dy=1.30)
    save(fig, "fig5_beta_propensity")


# =====================================================================
# Figure 5 - the whole ensemble moves
# =====================================================================
def fig5():
    rgE = np.array(S2["meta"]["rgEdges"], float)
    beE = np.array(S2["meta"]["betaEdges"], float)
    nR = len(rgE) - 1
    L = {s: np.array(S2["systems"][IDX[s]]["params"][K]["landscape"], float) for s in SYS_KEYS}
    # crop the beta axis to the sampled range; a few outlier frames must not set the scale
    perB = sum(L[s].sum(0) for s in SYS_KEYS)
    tot = perB.sum()
    nB = int(np.searchsorted(np.cumsum(perB), tot * 0.995) + 2)
    nB = min(nB, len(beE) - 1)
    FMAX = 5.0

    fig = plt.figure(figsize=(COL2, 3.75))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1, 0.62],
                          hspace=0.60, wspace=0.14, left=0.072, right=0.925,
                          top=0.88, bottom=0.125)
    for c, s in enumerate(SYS_KEYS):
        ax = fig.add_subplot(gs[0, c])
        A = L[s][:, :nB]
        mx = A.max()
        F = np.where(A > 0, np.log(np.maximum(mx / np.maximum(A, 1e-9), 1.0)), np.nan)
        im = ax.imshow(np.clip(F, 0, FMAX).T, origin="lower", aspect="auto",
                       cmap=SEQ.reversed(), vmin=0, vmax=FMAX,
                       extent=[rgE[0], rgE[nR], beE[0] * 100, beE[nB] * 100],
                       interpolation="nearest")
        ax.set_facecolor("#eef0f2")
        P = S2["systems"][IDX[s]]["params"][K]["states"]
        # SPA collapses the states almost on top of each other, so labels fan out
        fan = [(6.0, 4.0), (6.0, -8.5), (-7.0, 4.0)]
        for ki, k in enumerate(sorted(P, key=int)):
            v = P[k]
            ax.plot(v["rgMean"], v["betaMean"] * 100, "o", mfc="none",
                    mec=STATE_COL[int(k)], mew=1.3, ms=6.2, zorder=5)
            off = fan[ki % 3]
            ax.annotate(f"S{k}", (v["rgMean"], v["betaMean"] * 100),
                        textcoords="offset points", xytext=off, fontsize=6.2,
                        ha="left" if off[0] > 0 else "right",
                        color=STATE_COL[int(k)], fontweight="bold", zorder=6)
        ax.set_xlim(rgE[0], 24)
        ax.set_ylim(0, beE[nB] * 100)
        ax.set_xlabel(u"Radius of gyration, Rg (Å)")
        if c == 0:
            ax.set_ylabel(u"β-content\n(% of residues)")
            panel_label(ax, "a", dx=-0.20, dy=1.24)
        ax.set_title(SYS_LAB[s], color=SYS_COL[s], fontsize=7.4, pad=3)
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(u"Relative free energy,\n" + u"−ln p (kT)", fontsize=6.6)
    cb.ax.tick_params(labelsize=6.4, width=0.6, size=2.2)
    cb.outline.set_linewidth(0.5)

    ax = fig.add_subplot(gs[1, 0:3])
    centres = (rgE[:-1] + rgE[1:]) / 2
    for s in SYS_KEYS:
        m = L[s].sum(1)
        m = m / m.sum()
        mean = float((m * centres).sum())
        ax.plot(centres, m * 100, color=SYS_COL[s], lw=1.35,
                label=f"{SYS_LAB[s]}   mean {mean:.1f} " + u"Å", zorder=3)
        ax.axvline(mean, color=SYS_COL[s], lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.set_xlim(rgE[0], 24)
    ax.set_xlabel(u"Radius of gyration, Rg (Å)")
    ax.set_ylabel("Frames (%)")
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 17.5)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=6.5)
    panel_label(ax, "b", dx=-0.068, dy=1.34)
    save(fig, "fig6_conformational_landscape")


if __name__ == "__main__":
    print("rendering submission figures ...")
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("done")
