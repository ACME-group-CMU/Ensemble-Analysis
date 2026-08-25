"""
Fig. 1 — (a) neutron S(q) against Grimley 1990, (b) partial g(r).

S(q) is evaluated exactly on each cell's reciprocal lattice and binned, which is
why the curves are stepped at low q: a small cell simply has few commensurate
k-vectors there. That sparsity is real information about the finite-size limit,
so it is shown rather than smoothed away.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_completion._load import SizeEnsemble, SIZE_COLORS, SIZE_LABELS  # noqa: E402

SIZES = (24, 36, 48)
OUT = Path(__file__).parent
PAIRS = [("Si", "O"), ("Si", "Si"), ("O", "O")]
PAIR_LABEL = {("Si", "O"): "Si–O", ("Si", "Si"): "Si–Si", ("O", "O"): "O–O"}


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    ens = {s: SizeEnsemble(s) for s in SIZES}
    fig = plt.figure(figsize=(9.5, 4.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], hspace=0.12, wspace=0.28)
    ax_sq = fig.add_subplot(gs[:, 0])
    gr_axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    grim_path = ROOT / "References" / "Grimley_Sq.csv"
    if grim_path.exists():
        grim = pd.read_csv(grim_path)
        ax_sq.plot(grim["x"], grim["y"], color="black", lw=1.6,
                   label="Grimley 1990 (neutron)", zorder=5)

    for s in SIZES:
        q, S = ens[s].structure_factor()
        m = np.isfinite(S)
        ax_sq.plot(q[m], q[m] * (S[m] - 1.0), color=SIZE_COLORS[s], lw=1.3,
                   alpha=0.9, label=SIZE_LABELS[s])

    ax_sq.set_xlabel(r"$q$ (Å$^{-1}$)")
    ax_sq.set_ylabel(r"$q\,[S(q)-1]$ (Å$^{-1}$)")
    ax_sq.set_xlim(0, 12)
    ax_sq.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax_sq.legend(frameon=False)
    ax_sq.text(0.02, 0.96, "(a)", transform=ax_sq.transAxes, va="top", fontsize=11)

    for ax, pair in zip(gr_axes, PAIRS):
        for s in SIZES:
            r, g = ens[s].g_r(pair=pair)
            ax.plot(r, g, color=SIZE_COLORS[s], lw=1.2, alpha=0.9,
                    label=SIZE_LABELS[s] if pair == PAIRS[0] else None)
        ax.axhline(1.0, color="0.85", lw=0.6, zorder=0)
        ax.set_xlim(0, 6.5)
        ax.text(0.97, 0.88, PAIR_LABEL[pair], transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
        if ax is not gr_axes[-1]:
            ax.set_xticklabels([])

    gr_axes[1].legend(frameon=False, loc="upper left", fontsize=7)
    gr_axes[0].text(0.02, 0.92, "(b)", transform=gr_axes[0].transAxes,
                    va="top", fontsize=11)
    gr_axes[-1].set_xlabel(r"$r$ (Å)")
    gr_axes[1].set_ylabel(r"$g_{\alpha\beta}(r)$")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_sq_and_partials.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT/'fig1_sq_and_partials.pdf'}")

    for s in SIZES:
        r, g = ens[s].g_r(pair=("Si", "O"))
        print(f"  {s}-atom Si-O first peak: {r[np.argmax(g)]:.3f} Å")


if __name__ == "__main__":
    main()
