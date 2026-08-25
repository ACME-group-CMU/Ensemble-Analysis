"""
Fig. S6 — Distribution of 2MR count per microstate (unweighted) and per
Boltzmann-weighted microstate, for the three system sizes.
Shows that 2MR-containing structures are common in the raw RSL ensemble
but contribute progressively less weight as their concentration grows
because they sit in the high-energy tail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_completion._load import (   # noqa: E402
    SizeEnsemble, SIZE_COLORS, SIZE_LABELS, load_rings,
)

SIZES = (24, 36, 48)
OUT = Path(__file__).parent


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharex=True)

    for sz in SIZES:
        e = SizeEnsemble(sz)
        rings = load_rings(e.ids)
        n2mr = np.array([rings[s]["RN"].get(2, 0.0) * e.n_si[s] for s in e.ids])
        w = e.weight_array()

        # unweighted PMF
        max_n = int(n2mr.max())
        bins = np.arange(-0.5, max_n + 1.5, 1.0)
        h_uw, _ = np.histogram(n2mr, bins=bins)
        h_uw = h_uw / h_uw.sum()
        h_w, _ = np.histogram(n2mr, bins=bins, weights=w)
        h_w = h_w / h_w.sum()
        centers = 0.5 * (bins[:-1] + bins[1:])

        axes[0].plot(centers, h_uw, "o-", color=SIZE_COLORS[sz], lw=1.4, ms=4,
                     label=f"{SIZE_LABELS[sz]} (N={len(e.ids)})")
        axes[1].plot(centers, h_w,  "o-", color=SIZE_COLORS[sz], lw=1.4, ms=4,
                     label=f"{SIZE_LABELS[sz]}")

    for ax, title in zip(axes, ["(a) unweighted", "(b) Boltzmann-weighted (T = 2000 K)"]):
        ax.set_xlabel("2MRs per microstate")
        ax.set_ylabel("fraction")
        ax.set_yscale("log")
        ax.set_ylim(5e-5, 1.0)
        ax.set_xlim(-0.5, 16.5)
        ax.legend(frameon=False, loc="upper right")
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.25, ls=":")

    fig.suptitle("Fig. S6 — Per-microstate 2MR count: raw vs. Boltzmann-weighted",
                 x=0.04, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf = OUT / "figS6_2mr_count_distribution.pdf"
    png = OUT / "figS6_2mr_count_distribution.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    print(f"Wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
