"""
Fig. S8 — Raw coordination distribution (not bridging-only) for Si and O,
at the 2.0 Å Si–O cutoff. Shows the full per-Si and per-O neighbor-count
PMF, with both unweighted (top row) and Boltzmann-weighted at T = 2000 K
(bottom row). Surfaces the 5- and 6-fold Si tail and the 3-fold O tail
that the Q^n classifier silently caps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_completion._load import (   # noqa: E402
    SizeEnsemble, SIZE_COLORS, SIZE_LABELS, boltzmann_coord_distribution,
)

SIZES = (24, 36, 48)
CUTOFF = 2.0
MAX_N = 7
OUT = Path(__file__).parent


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    pmfs = {}
    for sz in SIZES:
        e = SizeEnsemble(sz)
        pmfs[sz] = {
            ("Si_O", False): boltzmann_coord_distribution(e, "Si", "O", CUTOFF, MAX_N, use_weights=False),
            ("Si_O", True):  boltzmann_coord_distribution(e, "Si", "O", CUTOFF, MAX_N, use_weights=True),
            ("O_Si", False): boltzmann_coord_distribution(e, "O",  "Si", CUTOFF, MAX_N, use_weights=False),
            ("O_Si", True):  boltzmann_coord_distribution(e, "O",  "Si", CUTOFF, MAX_N, use_weights=True),
        }

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), sharex=True)

    panels = [
        (axes[0, 0], "Si_O", False, "(a) Si coordination — unweighted",
         r"# O neighbors per Si", 4),
        (axes[0, 1], "O_Si", False, "(b) O coordination — unweighted",
         r"# Si neighbors per O", 2),
        (axes[1, 0], "Si_O", True,  "(c) Si coordination — Boltzmann (2000 K)",
         r"# O neighbors per Si", 4),
        (axes[1, 1], "O_Si", True,  "(d) O coordination — Boltzmann (2000 K)",
         r"# Si neighbors per O", 2),
    ]

    width = 0.26
    offsets = (-1, 0, 1)

    for ax, key, weighted, title, xlabel, ideal in panels:
        x = np.arange(MAX_N + 1)
        for i, sz in enumerate(SIZES):
            h = pmfs[sz][(key, weighted)]
            bars = ax.bar(x + offsets[i] * width, h, width,
                          color=SIZE_COLORS[sz], edgecolor="white", linewidth=0.4,
                          alpha=0.9, label=SIZE_LABELS[sz] if ax is axes[0, 0] else None)
            for xi, hi in zip(x + offsets[i] * width, h):
                if hi >= 0.005:
                    ax.text(xi, hi + 0.01, f"{hi*100:.1f}", ha="center",
                            va="bottom", fontsize=6.5, rotation=0)
        ax.axvline(ideal, ls="--", color="0.3", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 1.0)
        ax.set_xlim(-0.5, MAX_N + 0.5)
        ax.set_xticks(x)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("fraction")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(alpha=0.25, ls=":", which="both")

    axes[0, 0].legend(frameon=False, loc="upper left")

    fig.suptitle(
        f"Fig. S8 — Raw neighbor-count distributions  (Si–O cutoff {CUTOFF:.1f} Å, "
        "ideal at dashed line)",
        x=0.04, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf = OUT / "figS8_raw_coordination_distribution.pdf"
    png = OUT / "figS8_raw_coordination_distribution.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    print(f"Wrote {pdf.name} and {png.name}")

    # Also print a small text summary so the numbers can go straight into the SI text
    print("\nBoltzmann-weighted fractions (2.0 Å cutoff):")
    for sz in SIZES:
        sio = pmfs[sz][("Si_O", True)]
        osi = pmfs[sz][("O_Si", True)]
        under_si = sio[:4].sum()
        ideal_si = sio[4]
        over_si  = sio[5:].sum()
        under_o  = osi[:2].sum()
        ideal_o  = osi[2]
        over_o   = osi[3:].sum()
        print(f"  {sz}-atom: Si  under(<4)={under_si*100:5.2f}%  ideal(4)={ideal_si*100:5.2f}%  over(>4)={over_si*100:5.2f}%   "
              f"O  under(<2)={under_o*100:5.2f}%  ideal(2)={ideal_o*100:5.2f}%  over(>2)={over_o*100:5.2f}%")


if __name__ == "__main__":
    main()
