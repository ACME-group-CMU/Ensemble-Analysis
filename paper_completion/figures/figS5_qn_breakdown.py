"""
Fig. S5 — Boltzmann-weighted Q^n speciation broken down per system size.
Stacked bars showing Q^0..Q^4 fractions; companion panel zooms into the
non-Q^4 regime (Q^0..Q^3) so the trend across sizes is visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_completion._load import (   # noqa: E402
    SizeEnsemble, SIZE_COLORS, SIZE_LABELS,
)

SIZES = (24, 36, 48)
QN_COLORS = {0: "#440154", 1: "#3b528b", 2: "#21918c", 3: "#5ec962", 4: "#fde725"}
OUT = Path(__file__).parent


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    fracs = {sz: SizeEnsemble(sz).avg_qn_fractions() for sz in SIZES}

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4),
                              gridspec_kw={"width_ratios": [1, 1]})

    # ---- left: stacked bars (full distribution) ----
    ax = axes[0]
    x = np.arange(len(SIZES))
    bottom = np.zeros(len(SIZES))
    for n in range(5):
        h = np.array([fracs[sz][n] for sz in SIZES])
        ax.bar(x, h, bottom=bottom, color=QN_COLORS[n],
               edgecolor="white", linewidth=0.6, label=f"Q$^{n}$")
        for i, (b, v) in enumerate(zip(bottom, h)):
            if v >= 0.04:
                ax.text(x[i], b + v / 2, f"{v*100:.1f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if n in (0, 1) else "black")
        bottom += h
    ax.set_xticks(x); ax.set_xticklabels([SIZE_LABELS[s] for s in SIZES])
    ax.set_ylabel("Q$^n$ fraction (per Si)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    ax.set_title("(a) full Q$^n$ stack", loc="left")

    # ---- right: non-Q4 zoom ----
    ax = axes[1]
    width = 0.18
    offsets = (-1.5, -0.5, 0.5, 1.5)
    for i, n in enumerate((0, 1, 2, 3)):
        h = np.array([fracs[sz][n] for sz in SIZES]) * 100
        ax.bar(x + offsets[i] * width, h, width, color=QN_COLORS[n],
               edgecolor="white", linewidth=0.6, label=f"Q$^{n}$")
        for xi, hi in zip(x + offsets[i] * width, h):
            if hi >= 0.5:
                ax.text(xi, hi + 0.4, f"{hi:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([SIZE_LABELS[s] for s in SIZES])
    ax.set_ylabel("non-Q$^4$ fraction (%)")
    ax.set_title("(b) non-Q$^4$ zoom", loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.set_ylim(0, max(np.array([fracs[sz][3] for sz in SIZES]) * 100) * 1.4)

    fig.suptitle("Fig. S5 — Q$^n$ speciation by system size",
                 x=0.04, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf = OUT / "figS5_qn_breakdown.pdf"
    png = OUT / "figS5_qn_breakdown.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    print(f"Wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
