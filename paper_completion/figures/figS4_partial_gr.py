"""
Fig. S4 — Boltzmann-weighted partial pair distribution functions
g_Si-Si(r), g_Si-O(r), g_O-O(r) for the three system sizes.
Highlights the absence of an edge-sharing O–O peak in the 2.2–2.4 Å range,
despite finite-weight 2MR microstates in the ensemble.
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
PAIRS = [("Si_Si", "Si–Si"), ("Si_O", "Si–O"), ("O_O", "O–O")]
EDGE_O_O_BAND = (2.2, 2.4)
OUT = Path(__file__).parent


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharex=True)

    for ax, (pair, label) in zip(axes, PAIRS):
        for sz in SIZES:
            e = SizeEnsemble(sz)
            r, g = e.avg_partial_g(pair)
            if r is None:
                continue
            ax.plot(r, g, color=SIZE_COLORS[sz], lw=1.4, label=f"{SIZE_LABELS[sz]}")
        if pair == "O_O":
            ax.axvspan(*EDGE_O_O_BAND, color="0.85", alpha=0.5,
                       label="edge-sharing band\n(2.2–2.4 Å)")
        ax.axhline(1.0, color="0.6", lw=0.6)
        ax.set_xlim(0.5, 6.0)
        ax.set_xlabel(r"$r$ (Å)")
        ax.set_ylabel(rf"$g_{{{label}}}(r)$")
        ax.set_title(label, loc="left")
        ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Fig. S4 — Boltzmann-weighted partial pair distribution functions",
                 x=0.04, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf = OUT / "figS4_partial_gr.pdf"
    png = OUT / "figS4_partial_gr.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    print(f"Wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
