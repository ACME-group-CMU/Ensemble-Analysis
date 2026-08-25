"""
Fig. S2 — Boltzmann-weighted Si and O coordination numbers vs system size,
computed by direct bond counting at the 2.0 Å Si–O cutoff used in the Methods.
Confirms tetrahedral Si coordination (~4) and 2-fold O coordination (~2) in
all three RSL ensembles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_completion._load import (   # noqa: E402
    SizeEnsemble, SIZE_COLORS, SIZE_LABELS, boltzmann_avg_coordination,
)

SIZES = (24, 36, 48)
OUT = Path(__file__).parent
CUTOFF = 2.0


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })

    n_o_si  = []
    n_si_o  = []
    for sz in SIZES:
        e = SizeEnsemble(sz)
        n_o_si.append(boltzmann_avg_coordination(e, "Si", "O", cutoff=CUTOFF))
        n_si_o.append(boltzmann_avg_coordination(e, "O",  "Si", cutoff=CUTOFF))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))

    x = np.arange(len(SIZES))
    width = 0.55

    for ax, vals, ref, ref_label, ylabel, title in [
        (axes[0], n_o_si, 4.0, "tetrahedral (4)",
         r"$\langle N_{\rm O}\rangle$ around Si", "Si coordination"),
        (axes[1], n_si_o, 2.0, "bridging (2)",
         r"$\langle N_{\rm Si}\rangle$ around O", "O coordination"),
    ]:
        bars = ax.bar(x, vals, width, color=[SIZE_COLORS[s] for s in SIZES],
                      edgecolor="white", linewidth=1.0, alpha=0.9)
        ax.axhline(ref, ls="--", color="0.3", lw=0.9, label=f"ideal {ref_label}")
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([SIZE_LABELS[s] for s in SIZES])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(loc="lower right", frameon=False)
        ax.set_ylim(0, max(vals) + 0.6)

    fig.suptitle(f"Fig. S2 — Coordination numbers (Si–O cutoff {CUTOFF:.1f} Å)",
                 x=0.04, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf = OUT / "figS2_coordination_numbers.pdf"
    png = OUT / "figS2_coordination_numbers.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    print(f"Wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
