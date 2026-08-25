"""
Fig. 2 — ring size distribution R_N against Rino 1993.

Panel (a) uses R_N = unique rings per Si excluding rings that close through a
periodic image of an atom already on the ring, which is the only counting that
describes a local feature of the network. Panel (b) shows what fraction of the
rings at each size are such self-image closures, i.e. where the cell stops being
large enough for the observable to mean anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_completion._load import SizeEnsemble, SIZE_COLORS, SIZE_LABELS  # noqa: E402

SIZES = (24, 36, 48)
OUT = Path(__file__).parent
N_RANGE = range(2, 13)

# Rino et al. PRB 47, 3053 (1993), classical MD, rings per Si by ring size
RINO = {3: 0.02, 4: 0.10, 5: 0.28, 6: 0.44, 7: 0.40, 8: 0.28, 9: 0.14, 10: 0.06}


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    ens = {s: SizeEnsemble(s) for s in SIZES}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ns = list(N_RANGE)
    ax.plot(list(RINO), list(RINO.values()), color="black", lw=1.5, ls="--",
            marker="s", ms=4, label="Rino 1993 (MD)", zorder=5)

    for s in SIZES:
        rc = ens[s].rings("RC_local")
        ax.plot(ns, [rc.get(n, 0.0) for n in ns], color=SIZE_COLORS[s], lw=1.4,
                marker="o", ms=4, label=SIZE_LABELS[s])

    ax.set_xlabel("ring size $n$ (number of Si)")
    ax.set_ylabel("$R_N$ (rings per Si)")
    ax.set_xticks(ns)
    ax.legend(frameon=False)
    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, va="top", fontsize=11)

    for s in SIZES:
        frac = ens[s].self_image_fraction()
        ax2.plot(ns, [100 * frac.get(n, 0.0) for n in ns], color=SIZE_COLORS[s],
                 lw=1.4, marker="o", ms=4, label=SIZE_LABELS[s])

    ax2.axhline(50, color="0.7", lw=0.8, ls=":")
    ax2.set_xlabel("ring size $n$ (number of Si)")
    ax2.set_ylabel("rings closing through a periodic image (%)")
    ax2.set_xticks(ns)
    ax2.set_ylim(-3, 103)
    ax2.legend(frameon=False, loc="upper left")
    ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes, va="top", fontsize=11)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_ring_distribution.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT/'fig2_ring_distribution.pdf'}")

    for s in SIZES:
        frac = ens[s].self_image_fraction()
        clean = [n for n in ns if frac.get(n, 0.0) < 0.5]
        print(f"  {s}-atom: ring sizes below 50% self-image contamination: "
              f"n <= {max(clean) if clean else 'none'}")


if __name__ == "__main__":
    main()
