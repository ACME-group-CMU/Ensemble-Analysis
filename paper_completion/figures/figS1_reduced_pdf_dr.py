"""
Fig. S1 — Boltzmann-weighted reduced PDF D(r) for all three system sizes.
Shows that the first-shell Si–O peak is converged with cell size and matches
the experimental position of 1.61 Å (Grimley et al., 1990).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_completion._load import (   # noqa: E402
    SizeEnsemble, SIZE_COLORS, SIZE_LABELS, first_peak_position,
)

SIZES = (24, 36, 48)
GRIMLEY_SIO_BOND = 1.61  # Å, Grimley et al. 1990
OUT = Path(__file__).parent


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    summary_rows = []
    for sz in SIZES:
        e = SizeEnsemble(sz)
        r, D, _ = e.avg_total_dr()
        if r is None:
            continue
        ax.plot(r, D, color=SIZE_COLORS[sz], lw=1.6,
                label=f"{SIZE_LABELS[sz]} (N={len(e.ids)})")
        peak = first_peak_position(r, D, 1.4, 1.9)
        summary_rows.append((sz, peak))

    ax.axvline(GRIMLEY_SIO_BOND, ls=":", color="k", lw=0.9,
               label=f"Grimley 1990 (Si–O = {GRIMLEY_SIO_BOND:.2f} Å)")
    ax.set_xlabel(r"$r$ (Å)")
    ax.set_ylabel(r"$D(r)$  (Å$^{-2}$)")
    ax.set_xlim(0.0, 8.0)
    ax.legend(loc="upper right", frameon=False)

    txt = "\n".join([f"{sz}-atom first peak: {p:.3f} Å" for sz, p in summary_rows])
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(facecolor="white", edgecolor="0.85"))
    ax.set_title("Fig. S1 — Boltzmann-weighted reduced PDF $D(r)$ vs cell size",
                 loc="left", fontsize=11)

    fig.tight_layout()
    pdf = OUT / "figS1_reduced_pdf_dr.pdf"
    png = OUT / "figS1_reduced_pdf_dr.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    print(f"Wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
