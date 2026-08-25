"""
Fig. 3 — Si–O–Si and O–Si–O bond angle distributions.

References overlaid: Mauri et al. 2000 (29Si NMR, 151 +/- 11 deg) as a dashed
Gaussian, and Sarnthein et al. 1995 (CPMD, 136-142 deg) as a shaded band.
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

MAURI_MU, MAURI_SIGMA = 151.0, 11.0
SARNTHEIN_BAND = (136.0, 142.0)
TETRAHEDRAL = 109.47


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    ens = {s: SizeEnsemble(s) for s in SIZES}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax1.axvspan(*SARNTHEIN_BAND, color="0.85", zorder=0,
                label="Sarnthein 1995 (CPMD)")
    x = np.linspace(60, 180, 400)
    gauss = np.exp(-0.5 * ((x - MAURI_MU) / MAURI_SIGMA) ** 2)
    gauss /= gauss.sum() * (x[1] - x[0])
    ax1.plot(x, gauss, color="black", ls="--", lw=1.5,
             label=rf"Mauri 2000 ($\mu$={MAURI_MU:.0f}°, $\sigma$={MAURI_SIGMA:.0f}°)")

    for s in SIZES:
        c, p = ens[s].bad("si_o_si")
        m = (c >= 60) & (c <= 180)
        ax1.plot(c[m], p[m], color=SIZE_COLORS[s], lw=1.4, label=SIZE_LABELS[s])

    ax1.set_xlabel("Si–O–Si angle (°)")
    ax1.set_ylabel("probability density (°$^{-1}$)")
    ax1.set_xlim(60, 180)
    ax1.legend(frameon=False)
    ax1.text(0.02, 0.96, "(a)", transform=ax1.transAxes, va="top", fontsize=11)

    ax2.axvline(TETRAHEDRAL, color="black", ls="--", lw=1.4,
                label=f"tetrahedral ({TETRAHEDRAL:.1f}°)")
    for s in SIZES:
        c, p = ens[s].bad("o_si_o")
        m = (c >= 60) & (c <= 180)
        ax2.plot(c[m], p[m], color=SIZE_COLORS[s], lw=1.4, label=SIZE_LABELS[s])

    ax2.set_xlabel("O–Si–O angle (°)")
    ax2.set_ylabel("probability density (°$^{-1}$)")
    ax2.set_xlim(60, 180)
    ax2.legend(frameon=False)
    ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes, va="top", fontsize=11)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_bond_angles.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT/'fig3_bond_angles.pdf'}")

    for s in SIZES:
        for kind in ("si_o_si", "o_si_o"):
            m = ens[s].bad_moments(kind)
            print(f"  {s}-atom {kind}: peak {m['peak']:.1f}°, "
                  f"mean {m['mean']:.1f}°, sigma {m['std']:.1f}°")


if __name__ == "__main__":
    main()
