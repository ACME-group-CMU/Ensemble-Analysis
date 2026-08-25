"""
Fig. S3 — Bond-angle distribution summary table+plot.
Shows Si–O–Si and O–Si–O peak locations and FWHMs for all three system sizes
alongside CPMD reference (Sarnthein et al., 1995) and 29Si-NMR reference
(Mauri et al., 2000).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_completion._load import (   # noqa: E402
    SizeEnsemble, SIZE_COLORS, SIZE_LABELS, peak_of_hist, fwhm_of_hist,
)

SIZES = (24, 36, 48)
OUT = Path(__file__).parent

# References
NMR = {"Si-O-Si": (151.0, 11.0)}            # Mauri et al. 2000: peak, sigma (NOT fwhm)
NMR_FWHM_SIO = 2.355 * 11.0                 # 2 sqrt(2 ln 2) sigma ~ 25.9°
CPMD_SIOSI_PEAK = 137.0                     # Sarnthein 1995, midpoint of 136-142°
CPMD_OSIO = (109.5, 9.0)                    # peak, FWHM


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    rows = []
    bads_sios = {}
    bads_osio = {}
    for sz in SIZES:
        e = SizeEnsemble(sz)
        bads_sios[sz] = e.avg_bad("si_o_si", angle_range=(60, 180), bins=240)
        bads_osio[sz] = e.avg_bad("o_si_o",  angle_range=(60, 180), bins=240)
        rows.append((sz,
                     peak_of_hist(bads_sios[sz]["centers"], bads_sios[sz]["pdf"]),
                     fwhm_of_hist(bads_sios[sz]["centers"], bads_sios[sz]["pdf"]),
                     peak_of_hist(bads_osio[sz]["centers"], bads_osio[sz]["pdf"]),
                     fwhm_of_hist(bads_osio[sz]["centers"], bads_osio[sz]["pdf"])))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))

    # ---- left: Si-O-Si ----
    ax = axes[0]
    for sz in SIZES:
        d = bads_sios[sz]
        ax.plot(d["centers"], d["pdf"], color=SIZE_COLORS[sz], lw=1.5,
                label=f"{SIZE_LABELS[sz]}")
    # NMR Mauri (Gaussian)
    ang = np.linspace(60, 180, 600)
    mu, sig = NMR["Si-O-Si"]
    g = np.exp(-(ang - mu) ** 2 / (2 * sig ** 2)) / (sig * np.sqrt(2 * np.pi))
    ax.plot(ang, g, "k--", lw=1.0, label="Mauri 2000 (NMR, 151°/11°σ)")
    ax.axvline(CPMD_SIOSI_PEAK, color="0.4", ls=":", lw=0.9,
               label=f"Sarnthein CPMD (~{CPMD_SIOSI_PEAK:.0f}°)")
    ax.set_xlim(80, 180)
    ax.set_xlabel("Si–O–Si angle (deg)")
    ax.set_ylabel("PDF (deg$^{-1}$)")
    ax.set_title("Si–O–Si", loc="left")
    ax.legend(frameon=False, loc="upper left")

    # ---- right: O-Si-O ----
    ax = axes[1]
    for sz in SIZES:
        d = bads_osio[sz]
        ax.plot(d["centers"], d["pdf"], color=SIZE_COLORS[sz], lw=1.5,
                label=f"{SIZE_LABELS[sz]}")
    mu, fw = CPMD_OSIO
    sig = fw / 2.355
    g = np.exp(-(ang - mu) ** 2 / (2 * sig ** 2)) / (sig * np.sqrt(2 * np.pi))
    ax.plot(ang, g, "k--", lw=1.0,
            label=f"Sarnthein CPMD (Gaussian, {mu:.1f}°/{fw:.0f}° FWHM)")
    ax.set_xlim(70, 160)
    ax.set_xlabel("O–Si–O angle (deg)")
    ax.set_ylabel("PDF (deg$^{-1}$)")
    ax.set_title("O–Si–O", loc="left")
    ax.legend(frameon=False, loc="upper left")

    # ---- bottom: small text summary table ----
    txt = "Peak ° / FWHM °\n"
    txt += f"  Si–O–Si  ref. NMR (Mauri):  151 / {NMR_FWHM_SIO:.0f} (Gaussian σ→FWHM)\n"
    txt += f"  Si–O–Si  ref. CPMD:        ~137 (peak)\n"
    for sz, ps, fs, po, fo in rows:
        txt += f"  {sz}-atom  Si–O–Si  {ps:5.1f} / {fs:5.1f}     O–Si–O  {po:5.1f} / {fo:5.1f}\n"
    txt += f"  ref. CPMD O–Si–O:           {CPMD_OSIO[0]} / {CPMD_OSIO[1]}\n"
    fig.text(0.04, -0.04, txt, family="monospace", fontsize=8.4, ha="left", va="top")

    fig.suptitle("Fig. S3 — Bond-angle distributions vs reference data",
                 x=0.04, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    pdf = OUT / "figS3_bad_summary.pdf"
    png = OUT / "figS3_bad_summary.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
