"""
Fig. 4 (main text)
==================
Per-atom DFT energy of every relaxed RSL microstate, color-coded by 2MR
presence and by Q^4 fraction. Shows that 2MR-bearing and non-Q^4-bearing
microstates concentrate in a high-energy tail of the basin distribution.

Three rows (one per system size). Left column: histogram split by 2MR
presence. Right column: scatter of E vs Q^4 fraction, colored by 2MR
count. Bottom: linear regression of total energy vs N_2MR (24-atom only).

Run from repo root:
    python paper_completion/figures/fig4_energy_2mr_q4.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_completion._load import (   # noqa: E402
    SizeEnsemble, SIZE_COLORS, SIZE_LABELS, load_rings, load_qn,
)

RY_TO_EV = 13.605693122994
HAMANN_PER_2MR_eV = 1.23
SIZES = (24, 36, 48)
OUT = Path(__file__).parent


def gather(size: int):
    e = SizeEnsemble(size)
    rings = load_rings(e.ids)
    qn    = load_qn(e.ids)
    n2mr  = np.array([rings[s]["RN"].get(2, 0.0) * e.n_si[s] for s in e.ids])
    q4    = np.array([qn[s]["qn_fractions"].get(4, 0.0) for s in e.ids])
    e_pa  = e.per_atom_energies() * RY_TO_EV     # eV / atom
    n_at  = np.array([e.n_atoms[s] for s in e.ids])
    e_tot = e_pa * n_at
    return {
        "ids": e.ids,
        "n2mr": n2mr,
        "q4":   q4,
        "e_pa": e_pa,
        "e_tot": e_tot,
        "n_at": n_at,
        "weights": e.weight_array(),
    }


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 8,
    })

    data = {sz: gather(sz) for sz in SIZES}

    # Reference: per-atom energy at the lowest-energy microstate of the largest cell
    e_min_pa_global = min(d["e_pa"].min() for d in data.values())

    fig, axes = plt.subplots(3, 2, figsize=(8.8, 8.5),
                              gridspec_kw={"width_ratios": [1, 1.1]})

    for row, sz in enumerate(SIZES):
        d = data[sz]
        color = SIZE_COLORS[sz]
        de = (d["e_pa"] - e_min_pa_global) * 1000  # meV/atom above global min
        has_2mr = d["n2mr"] > 0

        # ---- left: histogram split by 2MR presence ----
        ax = axes[row, 0]
        bins = np.linspace(0, np.percentile(de, 99.5), 50)
        ax.hist(de[~has_2mr], bins=bins, color="0.55", alpha=0.85,
                edgecolor="white", linewidth=0.4, label="no 2MR")
        ax.hist(de[has_2mr], bins=bins, color=color, alpha=0.75,
                edgecolor="white", linewidth=0.4, label="≥1 2MR")
        ax.set_xlabel(r"$E - E_{\min}$  (meV / atom)")
        ax.set_ylabel("microstates")
        ax.set_title(f"{SIZE_LABELS[sz]}  (N = {len(d['ids'])})", loc="left")
        ax.legend(loc="upper right", frameon=False)

        # ---- right: scatter E vs Q^4, color = 2MR count ----
        ax = axes[row, 1]
        size_marker = 12
        sc = ax.scatter(d["q4"], de, c=d["n2mr"], cmap="magma_r",
                        s=size_marker, alpha=0.85, edgecolor="none",
                        vmin=0, vmax=max(4, np.percentile(d["n2mr"], 95)))
        ax.set_xlabel(r"Q$^4$ fraction (per Si)")
        ax.set_ylabel(r"$E - E_{\min}$  (meV / atom)")
        ax.set_xlim(-0.02, 1.02)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label("2MR count")
        # marginal regression line in eV/atom space
        if d["n2mr"].max() > 0 and (~has_2mr).any():
            A = np.vstack([d["n2mr"], np.ones_like(d["n2mr"])]).T
            slope_eV, intercept_eV = np.linalg.lstsq(A, d["e_tot"], rcond=None)[0]
            ax.text(0.02, 0.98,
                    f"ΔE/2MR (regr.) = {slope_eV:.2f} eV\n"
                    f"frac. ≥1 2MR = {has_2mr.mean()*100:.0f}%",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

    # Annotate Hamann reference on first row
    axes[0, 0].axvline(HAMANN_PER_2MR_eV * 1000 / 24,
                        color="k", ls="--", lw=0.8, alpha=0.6)
    axes[0, 0].text(HAMANN_PER_2MR_eV * 1000 / 24, axes[0, 0].get_ylim()[1] * 0.92,
                    "  Hamann: 1.23 eV / 2MR\n  (= 51 meV/atom for 24-cell)",
                    fontsize=7, va="top")

    fig.suptitle(
        "Fig. 4 — Microstate energies, sorted by 2MR presence and Q$^4$ content",
        y=0.995, fontsize=11, x=0.04, ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    pdf = OUT / "fig4_energy_2mr_q4.pdf"
    png = OUT / "fig4_energy_2mr_q4.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"Wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
