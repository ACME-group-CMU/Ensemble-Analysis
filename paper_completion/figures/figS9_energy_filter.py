"""
Fig. S9 — Effect of energy filter on ensemble observables.

For each system size, sweep a per-atom energy cutoff ΔE (eV above the
ensemble minimum). At each threshold keep only structures within that window
and recompute Boltzmann-weighted Q⁴ fraction, R_N(2), and Si–O–Si peak.
Shows how the three key observables approach (or don't approach) experimental
reference values as the strained high-energy tail is progressively excluded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_completion._load import (
    SizeEnsemble, SIZE_COLORS, SIZE_LABELS,
    load_qn, load_rings, load_bad,
    boltzmann_weights, peak_of_hist,
)

OUT = Path(__file__).parent
RY_TO_EV = 13.6057  # 1 Ry = 13.6057 eV

# Experimental / reference values
EXP_Q4   = 1.00        # pure v-SiO2 ~ 100% Q4
EXP_2MR  = 0.0         # no 2MRs in experimental glass
EXP_SIOS = 151.0       # Mauri 2000 NMR


def filtered_averages(ens, qn_map, rings_map, bad_map, de_ev):
    """Boltzmann averages over structures within de_ev of the per-atom minimum."""
    e_min = min(ens.energies[sid] / ens.n_atoms[sid] for sid in ens.ids)
    ids_f = [sid for sid in ens.ids
             if (ens.energies[sid] / ens.n_atoms[sid] - e_min) * RY_TO_EV <= de_ev]
    if len(ids_f) < 2:
        return None, None, None

    w = boltzmann_weights(ids_f, ens.energies, ens.n_atoms)

    # Q4
    q4 = sum(w[sid] * qn_map[sid]["qn_fractions"].get(4, 0.0) for sid in ids_f)

    # R_N(2)
    rn2 = sum(w[sid] * rings_map[sid]["RN"].get(2, 0.0) for sid in ids_f)

    # Si-O-Si peak
    edges = np.linspace(60, 180, 241)
    centers = 0.5 * (edges[1:] + edges[:-1])
    bw = edges[1] - edges[0]
    pdf = np.zeros(len(centers))
    for sid in ids_f:
        a = np.asarray(bad_map[sid]["si_o_si_angles"])
        if a.size == 0:
            continue
        h, _ = np.histogram(a, bins=edges)
        if h.sum() > 0:
            pdf += w[sid] * (h / (h.sum() * bw))
    area = (pdf * bw).sum()
    if area > 0:
        pdf /= area
    sios = peak_of_hist(centers, pdf)

    return q4, rn2, sios


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    # ΔE sweep: 0.05 to 3.0 eV/atom, log-spaced
    de_vals = np.concatenate([
        np.linspace(0.05, 0.5, 12),
        np.linspace(0.6, 3.0, 10),
    ])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    ylabels = [r"$\langle$Q$^4\rangle$ fraction", r"$R_N(n=2)$", r"Si–O–Si peak (deg)"]
    refs    = [EXP_Q4, EXP_2MR, EXP_SIOS]
    ref_labels = ["expt. (~100%)", "expt. (~0)", "NMR (Mauri 2000, 151°)"]

    for sz in (24, 36, 48):
        ens = SizeEnsemble(sz)
        qn_map    = load_qn(ens.ids)
        rings_map = load_rings(ens.ids)
        bad_map   = load_bad(ens.ids)

        # full-ensemble values (ΔE = ∞)
        e_min = min(ens.energies[sid] / ens.n_atoms[sid] for sid in ens.ids)
        e_max_ev = max((ens.energies[sid] / ens.n_atoms[sid] - e_min) * RY_TO_EV
                       for sid in ens.ids)

        q4s, rn2s, sioss, n_kept = [], [], [], []
        for de in de_vals:
            q4, rn2, sios = filtered_averages(ens, qn_map, rings_map, bad_map, de)
            if q4 is None:
                q4s.append(np.nan); rn2s.append(np.nan); sioss.append(np.nan)
            else:
                q4s.append(q4); rn2s.append(rn2); sioss.append(sios)
            nk = sum(1 for sid in ens.ids
                     if (ens.energies[sid] / ens.n_atoms[sid] - e_min) * RY_TO_EV <= de)
            n_kept.append(nk)

        for ax_idx, (vals, ax) in enumerate(zip([q4s, rn2s, sioss], axes)):
            ax.plot(de_vals, vals, color=SIZE_COLORS[sz], lw=1.5,
                    label=SIZE_LABELS[sz])

    # reference lines
    for ax, ref, rl in zip(axes, refs, ref_labels):
        ax.axhline(ref, color="k", ls="--", lw=1.0, label=rl)

    for ax, yl in zip(axes, ylabels):
        ax.set_xlabel(r"$\Delta E$ cutoff (eV / atom)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25, ls=":")

    axes[2].legend(frameon=False, loc="best")

    for ax, t in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.set_title(t, loc="left")

    fig.suptitle(
        "Fig. S9 — Ensemble observables vs energy filter cutoff "
        r"($\Delta E$ per atom above ensemble minimum)",
        x=0.04, ha="left", fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    pdf = OUT / "figS9_energy_filter.pdf"
    png = OUT / "figS9_energy_filter.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    print(f"Wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
