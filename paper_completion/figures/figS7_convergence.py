"""
Fig. S7 — Convergence of selected ensemble averages with subset size.
For the 24-atom ensemble (largest N), redraw the Boltzmann-weighted average
of (a) Q^4 fraction, (b) R_N(6) (six-membered rings per Si), and
(c) Si–O–Si peak position over a sequence of growing random subsets.

Empirical 1/sqrt(N) bands shown for reference; this is the convergence
behaviour Wolf et al. (JAP, 2025) demonstrated for short-range observables.
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
    load_qn, load_rings, load_bad, boltzmann_weights, load_density,
)

OUT = Path(__file__).parent
N_TRIALS = 12   # bootstrap replicates per subset size


def boltzmann_subset_qn(ids_subset, energies, n_atoms_map, qn_map):
    w = boltzmann_weights(ids_subset, energies, n_atoms_map)
    out = 0.0
    for sid in ids_subset:
        out += w[sid] * qn_map[sid]["qn_fractions"].get(4, 0.0)
    return out


def boltzmann_subset_rn(ids_subset, energies, n_atoms_map, rings_map, n=6):
    w = boltzmann_weights(ids_subset, energies, n_atoms_map)
    out = 0.0
    for sid in ids_subset:
        out += w[sid] * rings_map[sid]["RN"].get(n, 0.0)
    return out


def boltzmann_subset_sios_peak(ids_subset, energies, n_atoms_map, bad_map):
    w = boltzmann_weights(ids_subset, energies, n_atoms_map)
    edges = np.linspace(60, 180, 241)
    centers = 0.5 * (edges[1:] + edges[:-1])
    bw = edges[1] - edges[0]
    pdf = np.zeros(len(centers))
    for sid in ids_subset:
        a = np.asarray(bad_map[sid]["si_o_si_angles"])
        if a.size == 0: continue
        h, _ = np.histogram(a, bins=edges)
        if h.sum() > 0:
            pdf += w[sid] * (h / (h.sum() * bw))
    if (pdf * bw).sum() > 0:
        pdf /= (pdf * bw).sum()
    return peak_of_hist(centers, pdf)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))

    for sz, ax_i in zip((24, 36, 48), range(3)):
        e = SizeEnsemble(sz)
        ids = e.ids
        n_max = len(ids)
        if n_max < 30:
            continue
        # subset sizes: log-spaced
        sub_sizes = np.unique(np.round(np.logspace(np.log10(20), np.log10(n_max), 14)).astype(int))
        sub_sizes = sub_sizes[sub_sizes <= n_max]

        # cache observables once
        qn = load_qn(ids); rings = load_rings(ids); bad = load_bad(ids)
        energies = e.energies
        n_atoms_map = e.n_atoms

        q4_means, rn_means, sios_means = [], [], []
        for ns in sub_sizes:
            tq, tr, ts_ = [], [], []
            for _ in range(N_TRIALS):
                samp = list(rng.choice(ids, size=ns, replace=False))
                tq.append(boltzmann_subset_qn(samp, energies, n_atoms_map, qn))
                tr.append(boltzmann_subset_rn(samp, energies, n_atoms_map, rings, n=6))
                ts_.append(boltzmann_subset_sios_peak(samp, energies, n_atoms_map, bad))
            q4_means.append((np.mean(tq), np.std(tq)))
            rn_means.append((np.mean(tr), np.std(tr)))
            sios_means.append((np.mean(ts_), np.std(ts_)))

        q4 = np.array(q4_means); rn = np.array(rn_means); sios = np.array(sios_means)
        for ax_idx, (arr, ylabel) in enumerate([
            (q4,   r"$\langle$Q$^4\rangle$ fraction"),
            (rn,   r"$R_N(n=6)$"),
            (sios, r"Si–O–Si peak (deg)"),
        ]):
            ax = axes[ax_idx]
            ax.errorbar(sub_sizes, arr[:, 0], yerr=arr[:, 1],
                        marker="o", ms=4, lw=1.0, capsize=2,
                        color=SIZE_COLORS[sz], label=f"{SIZE_LABELS[sz]}")
            ax.set_xlabel("subset size $N$")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_xlim(left=10)
            ax.grid(alpha=0.25, ls=":")
            if ax_idx == 2:
                ax.legend(frameon=False, loc="best")

    for ax, t in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.set_title(t, loc="left")

    fig.suptitle("Fig. S7 — Convergence of ensemble averages with subset size "
                 "(error bars = std over 12 random redraws)",
                 x=0.04, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf = OUT / "figS7_convergence.pdf"
    png = OUT / "figS7_convergence.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    print(f"Wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
