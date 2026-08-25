"""
Fig. S10 — Effect of topology filter on ensemble observables.

Compares four ensemble variants for each system size:
  - Unfiltered (all structures)
  - 2MR filter  (remove any structure containing ≥1 two-membered ring)
  - Q⁴ filter   (remove any structure with a non-Q⁴ Si atom)
  - Both filters combined

Reports Q⁴ fraction, R_N(2), and Si–O–Si peak position, each re-weighted
with Boltzmann weights computed over the surviving subset. Experimental
references shown as horizontal dashed lines.
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

EXP_Q4   = 1.00
EXP_2MR  = 0.0
EXP_SIOS = 151.0

FILTER_LABELS = ["Unfiltered", "2MR filter", "Q⁴ filter", "Both"]
FILTER_MARKERS = ["o", "s", "^", "D"]
FILTER_ALPHA   = [1.0, 0.85, 0.85, 0.85]


def compute_averages(ids_f, ens, qn_map, rings_map, bad_map):
    if len(ids_f) < 2:
        return np.nan, np.nan, np.nan, 0

    w = boltzmann_weights(ids_f, ens.energies, ens.n_atoms)

    q4  = sum(w[sid] * qn_map[sid]["qn_fractions"].get(4, 0.0) for sid in ids_f)
    rn2 = sum(w[sid] * rings_map[sid]["RN"].get(2, 0.0) for sid in ids_f)

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

    return q4, rn2, sios, len(ids_f)


def apply_filters(ens, qn_map, rings_map):
    """Return four filtered id lists: unfiltered, 2MR, Q4, both."""
    ids = ens.ids

    has_2mr = {sid: rings_map[sid]["RN"].get(2, 0.0) > 0 for sid in ids}
    has_nonq4 = {sid: qn_map[sid]["qn_fractions"].get(4, 0.0) < 1.0 for sid in ids}

    return [
        ids,                                                      # unfiltered
        [sid for sid in ids if not has_2mr[sid]],                 # no 2MR
        [sid for sid in ids if not has_nonq4[sid]],               # all Q4
        [sid for sid in ids if not has_2mr[sid] and not has_nonq4[sid]],  # both
    ]


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    sizes = [24, 36, 48]
    x = np.arange(len(sizes))
    n_filters = 4
    width = 0.18
    offsets = np.linspace(-(n_filters - 1) / 2, (n_filters - 1) / 2, n_filters) * width

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    ylabels = [r"$\langle$Q$^4\rangle$ fraction", r"$R_N(n=2)$", r"Si–O–Si peak (deg)"]
    refs    = [EXP_Q4, EXP_2MR, EXP_SIOS]
    ref_labels = ["expt. (~100%)", "expt. (~0)", "NMR 151°"]

    # collect results: results[filter_idx][obs_idx] = list over sizes
    results = [[[] for _ in range(3)] for _ in range(n_filters)]
    n_structs = [[] for _ in range(n_filters)]  # for annotation

    for sz in sizes:
        ens = SizeEnsemble(sz)
        qn_map    = load_qn(ens.ids)
        rings_map = load_rings(ens.ids)
        bad_map   = load_bad(ens.ids)

        filter_sets = apply_filters(ens, qn_map, rings_map)
        for fi, ids_f in enumerate(filter_sets):
            q4, rn2, sios, nk = compute_averages(ids_f, ens, qn_map, rings_map, bad_map)
            results[fi][0].append(q4)
            results[fi][1].append(rn2)
            results[fi][2].append(sios)
            n_structs[fi].append(nk)

    # distinct colors per filter
    filter_colors = ["#555555", "#1f77b4", "#e07b39", "#9467bd"]

    for fi, (label, marker, alpha) in enumerate(zip(FILTER_LABELS, FILTER_MARKERS, FILTER_ALPHA)):
        for obs_idx, ax in enumerate(axes):
            vals = results[fi][obs_idx]
            ax.bar(x + offsets[fi], vals, width=width,
                   color=filter_colors[fi], alpha=alpha,
                   label=label if obs_idx == 2 else None,
                   edgecolor="white", linewidth=0.5)

    # reference lines
    for ax, ref, rl in zip(axes, refs, ref_labels):
        ax.axhline(ref, color="k", ls="--", lw=1.0, label=rl if ax is axes[2] else None)

    for ax, yl in zip(axes, ylabels):
        ax.set_xticks(x)
        ax.set_xticklabels([SIZE_LABELS[sz] for sz in sizes])
        ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.25, ls=":")

    # annotate N surviving per filter on panel (a)
    for fi in range(n_filters):
        for xi, sz in enumerate(sizes):
            nk = n_structs[fi][xi]
            axes[0].annotate(f"{nk}", xy=(xi + offsets[fi], results[fi][0][xi]),
                              ha="center", va="bottom", fontsize=6, rotation=90,
                             color=filter_colors[fi])

    axes[2].legend(frameon=False, loc="upper right", ncol=1)

    for ax, t in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.set_title(t, loc="left")

    fig.suptitle(
        "Fig. S10 — Ensemble observables under topology filters "
        "(numbers above bars = structures surviving each filter)",
        x=0.04, ha="left", fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    pdf = OUT / "figS10_topology_filter.pdf"
    png = OUT / "figS10_topology_filter.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    print(f"Wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
