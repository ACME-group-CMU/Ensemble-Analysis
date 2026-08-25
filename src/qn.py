"""Q^n speciation: how many bridging oxygens each Si carries.

An O is bridging iff it is bonded to exactly 2 Si. The `legacy` variant reproduces
the pre-refactor definition (any O with >=1 other Si counted as bridging, and any Si
with 5+ bridging O silently dropped) so the two can be compared directly.

Q^n speciation is standard glass-network notation; the non-Q^4 fraction as a
diagnostic of strained/defective Si in RSL ensembles follows this project's own
methods (Stevanović, Phys. Rev. Lett. (2016); Jones & Stevanović, npj Comput.
Mater. (2020)).
"""
import numpy as np

from src import bonding

MAX_N = 8


def qn_distribution(structure, cutoff=bonding.SI_O_CUTOFF, legacy=False):
    """Return {'counts': {n: n_si}, 'fractions': {n: frac}, 'total_si': int}.

    Fractions always sum to 1: no Si is dropped, whatever its coordination.
    """
    table = bonding.neighbor_table(structure, cutoff)
    si_idx = bonding.indices(structure, 'Si')
    n_si_of_o = bonding.si_per_oxygen(structure, cutoff)

    counts = {}
    dropped = 0
    for si in si_idx:
        bonded_o = [o for o, _, _ in table.get(si, [])]
        if legacy:
            n_bridging = sum(1 for o in bonded_o if n_si_of_o[o] >= 2)
            if n_bridging > 4:
                dropped += 1
                continue
        else:
            n_bridging = sum(1 for o in bonded_o if n_si_of_o[o] == 2)
        counts[n_bridging] = counts.get(n_bridging, 0) + 1

    total = len(si_idx)
    return {
        'counts': counts,
        'fractions': {n: c / total for n, c in counts.items()},
        'total_si': total,
        'dropped_si': dropped,
        'cutoff': cutoff,
        'legacy': legacy,
    }


def as_vector(result, max_n=MAX_N):
    """Fractions as a dense array indexed by n, for ensemble averaging."""
    vec = np.zeros(max_n + 1)
    for n, f in result['fractions'].items():
        if n <= max_n:
            vec[n] += f
    return vec
