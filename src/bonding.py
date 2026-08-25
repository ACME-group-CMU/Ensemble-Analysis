"""Bond topology for v-SiO2 — the single source of truth for who is bonded to whom.

Every observable (rings, angles, Q^n, coordination) derives its connectivity from
here, so the Si-O cutoff and the periodic-image handling cannot drift apart between
methods. Neighbours carry their periodic image, which is what makes image-consistent
angles and self-image ring detection possible.
"""
from collections import defaultdict

import numpy as np

SI_O_CUTOFF = 2.0


def neighbor_table(structure, cutoff=SI_O_CUTOFF):
    """Map site index -> list of (neighbour_index, image, displacement_vector).

    `image` is the integer lattice translation of the neighbour relative to the
    base cell; `displacement` is the true cartesian vector from site to that image.
    Only Si-O pairs are returned: Si-Si and O-O are not bonds in this system.
    """
    table = defaultdict(list)
    for i, site in enumerate(structure):
        for nbr in structure.get_neighbors(site, cutoff):
            if nbr.specie.symbol == site.specie.symbol:
                continue
            image = tuple(int(round(x)) for x in nbr.image)
            table[i].append((nbr.index, image, nbr.coords - site.coords))
    return dict(table)


def indices(structure, symbol):
    return [i for i, s in enumerate(structure) if s.specie.symbol == symbol]


def coordination(structure, cutoff=SI_O_CUTOFF):
    """Raw coordination number per site (Si-O bonds only), as {index: n}."""
    table = neighbor_table(structure, cutoff)
    return {i: len(table.get(i, [])) for i in range(len(structure))}


def si_per_oxygen(structure, cutoff=SI_O_CUTOFF):
    """How many Si each O is bonded to. 2 == bridging, 1 == terminal, 3+ == over-coordinated."""
    table = neighbor_table(structure, cutoff)
    return {i: len(table.get(i, [])) for i in indices(structure, 'O')}


def si_si_shared_oxygens(structure, cutoff=SI_O_CUTOFF):
    """Map (si_i, si_j, relative_image) -> list of shared O.

    Two Si sharing more than one O form an edge-sharing pair, i.e. a 2-membered ring.
    Keyed by relative image so a Si bonded to its own periodic image is distinguishable
    from a genuine distinct-atom pair.
    """
    table = neighbor_table(structure, cutoff)
    shared = defaultdict(list)
    for o_idx in indices(structure, 'O'):
        arms = table.get(o_idx, [])
        for a in range(len(arms)):
            for b in range(a + 1, len(arms)):
                si_a, img_a, _ = arms[a]
                si_b, img_b, _ = arms[b]
                rel = tuple(np.array(img_b) - np.array(img_a))
                if si_a > si_b or (si_a == si_b and rel < (0, 0, 0)):
                    si_a, si_b, rel = si_b, si_a, tuple(-np.array(rel))
                shared[(si_a, si_b, rel)].append(o_idx)
    return dict(shared)
