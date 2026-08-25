"""Compute and cache every observable for a structure.

One pickle per structure under data/observables/, holding all four observables plus
the legacy variants, so a figure never has to re-derive anything and the corrected
and legacy conventions can be compared without a second pass over the data.
"""
import os
import pickle

import numpy as np

from src import angles, bonding, io, qn, rdf, rings

CACHE_DIR = os.path.join(io.DATA_DIR, 'observables')
SCHEMA = 3

Q_EDGES = np.linspace(0.4, 12.0, 117)
R_EDGES = np.linspace(0.0, 8.0, 321)
PAIRS = (('Si', 'O'), ('Si', 'Si'), ('O', 'O'))


def cache_path(struct_id):
    return os.path.join(CACHE_DIR, f'{struct_id}.pkl')


def compute(struct_id, folder=io.POSCAR_DIR, cutoff=bonding.SI_O_CUTOFF):
    structure = io.load_structure(struct_id, folder)
    energy_total = io.load_energy(struct_id, folder)
    n_atoms = len(structure)

    s_q, s_counts = rdf.structure_factor(structure, Q_EDGES, weighting='neutron')
    g_total = rdf.pair_distribution(structure, R_EDGES)[1]
    g_partials = {p: rdf.pair_distribution(structure, R_EDGES, pair=p)[1] for p in PAIRS}

    ring_stats = rings.ring_statistics(structure, cutoff)
    coord = bonding.coordination(structure, cutoff)
    si_coord = [coord[i] for i in bonding.indices(structure, 'Si')]
    o_coord = [coord[i] for i in bonding.indices(structure, 'O')]

    return {
        'schema': SCHEMA,
        'struct_id': struct_id,
        'n_atoms': n_atoms,
        'volume': structure.volume,
        'density': n_atoms / structure.volume,
        'energy_total_ry': energy_total,
        'energy_per_atom_ry': energy_total / n_atoms,
        'cutoff': cutoff,

        'q_edges': Q_EDGES,
        'S_q': s_q,
        'S_q_counts': s_counts,
        'r_edges': R_EDGES,
        'g_r': g_total,
        'g_r_partials': g_partials,

        'qn': qn.qn_distribution(structure, cutoff, legacy=False),
        'qn_legacy': qn.qn_distribution(structure, cutoff, legacy=True),

        'bad': angles.angle_distribution(structure, cutoff, legacy=False),
        'bad_legacy': angles.angle_distribution(structure, cutoff, legacy=True),

        'rings': ring_stats,
        'si_coordination': np.bincount(si_coord, minlength=9),
        'o_coordination': np.bincount(o_coord, minlength=9),
    }


def load(struct_id):
    with open(cache_path(struct_id), 'rb') as fh:
        return pickle.load(fh)


def is_current(struct_id):
    path = cache_path(struct_id)
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'rb') as fh:
            return pickle.load(fh).get('schema') == SCHEMA
    except Exception:
        return False


def populate(struct_id, folder=io.POSCAR_DIR, cutoff=bonding.SI_O_CUTOFF):
    """Compute and write one structure. Exceptions propagate — a failure here is a
    real problem and must not be silently skipped."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    result = compute(struct_id, folder, cutoff)
    with open(cache_path(struct_id), 'wb') as fh:
        pickle.dump(result, fh)
    return struct_id
