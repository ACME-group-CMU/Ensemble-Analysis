"""Ensemble-level results, built from the cached per-structure observables.

This is the layer figures and summary tables read from. Everything is Boltzmann
weighted within a cell size; nothing is averaged across sizes, since the sizes are
the independent variable of the study.
"""
import numpy as np

from src import ensemble, io, observables, qn as qn_mod

SIZE_COLORS = {24: '#1f4e8c', 36: '#2a9d8f', 48: '#e76f51'}


class SizeEnsemble:
    """All cached structures of one cell size, with Boltzmann weights."""

    def __init__(self, size, temperature=ensemble.T_BOLTZMANN):
        self.size = size
        self.temperature = temperature
        self.ids = [s for s in io.structure_ids(size) if observables.is_current(s)]
        if not self.ids:
            raise RuntimeError(f'no cached observables for size {size}; run regenerate.py')
        self.data = [observables.load(s) for s in self.ids]
        self.energies = np.array([d['energy_per_atom_ry'] for d in self.data])
        self.weights = ensemble.weights(self.energies, temperature)

    def __len__(self):
        return len(self.ids)

    @property
    def n_eff(self):
        return ensemble.effective_sample_size(self.weights)

    @property
    def density(self):
        return float(self.weights @ np.array([d['density'] for d in self.data]))

    # --- structure factor and pair distribution ---

    def structure_factor(self):
        q = 0.5 * (observables.Q_EDGES[:-1] + observables.Q_EDGES[1:])
        s = ensemble.average_binned(
            np.array([d['S_q'] for d in self.data]),
            np.array([d['S_q_counts'] for d in self.data]),
            self.weights)
        return q, s

    def g_r(self, pair=None):
        r = 0.5 * (observables.R_EDGES[:-1] + observables.R_EDGES[1:])
        key = 'g_r' if pair is None else None
        values = np.array([d['g_r'] if pair is None else d['g_r_partials'][pair]
                           for d in self.data])
        return r, ensemble.average(values, self.weights)

    # --- speciation ---

    def qn_fractions(self, legacy=False):
        key = 'qn_legacy' if legacy else 'qn'
        vecs = np.array([qn_mod.as_vector(d[key]) for d in self.data])
        return ensemble.average(vecs, self.weights)

    def coordination(self, species='Si'):
        key = 'si_coordination' if species == 'Si' else 'o_coordination'
        counts = np.array([d[key] / d[key].sum() for d in self.data])
        return ensemble.average(counts, self.weights)

    # --- angles ---

    def bad(self, kind='si_o_si', legacy=False):
        key = 'bad_legacy' if legacy else 'bad'
        centers = self.data[0][key]['bin_centers']
        hists = np.array([d[key][f'{kind}_hist'] for d in self.data])
        return centers, ensemble.average(hists, self.weights)

    def bad_moments(self, kind='si_o_si', legacy=False):
        """Weighted peak, mean and std of an angle distribution."""
        centers, density = self.bad(kind, legacy)
        width = centers[1] - centers[0]
        p = density * width
        p = p / p.sum()
        mean = float((centers * p).sum())
        std = float(np.sqrt((p * (centers - mean) ** 2).sum()))
        return {'peak': float(centers[np.argmax(density)]), 'mean': mean, 'std': std}

    # --- rings ---

    def rings(self, convention='RC_local'):
        """convention: 'RC_local' (unique, no self-image), 'RC' (unique, all),
        or 'RN' (incidences per Si)."""
        return ensemble.average_dicts([d['rings'][convention] for d in self.data],
                                      self.weights)

    def self_image_fraction(self):
        """Fraction of unique rings of each size that close through a periodic image."""
        total = ensemble.average_dicts([d['rings']['RC'] for d in self.data], self.weights)
        local = ensemble.average_dicts([d['rings']['RC_local'] for d in self.data], self.weights)
        return {n: 1.0 - local.get(n, 0.0) / v for n, v in total.items() if v > 0}

    def mean_2mr_per_cell(self, convention='RC_local'):
        n_si = self.size // 3
        return self.rings(convention).get(2, 0.0) * n_si


def all_sizes(temperature=ensemble.T_BOLTZMANN):
    return {s: SizeEnsemble(s, temperature) for s in io.SIZES}
