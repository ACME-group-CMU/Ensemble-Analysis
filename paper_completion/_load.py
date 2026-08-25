"""Adapter from the figure scripts onto the refactored pipeline in src/.

The figure scripts keep their original API; the data underneath now comes from
`data/observables/` via `src.analysis`, so they pick up the corrected conventions.

Two behaviours deliberately changed, because the old ones were wrong:
  * `avg_qn_fractions` no longer renormalises. Fractions sum to 1 on their own now
    that no Si is dropped; renormalising previously hid that loss.
  * `avg_rn` returns unique rings per Si (RC), the convention of Rino 1993 and of
    the project's own definition of R_N. The old per-incidence count is still
    available as `avg_rn(convention='RN')`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis, bonding, ensemble, io, observables  # noqa: E402

T_BOLTZ = ensemble.T_BOLTZMANN
SIZE_COLORS = {24: '#1f4e8c', 36: '#2a9d8f', 48: '#e76f51'}
SIZE_LABELS = {24: '24-atom', 36: '36-atom', 48: '48-atom'}

DEFAULT_RING_CONVENTION = 'RC_local'


def common_struct_ids(size):
    return [s for s in io.structure_ids(size) if observables.is_current(s)]


def _cached(struct_ids):
    return {sid: observables.load(sid) for sid in struct_ids}


def load_qn(struct_ids):
    out = {}
    for sid, d in _cached(struct_ids).items():
        q = dict(d['qn'])
        q['qn_fractions'] = q['fractions']
        q['qn_counts'] = q['counts']
        out[sid] = q
    return out


def load_bad(struct_ids):
    out = {}
    for sid, d in _cached(struct_ids).items():
        bad = dict(d['bad'])
        bad['si_o_si_angles'] = bad['si_o_si']
        bad['o_si_o_angles'] = bad['o_si_o']
        bad['angles'] = bad['si_o_si']
        out[sid] = bad
    return out


def load_rings(struct_ids):
    return {sid: d['rings'] for sid, d in _cached(struct_ids).items()}


def load_density(struct_ids):
    out = {}
    for sid, d in _cached(struct_ids).items():
        n_si = d['n_atoms'] // 3
        out[sid] = {
            'density': d['density'],
            'volume': d['volume'],
            'num_atoms': d['n_atoms'],
            'element_counts': {'Si': n_si, 'O': 2 * n_si},
            'partial_densities': {'Si': n_si / d['volume'], 'O': 2 * n_si / d['volume']},
        }
    return out


def boltzmann_weights(struct_ids, energies, n_atoms_per_id, temperature=T_BOLTZ):
    per_atom = [energies[s] / n_atoms_per_id[s] for s in struct_ids]
    w = ensemble.weights(per_atom, temperature)
    return dict(zip(struct_ids, w))


class SizeEnsemble(analysis.SizeEnsemble):
    """`analysis.SizeEnsemble` with the names the figure scripts already use."""

    def __init__(self, size, ids=None, temperature=T_BOLTZ):
        super().__init__(size, temperature)
        if ids is not None:
            keep = [i for i, s in enumerate(self.ids) if s in set(ids)]
            self.ids = [self.ids[i] for i in keep]
            self.data = [self.data[i] for i in keep]
            self.energies = self.energies[keep]
            self.weights = ensemble.weights(self.energies, temperature)
        self.n_atoms = {d['struct_id']: d['n_atoms'] for d in self.data}
        self.volumes = {d['struct_id']: d['volume'] for d in self.data}
        self.n_si = {d['struct_id']: d['n_atoms'] // 3 for d in self.data}
        self.weights_by_id = dict(zip(self.ids, self.weights))
        # the figure scripts expect `energies` keyed by id and holding TOTAL energy
        self.energies_per_atom = self.energies
        self.energies = {d['struct_id']: d['energy_total_ry'] for d in self.data}

    def weight_array(self):
        return self.weights

    def per_atom_energies(self):
        return self.energies_per_atom

    def per_atom_energy(self, sid):
        return self.energies[sid] / self.n_atoms[sid]

    def avg_qn_fractions(self, max_n=8):
        vec = self.qn_fractions(legacy=False)
        return {n: float(vec[n]) for n in range(max_n + 1)}

    def per_struct_qn_fractions(self):
        return {d['struct_id']: dict(d['qn']['fractions']) for d in self.data}

    def avg_rn(self, max_n=14, convention=DEFAULT_RING_CONVENTION):
        rings = self.rings(convention)
        return {n: rings.get(n, 0.0) for n in range(2, max_n + 1)}

    def per_struct_rn(self, max_n=14, convention=DEFAULT_RING_CONVENTION):
        return {d['struct_id']: {n: d['rings'][convention].get(n, 0.0)
                                 for n in range(2, max_n + 1)}
                for d in self.data}

    def avg_bad(self, kind='si_o_si', angle_range=(60.0, 180.0), bins=240):
        centers, density = self.bad(kind, legacy=False)
        mask = (centers >= angle_range[0]) & (centers <= angle_range[1])
        centers, density = centers[mask], density[mask]
        moments = self.bad_moments(kind, legacy=False)
        return {'centers': centers, 'pdf': density,
                'mean': moments['mean'], 'std': moments['std'],
                'peak': moments['peak'],
                'fwhm': fwhm_of_hist(centers, density)}

    def avg_total_sq(self):
        return self.structure_factor()

    def avg_partial_g(self, pair='Si_O'):
        key = tuple(pair.split('_')) if isinstance(pair, str) else tuple(pair)
        return self.g_r(pair=key)

    def avg_total_dr(self):
        """D(r) = 4 pi r rho (g(r) - 1), returned alongside g(r)."""
        r, g = self.g_r()
        return r, 4 * np.pi * r * self.density * (g - 1.0), g


def peak_of_hist(centers, pdf):
    return float(np.asarray(centers)[int(np.nanargmax(pdf))])


def fwhm_of_hist(centers, pdf):
    centers, pdf = np.asarray(centers, float), np.asarray(pdf, float)
    peak = np.nanmax(pdf)
    if not np.isfinite(peak) or peak <= 0:
        return float('nan')
    above = np.where(pdf >= 0.5 * peak)[0]
    if len(above) < 2:
        return float('nan')
    return float(centers[above[-1]] - centers[above[0]])


def first_peak_position(r, y, r_min=1.4, r_max=1.9):
    r, y = np.asarray(r, float), np.asarray(y, float)
    m = (r >= r_min) & (r <= r_max)
    return float(r[m][int(np.nanargmax(y[m]))]) if m.any() else float('nan')


def coordination_number(r, g, density_partner, r_cut):
    r, g = np.asarray(r, float), np.asarray(g, float)
    m = r <= r_cut
    return float(np.trapezoid(4 * np.pi * r[m] ** 2 * density_partner * g[m], r[m]))


def boltzmann_avg_coordination(ens, e_center='Si', e_partner='O',
                               cutoff=bonding.SI_O_CUTOFF, **_):
    pmf = boltzmann_coord_distribution(ens, e_center, e_partner, cutoff)
    return float((np.arange(len(pmf)) * pmf).sum())


def boltzmann_coord_distribution(ens, e_center='Si', e_partner='O',
                                 cutoff=bonding.SI_O_CUTOFF, max_n=8, **_):
    """Coordination-number PMF as an array indexed by n."""
    vec = ens.coordination('Si' if e_center == 'Si' else 'O')
    return np.asarray(vec[:max_n + 1], dtype=float)
