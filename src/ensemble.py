"""Boltzmann weighting and ensemble averaging.

Weights use the per-atom DFT energy, which is what makes different cell sizes
comparable: exp(-E_atom/kT) rather than exp(-E_cell/kT) means the effective cell
temperature scales with N. E_min is subtracted before exponentiating to avoid
underflow.

RSL (Random Structure/Superlattice sampling) and its Boltzmann-weighted ensemble
averaging follow Stevanović, Phys. Rev. Lett. (2016) and Jones & Stevanović, npj
Comput. Mater. (2020); the partial-ergodicity argument for treating a Boltzmann
average over RSL basins as representative of the glass ensemble is theirs.
"""
import numpy as np

KB_RY = 6.33362e-6
T_BOLTZMANN = 2000.0


def weights(energies_per_atom, temperature=T_BOLTZMANN):
    """Normalised Boltzmann weights from per-atom energies in Ry."""
    e = np.asarray(energies_per_atom, dtype=float)
    w = np.exp(-(e - e.min()) / (KB_RY * temperature))
    return w / w.sum()


def effective_sample_size(w):
    """Kish effective N — how many structures the weighted average really rests on."""
    w = np.asarray(w, dtype=float)
    return 1.0 / np.sum(w ** 2)


def average(values, w):
    """Weighted mean over the leading axis. NaNs are ignored per element, with the
    weights renormalised over whatever contributed to each element."""
    values = np.asarray(values, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1, *([1] * (values.ndim - 1)))
    mask = np.isfinite(values)
    num = np.nansum(np.where(mask, values * w, 0.0), axis=0)
    den = np.sum(np.where(mask, w, 0.0), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def average_binned(values, counts, w):
    """Weighted mean where each structure contributes a per-bin sample count,
    as with reciprocal-lattice S(q): bins backed by more k-points count for more."""
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    w = np.asarray(w, dtype=float)[:, None]
    mask = np.isfinite(values) & (counts > 0)
    num = np.sum(np.where(mask, np.nan_to_num(values) * counts * w, 0.0), axis=0)
    den = np.sum(np.where(mask, counts * w, 0.0), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def average_dicts(dicts, w):
    """Weighted mean of {key: value} maps, e.g. ring counts by size."""
    out = {}
    for weight, d in zip(w, dicts):
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + weight * v
    return out
