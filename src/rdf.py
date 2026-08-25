"""Structure factor and pair distribution functions.

S(q) is evaluated directly on the reciprocal lattice of the cell itself:

    S_FZ(q) = 1 + (|F(k)|^2 / N - <b^2>) / <b>^2 ,   F(k) = sum_j b_j exp(i k.r_j)

restricted to the k commensurate with the periodic cell, then spherically binned.
This is exact for a periodic cell and needs no minimum-image convention, no cutoff
sphere and no density correction -- all of which are ill-defined for cells this
small and strongly triclinic. The Debye variant is retained for comparison only.

Method attribution:
    Faber, T. E. & Ziman, J. M. Phil. Mag. 11, 153 (1965) -- partial structure
        factor formalism (Faber-Ziman weighting) used for S_total.
    Grimley, D. I., Wright, A. C. & Sinclair, R. N. J. Non-Cryst. Solids 119, 49
        (1990) -- primary neutron S(q) reference this is validated against.
"""
import numpy as np

XRAY_Z = {'Si': 14.0, 'O': 8.0}
NEUTRON_B = {'Si': 4.1491, 'O': 5.803}


def scattering_lengths(structure, weighting='neutron'):
    if weighting == 'neutron':
        table = NEUTRON_B
    elif weighting == 'xray':
        table = XRAY_Z
    elif weighting == 'unweighted':
        table = {'Si': 1.0, 'O': 1.0}
    else:
        raise ValueError(f'unknown weighting {weighting!r}')
    return np.array([table[s.specie.symbol] for s in structure.sites])


def structure_factor(structure, q_edges, weighting='neutron'):
    """Exact S(q) on the cell's reciprocal lattice.

    Returns (S, counts) per q bin; S is NaN where no k-vector falls in the bin.
    `counts` is the number of contributing k-points, needed to weight bins correctly
    when averaging over an ensemble.
    """
    b = scattering_lengths(structure, weighting)
    n_atoms = len(structure)
    b_mean, b2_mean = b.mean(), (b ** 2).mean()
    positions = np.array([s.coords for s in structure.sites])

    recip = 2 * np.pi * np.linalg.inv(structure.lattice.matrix).T
    q_max = q_edges[-1]
    n_max = [int(np.ceil(q_max / np.linalg.norm(recip[i]))) + 1 for i in range(3)]
    grids = np.meshgrid(*[np.arange(-n, n + 1) for n in n_max], indexing='ij')
    hkl = np.stack([g.ravel() for g in grids], axis=1)
    hkl = hkl[np.any(hkl != 0, axis=1)]

    k_vec = hkl @ recip
    k_mag = np.linalg.norm(k_vec, axis=1)
    keep = k_mag <= q_max
    k_vec, k_mag = k_vec[keep], k_mag[keep]

    amplitude = (b * np.exp(1j * (k_vec @ positions.T))).sum(axis=1)
    s_k = (np.abs(amplitude) ** 2 / n_atoms - b2_mean) / b_mean ** 2 + 1.0

    bin_idx = np.digitize(k_mag, q_edges) - 1
    n_bins = len(q_edges) - 1
    s_binned = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = bin_idx == i
        counts[i] = mask.sum()
        if counts[i]:
            s_binned[i] = s_k[mask].mean()
    return s_binned, counts


def structure_factor_debye(structure, q, weighting='neutron', r_max=None, lorch=True):
    """Debye sum over periodic images out to r_max, with the uniform-density term
    removed. Kept for comparison against `structure_factor`; the reciprocal-lattice
    method is the one to use for these cells."""
    b = scattering_lengths(structure, weighting)
    symbols = [s.specie.symbol for s in structure.sites]
    table = {'neutron': NEUTRON_B, 'xray': XRAY_Z,
             'unweighted': {'Si': 1.0, 'O': 1.0}}[weighting]
    n_atoms = len(structure)
    rho = n_atoms / structure.volume
    b_mean = b.mean()

    if r_max is None:
        m = structure.lattice.matrix
        widths = [structure.volume / np.linalg.norm(np.cross(m[i], m[j]))
                  for i, j in [(1, 2), (0, 2), (0, 1)]]
        r_max = 0.5 * min(widths)

    distances, weights = [], []
    for i, neighbors in enumerate(structure.get_all_neighbors(r_max)):
        for nbr in neighbors:
            if nbr.nn_distance > 1e-8:
                distances.append(nbr.nn_distance)
                weights.append(b[i] * table[nbr.specie.symbol])
    r = np.array(distances)
    w = np.array(weights)
    window = np.sinc(r / r_max) if lorch else np.ones_like(r)

    qr = np.outer(q, r)
    pair_term = (np.sin(qr) / qr * (w * window)).sum(axis=1) / (n_atoms * b_mean ** 2)
    q_rmax = q * r_max
    density_term = (4 * np.pi * rho / q ** 3) * (np.sin(q_rmax) - q_rmax * np.cos(q_rmax))
    return 1.0 + pair_term - density_term


def pair_distribution(structure, r_edges, pair=None, cutoff=None):
    """Partial or total g(r) in real space, from true periodic distances.

    `pair` is e.g. ('Si', 'O'); None gives the total g(r).
    """
    r_max = r_edges[-1] if cutoff is None else cutoff
    symbols = [s.specie.symbol for s in structure.sites]
    if pair is None:
        src_idx = list(range(len(structure)))
        target = None
    else:
        src_idx = [i for i, s in enumerate(symbols) if s == pair[0]]
        target = pair[1]

    distances = []
    all_neighbors = structure.get_all_neighbors(r_max)
    for i in src_idx:
        for nbr in all_neighbors[i]:
            if target is None or nbr.specie.symbol == target:
                if nbr.nn_distance > 1e-8:
                    distances.append(nbr.nn_distance)

    counts, _ = np.histogram(distances, bins=r_edges)
    centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    shell = 4 * np.pi * centers ** 2 * np.diff(r_edges)

    n_target = len(structure) if target is None else symbols.count(target)
    rho_target = n_target / structure.volume
    norm = len(src_idx) * rho_target * shell
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(norm > 0, counts / norm, 0.0)
    return centers, g
