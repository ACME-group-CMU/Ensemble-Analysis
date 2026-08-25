"""Bond angle distributions (Si-O-Si and O-Si-O).

Angles come from the true displacement vectors to the specific periodic images that
are actually bonded. The `legacy` variant reproduces the pre-refactor law-of-cosines
on three independently minimum-imaged distances, which can mix images and yield an
inconsistent triangle; it is kept only for comparison.

Reference values compared against: Mauri et al., Phys. Rev. B 62, R4786 (2000) --
NMR-derived Si-O-Si (151 +/- 11 deg); Sarnthein, Pasquarello & Car, Phys. Rev. B
52, 12690 (1995) -- CPMD Si-O-Si (136-142 deg) and O-Si-O (109.5 deg, FWHM ~9 deg).
"""
import numpy as np

from src import bonding

ANGLE_RANGE = (0.0, 180.0)
N_BINS = 180


def _angle_between(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def bond_angles(structure, cutoff=bonding.SI_O_CUTOFF, legacy=False):
    """Return {'si_o_si': array, 'o_si_o': array} in degrees."""
    if legacy:
        return _bond_angles_legacy(structure, cutoff)

    table = bonding.neighbor_table(structure, cutoff)
    si_o_si, o_si_o = [], []
    for i, site in enumerate(structure):
        arms = table.get(i, [])
        target = si_o_si if site.specie.symbol == 'O' else o_si_o
        for a in range(len(arms)):
            for b in range(a + 1, len(arms)):
                target.append(_angle_between(arms[a][2], arms[b][2]))
    return {'si_o_si': np.array(si_o_si), 'o_si_o': np.array(o_si_o)}


def _bond_angles_legacy(structure, cutoff):
    si_idx = bonding.indices(structure, 'Si')
    o_idx = bonding.indices(structure, 'O')

    def law_of_cosines(a, apex, b):
        d1 = structure.get_distance(apex, a)
        d2 = structure.get_distance(apex, b)
        d12 = structure.get_distance(a, b)
        cos = (d1**2 + d2**2 - d12**2) / (2 * d1 * d2)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    si_o_si = []
    for o in o_idx:
        bonded = [si for si in si_idx if structure.get_distance(o, si) < cutoff]
        if len(bonded) == 2:
            si_o_si.append(law_of_cosines(bonded[0], o, bonded[1]))

    o_si_o = []
    for si in si_idx:
        bonded = [o for o in o_idx if structure.get_distance(si, o) < cutoff]
        for a in range(len(bonded)):
            for b in range(a + 1, len(bonded)):
                o_si_o.append(law_of_cosines(bonded[a], si, bonded[b]))

    return {'si_o_si': np.array(si_o_si), 'o_si_o': np.array(o_si_o)}


def histogram(angles, angle_range=ANGLE_RANGE, bins=N_BINS):
    """Probability density, normalised per structure so every structure carries
    equal weight in an ensemble average regardless of how many angles it has."""
    edges = np.linspace(angle_range[0], angle_range[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(angles, bins=edges)
    width = edges[1] - edges[0]
    total = counts.sum()
    density = counts / (total * width) if total else counts.astype(float)
    return centers, density


def angle_distribution(structure, cutoff=bonding.SI_O_CUTOFF, legacy=False,
                       angle_range=ANGLE_RANGE, bins=N_BINS):
    angles = bond_angles(structure, cutoff, legacy)
    out = {'cutoff': cutoff, 'legacy': legacy}
    for key, arr in angles.items():
        centers, density = histogram(arr, angle_range, bins)
        out[key] = arr
        out[f'{key}_hist'] = density
        out[f'n_{key}'] = len(arr)
    out['bin_centers'] = centers
    return out
