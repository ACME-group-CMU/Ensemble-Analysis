import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import NeighborList
from src.data_management_v2 import (load_energies, load_rdfs, load_counting_functions,
                                    load_structures, calculate_weights)


# =====================================================================
# ATOMIC SCATTERING PARAMETERS
# =====================================================================

XRAY_Z = {
    'Si': 14, 'O': 8,  'H': 1,  'Al': 13, 'Na': 11,
    'Ca': 20, 'Mg': 12,'Fe': 26,'Li': 3,  'K': 19,
    'B':  5,  'P': 15, 'S': 16, 'Cl': 17, 'Ti': 22, 'Zr': 40,
}

NEUTRON_B = {
    'Si':  4.1491, 'O':   5.803,  'H':  -3.7390, 'Al':  3.449,
    'Na':  3.630,  'Ca':  4.700,  'Mg':  5.375,  'Fe':  9.450,
    'Li': -1.90,   'K':   3.670,  'B':   5.30,   'P':   5.13,
    'S':   2.847,  'Cl':  9.577,  'Ti': -3.370,  'Zr':  7.16,
}


def _get_scattering_weights(elements, concentrations, weighting):
    if weighting == 'xray':
        f = {e: float(XRAY_Z.get(e, 1)) for e in elements}
    elif weighting == 'neutron':
        f = {e: NEUTRON_B.get(e, 1.0) for e in elements}
    elif weighting == 'unweighted':
        f = {e: 1.0 for e in elements}
    else:
        raise ValueError(f"Unknown weighting '{weighting}'.")
    f_mean = sum(concentrations[e] * f[e] for e in elements)
    f_mean_sq = f_mean ** 2
    return f, f_mean_sq


def calculate_sq(structure, q_range=(0.5, 20.0), n_q=500, weighting='xray'):
    """
    Calculate total and partial structure factors S(q) via Debye formula (Faber-Ziman).

    Parameters
    ----------
    structure : pymatgen Structure
    q_range : tuple  (q_min, q_max) in 1/Å
    n_q : int
    weighting : str  'xray' | 'neutron' | 'unweighted'

    Returns
    -------
    dict: 'q', 'S_total', 'partials', 'weighting', 'concentrations',
          'q_range', 'partial_densities_abs'
    """
    sites = structure.sites
    n_atoms = len(sites)
    volume = structure.volume
    elements = list(set(s.specie.symbol for s in sites))
    elem_sites = {e: [s for s in sites if s.specie.symbol == e] for e in elements}
    concentrations = {e: len(elem_sites[e]) / n_atoms for e in elements}
    f, f_mean_sq = _get_scattering_weights(elements, concentrations, weighting)
    q_values = np.linspace(q_range[0], q_range[1], n_q)

    symbols = [s.specie.symbol for s in sites]
    frac = np.array([s.frac_coords for s in sites])
    lattice = structure.lattice.matrix
    frac_diff = frac[:, np.newaxis, :] - frac[np.newaxis, :, :]
    frac_diff -= np.round(frac_diff)
    dist_matrix = np.linalg.norm(frac_diff @ lattice, axis=-1)

    pairs = [(e1, e2) for i, e1 in enumerate(elements) for e2 in elements[i:]]
    partial_sq = {}

    for (e1, e2) in pairs:
        idx1 = [i for i, s in enumerate(symbols) if s == e1]
        idx2 = [i for i, s in enumerate(symbols) if s == e2]
        n1, n2 = len(idx1), len(idx2)
        sub = dist_matrix[np.ix_(idx1, idx2)]

        if e1 == e2:
            r_arr = sub[~np.eye(n1, dtype=bool)]
        else:
            r_arr = sub.ravel()

        r_arr = r_arr[r_arr > 0]
        sq_partial = np.zeros(n_q)

        if len(r_arr) > 0:
            qr = np.outer(q_values, r_arr)
            debye_sum = (np.sin(qr) / qr).sum(axis=1)
            # Faber-Ziman (Wolf Eq. 2):
            norm = n1  # N_alpha for both same and cross (plain FZ, Eq. B2)
            sq_partial = 1.0 + debye_sum / norm

        partial_sq[(e1, e2)] = sq_partial
        if e1 != e2:
            partial_sq[(e2, e1)] = sq_partial

    S_total = np.zeros(n_q)
    if f_mean_sq > 0:
        for e1 in elements:
            for e2 in elements:
                w = concentrations[e1] * concentrations[e2] * f[e1] * f[e2] / f_mean_sq
                S_total += w * partial_sq.get((e1, e2), np.ones(n_q))
    else:
        S_total = np.ones(n_q)

    partial_densities_abs = {e: len(elem_sites[e]) / volume for e in elements}

    return {
        'q': q_values,
        'S_total': S_total,
        'partials': {p: partial_sq[p] for p in pairs},
        'weighting': weighting,
        'concentrations': concentrations,
        'q_range': q_range,
        'partial_densities_abs': partial_densities_abs,
    }


def calculate_gr_from_sq(sq_data, r_range=(0.5, 10.0), n_r=500,
                          use_lorch=True, density=None):
    """
    Compute D(r) and g(r) from S(q) via sine Fourier transform (Wolf et al. Eq. 8-9).

    D(r) = (2/π) ∫ q(S(q)-1) sin(qr) M(q) dq
    M(q) = Lorch modification function
    g(r) = 1 + D(r) / (4π r n)

    Parameters
    ----------
    sq_data : dict  output of calculate_sq()
    r_range : tuple
    n_r : int
    use_lorch : bool
    density : float or None  total number density (atoms/Å³)

    Returns
    -------
    dict: 'r', 'D_r', 'g_r', 'D_partials', 'g_partials', 'use_lorch', 'weighting'
    """
    q = sq_data['q']
    q_max = q[-1]
    dq = q[1] - q[0]
    r_values = np.linspace(r_range[0], r_range[1], n_r)

    if use_lorch:
        with np.errstate(divide='ignore', invalid='ignore'):
            M = (q_max / (q * np.pi)) * np.sin(q * np.pi / q_max)
        M[0] = 1.0
    else:
        M = np.ones_like(q)

    def _ft_to_Dr(S_q):
        integrand = q * (S_q - 1.0) * M
        qr = np.outer(r_values, q)
        return (2.0 / np.pi) * (np.sin(qr) @ integrand) * dq

    def _Dr_to_gr(D_r, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            gr = 1.0 + D_r / (4.0 * np.pi * r_values * n)
        gr[r_values <= 0] = 0.0
        return gr

    D_total = _ft_to_Dr(sq_data['S_total'])
    D_partials = {pair: _ft_to_Dr(S_ab) for pair, S_ab in sq_data['partials'].items()}

    g_total = None
    g_partials = None
    if density is not None:
        g_total = _Dr_to_gr(D_total, density)
        partial_densities_abs = sq_data.get('partial_densities_abs', {})
        g_partials = {}
        for pair, D_ab in D_partials.items():
            n_target = partial_densities_abs.get(pair[1], density)
            g_partials[pair] = _Dr_to_gr(D_ab, n_target)

    return {
        'r': r_values,
        'D_r': D_total,
        'g_r': g_total,
        'D_partials': D_partials,
        'g_partials': g_partials,
        'use_lorch': use_lorch,
        'weighting': sq_data['weighting'],
    }


def calculate_ensemble_sq(struct_ids, sq_data=None, energies=None,
                           use_weights=False, temperature=1800):
    """Ensemble-average S(q) (total and partials)."""
    from src.data_management_v2 import load_sq
    if sq_data is None:
        sq_data = load_sq(struct_ids, pairs='all')
    valid_ids = [sid for sid in struct_ids if sid in sq_data]
    if energies is None:
        energies = load_energies(valid_ids)
    structures = load_structures(valid_ids)
    if use_weights:
        weights = calculate_weights(valid_ids, energies, structures, temperature)
    else:
        weights = {sid: 1.0 / len(valid_ids) for sid in valid_ids}

    first = sq_data[valid_ids[0]]
    q = first['q']
    S_total_avg = np.zeros_like(q)
    partials_avg = {pair: np.zeros_like(q) for pair in first.get('partials', {})}

    for sid in valid_ids:
        w = weights.get(sid, 0.0)
        d = sq_data[sid]
        S_total_avg += w * d['S_total']
        for pair in partials_avg:
            if pair in d.get('partials', {}):
                partials_avg[pair] += w * d['partials'][pair]

    return {
        'q': q,
        'S_total': S_total_avg,
        'partials': partials_avg,
        'weighting': first['weighting'],
        'concentrations': first['concentrations'],
        'q_range': first['q_range'],
        'partial_densities_abs': first.get('partial_densities_abs', {}),
    }


def calculate_ensemble_gr_from_sq(struct_ids, gr_sq_data=None, energies=None,
                                   use_weights=False, temperature=1800):
    """Ensemble-average FT-derived D(r) and g(r)."""
    from src.data_management_v2 import load_gr_from_sq
    if gr_sq_data is None:
        gr_sq_data = load_gr_from_sq(struct_ids, pairs='all')
    valid_ids = [sid for sid in struct_ids if sid in gr_sq_data]
    if energies is None:
        energies = load_energies(valid_ids)
    structures = load_structures(valid_ids)
    if use_weights:
        weights = calculate_weights(valid_ids, energies, structures, temperature)
    else:
        weights = {sid: 1.0 / len(valid_ids) for sid in valid_ids}

    first = gr_sq_data[valid_ids[0]]
    r = first['r']
    D_avg = np.zeros_like(r)
    g_avg = np.zeros_like(r) if first['g_r'] is not None else None
    D_partials_avg = {pair: np.zeros_like(r) for pair in first.get('D_partials', {})}
    g_partials_avg = ({pair: np.zeros_like(r) for pair in first['D_partials']}
                      if first.get('g_partials') else None)

    for sid in valid_ids:
        w = weights.get(sid, 0.0)
        d = gr_sq_data[sid]
        D_avg += w * d['D_r']
        if g_avg is not None and d['g_r'] is not None:
            g_avg += w * d['g_r']
        for pair in D_partials_avg:
            if pair in d.get('D_partials', {}):
                D_partials_avg[pair] += w * d['D_partials'][pair]
        if g_partials_avg and d.get('g_partials'):
            for pair in g_partials_avg:
                if pair in d['g_partials']:
                    g_partials_avg[pair] += w * d['g_partials'][pair]

    return {
        'r': r,
        'D_r': D_avg,
        'g_r': g_avg,
        'D_partials': D_partials_avg,
        'g_partials': g_partials_avg,
        'use_lorch': first['use_lorch'],
        'weighting': first['weighting'],
    }

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import NeighborList
from src.data_management_v2 import (load_energies, load_rdfs, load_counting_functions,
                                    load_structures, calculate_weights)


# =====================================================================
# CORE RDF CALCULATION FUNCTIONS
# =====================================================================

def calculate_rdf(structure, r_range, bins, element_pairs=None, periodic=True):
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    r_min, r_max = r_range
    r_edges = np.linspace(r_min, r_max, bins + 1)
    r_values = 0.5 * (r_edges[1:] + r_edges[:-1])
    dr = r_edges[1] - r_edges[0]
    cell = atoms.get_cell()
    volume = atoms.get_volume()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)
    elements = np.unique(symbols)
    elem_count = {elem: symbols.count(elem) for elem in elements}
    if element_pairs is None:
        use_all_pairs = True
    else:
        use_all_pairs = False
        pair_table = {}
        for elem1, elem2 in element_pairs:
            pair_table[(elem1, elem2)] = True
            pair_table[(elem2, elem1)] = True
    cutoffs = [r_max/2] * n_atoms
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)
    hist = np.zeros(bins)
    if element_pairs is not None and len(element_pairs) == 1:
        elem1, elem2 = element_pairs[0]
        n_center = elem_count.get(elem1, 0)
        n_target = elem_count.get(elem2, 0)
    else:
        n_center = n_atoms
        n_target = n_atoms
    pair_count = 0
    for i in range(n_atoms):
        elem_i = symbols[i]
        if not use_all_pairs:
            if all(elem_i != pair[0] for pair in element_pairs):
                continue
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            elem_j = symbols[j]
            if not use_all_pairs:
                if (elem_i, elem_j) not in pair_table:
                    continue
            pos_i = positions[i]
            pos_j = positions[j] + np.dot(offset, cell)
            distance = np.linalg.norm(pos_j - pos_i)
            if r_min <= distance <= r_max:
                bin_index = int((distance - r_min) / (r_max - r_min) * bins)
                if 0 <= bin_index < bins:
                    hist[bin_index] += 1
                    pair_count += 1
    g_r = np.zeros_like(hist)
    for i in range(bins):
        shell_volume = 4 * np.pi * r_values[i]**2 * dr
        if use_all_pairs:
            rho = n_atoms / volume
            n_central = n_atoms
        else:
            elem1, elem2 = element_pairs[0]
            n_central = elem_count.get(elem1, 0)
            if elem1 == elem2:
                rho = elem_count.get(elem1, 0) / volume
            else:
                rho = elem_count.get(elem2, 0) / volume
        n_central = n_center
        if shell_volume > 0 and rho > 0 and n_central > 0:
            g_r[i] = hist[i] / (n_central * shell_volume * rho)
    return r_values, g_r


def calculate_partial_rdfs(structure, r_range, bins, element_pairs=None, periodic=True):
    if element_pairs is None:
        elements = np.unique([site.specie.symbol for site in structure])
        element_pairs = []
        for i, elem1 in enumerate(elements):
            for elem2 in elements[i:]:
                element_pairs.append((elem1, elem2))
    results = {}
    for pair in element_pairs:
        r_values, g_r = calculate_rdf(structure, r_range, bins, [pair], periodic)
        results[pair] = (r_values, g_r)
    return results


# =====================================================================
# ENSEMBLE AVERAGING FUNCTIONS
# =====================================================================

def calculate_ensemble_rdf(struct_ids=None, rdfs=None, energies=None, use_weights=False,
                            temperature=1800, r_range=(0, 10), bins=200):
    if rdfs is None:
        if struct_ids is None:
            raise ValueError("Must provide either struct_ids or rdfs")
        rdfs = load_rdfs(struct_ids, pairs='total')
    if energies is None:
        if struct_ids is None:
            struct_ids = list(rdfs.keys())
        energies = load_energies(struct_ids)
    rdf_struct_ids = list(rdfs.keys())
    if use_weights:
        structures = load_structures(rdf_struct_ids)
        valid_ids = [sid for sid in rdf_struct_ids if sid in energies and sid in structures]
        weights = calculate_weights(valid_ids, energies, structures, temperature)
    else:
        weights = {sid: 1.0 / len(rdf_struct_ids) for sid in rdf_struct_ids}
    ensemble_rdf = None
    r_values = None
    avg_number_density = 0.066
    for struct_id, (r, rdf) in tqdm(rdfs.items(), desc="Calculating ensemble RDF"):
        if struct_id not in weights:
            continue
        weight = weights[struct_id]
        number_density = 0.066
        density_factor = (number_density / avg_number_density) ** 2
        if r_values is None:
            r_values = r
            ensemble_rdf = np.zeros_like(rdf)
        ensemble_rdf += weight * density_factor * rdf
    return r_values, ensemble_rdf


def calculate_ensemble_partial_rdfs(struct_ids=None, rdfs=None, energies=None,
                                     densities=None, use_weights=False, temperature=1800,
                                     element_pairs=None, smoothed=True):
    if element_pairs is None:
        element_pairs = [('Si', 'Si'), ('Si', 'O'), ('O', 'O')]
    if rdfs is None:
        if struct_ids is None:
            raise ValueError("Must provide either struct_ids or rdfs")
        rdfs = load_rdfs(struct_ids, pairs='all', smoothed=smoothed)
    if energies is None:
        if struct_ids is None:
            struct_ids = list(rdfs.keys())
        energies = load_energies(struct_ids)
    if densities is None:
        if struct_ids is None:
            struct_ids = list(rdfs.keys())
        from src.data_management_v2 import load_densities
        densities = load_densities(struct_ids)
    rdf_struct_ids = list(rdfs.keys())
    if use_weights:
        structures = load_structures(rdf_struct_ids)
        valid_ids = [sid for sid in rdf_struct_ids if sid in energies and sid in structures]
        weights = calculate_weights(valid_ids, energies, structures, temperature)
    else:
        weights = {sid: 1.0 / len(rdf_struct_ids) for sid in rdf_struct_ids}
    from src.data_management_v2 import calculate_ensemble_average_density
    avg_densities = calculate_ensemble_average_density(list(rdfs.keys()), energies, temperature)
    ensemble_rdfs = {pair: None for pair in element_pairs}
    r_values = None
    for struct_id, struct_rdfs in tqdm(rdfs.items(), desc="Calculating ensemble partial RDFs"):
        if struct_id not in weights or struct_id not in densities:
            continue
        weight = weights[struct_id]
        density_data = densities[struct_id]
        if r_values is None and element_pairs[0] in struct_rdfs:
            r_values, _ = struct_rdfs[element_pairs[0]]
            for pair in element_pairs:
                if pair in struct_rdfs:
                    ensemble_rdfs[pair] = np.zeros_like(struct_rdfs[pair][1])
        for pair in element_pairs:
            if pair not in struct_rdfs:
                continue
            r, g_r = struct_rdfs[pair]
            elem1, elem2 = pair
            struct_density1 = density_data['partial_densities'].get(elem1, 0)
            avg_density1 = avg_densities['partial_densities'].get(elem1, 0)
            if elem1 == elem2:
                density_factor = struct_density1 / avg_density1 if avg_density1 > 0 else 1.0
            else:
                struct_total = density_data['total_density']
                avg_total = avg_densities['total_density']
                density_factor = struct_total / avg_total if avg_total > 0 else 1.0
            # Fix: pre-peak region must stay 0, not get shifted by density correction
            g_normalized = np.where(g_r > 0, (g_r - 1) * density_factor + 1, 0.0)
            if ensemble_rdfs[pair] is not None:
                ensemble_rdfs[pair] += weight * g_normalized
    final_ensemble_rdfs = {}
    for pair in element_pairs:
        if ensemble_rdfs[pair] is not None:
            final_ensemble_rdfs[pair] = (r_values, ensemble_rdfs[pair])
    return final_ensemble_rdfs


# =====================================================================
# COUNTING FUNCTION IMPLEMENTATIONS
# =====================================================================

def simple_counting_function(struct_id, structure_dict, element_pair, r_range=(0, 10), bins=100):
    structure = structure_dict[struct_id]["structure"]
    center_element, neighbor_element = element_pair
    sites = structure.sites
    center_sites = [site for site in sites if site.specie.symbol == center_element]
    distances = []
    r_min, r_max = r_range
    for center_site in center_sites:
        neighbors = structure.get_neighbors(center_site, r_max)
        for neighbor_info in neighbors:
            neighbor_site = neighbor_info[0]
            distance = neighbor_info[1]
            if neighbor_site.specie.symbol == neighbor_element:
                if (center_element == neighbor_element and
                        np.allclose(center_site.coords, neighbor_site.coords, atol=1e-3)):
                    continue
                if r_min <= distance <= r_max:
                    distances.append(distance)
    r_values = np.linspace(r_min, r_max, bins)
    hist, bin_edges = np.histogram(distances, bins=bins, range=r_range)
    cumulative = np.cumsum(hist)
    counting_function = cumulative / len(center_sites)
    return r_values, counting_function


def calculate_partial_counting_functions(structure_dict, struct_id, element_pairs=None,
                                          r_range=(0, 10), bins=100):
    if element_pairs is None:
        structure = structure_dict[struct_id]["structure"]
        elements = list(set([site.specie.symbol for site in structure.sites]))
        element_pairs = []
        for i, elem1 in enumerate(elements):
            for j in range(i, len(elements)):
                elem2 = elements[j]
                element_pairs.append((elem1, elem2))
    results = {}
    for pair in element_pairs:
        r_values, counting_func = simple_counting_function(struct_id, structure_dict,
                                                            pair, r_range, bins)
        results[pair] = (r_values, counting_func)
    return results


def calculate_ensemble_partial_counting_functions(struct_ids=None, counting_functions=None,
                                                   energies=None, densities=None,
                                                   use_weights=False, temperature=1800,
                                                   element_pairs=None):
    if element_pairs is None:
        element_pairs = [('Si', 'Si'), ('Si', 'O'), ('O', 'O')]
    if counting_functions is None:
        if struct_ids is None:
            raise ValueError("Must provide either struct_ids or counting_functions")
        counting_functions = load_counting_functions(struct_ids, pairs='all')
    if energies is None:
        if struct_ids is None:
            struct_ids = list(counting_functions.keys())
        energies = load_energies(struct_ids)
    if densities is None:
        if struct_ids is None:
            struct_ids = list(counting_functions.keys())
        from src.data_management_v2 import load_densities
        densities = load_densities(struct_ids)
    cf_struct_ids = list(counting_functions.keys())
    if use_weights:
        structures = load_structures(cf_struct_ids)
        weights = calculate_weights(cf_struct_ids, energies, structures, temperature)
    else:
        weights = {sid: 1.0 / len(cf_struct_ids) for sid in cf_struct_ids}
    from src.data_management_v2 import calculate_ensemble_average_density
    avg_densities = calculate_ensemble_average_density(list(counting_functions.keys()),
                                                       energies, temperature)
    ensemble_countings = {pair: None for pair in element_pairs}
    r_values = None
    for struct_id, struct_cfs in tqdm(counting_functions.items(),
                                       desc="Calculating ensemble counting functions"):
        if struct_id not in weights or struct_id not in densities:
            continue
        weight = weights[struct_id]
        density_data = densities[struct_id]
        if r_values is None and element_pairs[0] in struct_cfs:
            r_values = struct_cfs[element_pairs[0]][0]
            for pair in element_pairs:
                if pair in struct_cfs:
                    ensemble_countings[pair] = np.zeros_like(struct_cfs[pair][1])
        for pair in element_pairs:
            if pair not in struct_cfs:
                continue
            r, counting_func = struct_cfs[pair]
            elem1, elem2 = pair
            struct_density1 = density_data['partial_densities'].get(elem1, 0)
            avg_density1 = avg_densities['partial_densities'].get(elem1, 0)
            if elem1 == elem2:
                density_factor = struct_density1 / avg_density1 if avg_density1 > 0 else 1.0
            else:
                struct_total = density_data['total_density']
                avg_total = avg_densities['total_density']
                density_factor = struct_total / avg_total if avg_total > 0 else 1.0
            normalized_counting = counting_func * density_factor
            if ensemble_countings[pair] is not None:
                ensemble_countings[pair] += weight * normalized_counting
    results = {}
    for pair in element_pairs:
        if ensemble_countings[pair] is not None:
            results[pair] = (r_values, ensemble_countings[pair])
    return results


# =====================================================================
# PLOTTING FUNCTIONS
# =====================================================================

def plot_single_counting_function(struct_ids=None, counting_functions=None, energies=None,
                                   use_weights=False, temperature=1800, figsize=(8, 10),
                                   label=None, save_path=None, show_plot=True):
    print("Calculating ensemble partial counting functions...")
    ensemble_countings = calculate_ensemble_partial_counting_functions(
        struct_ids=struct_ids, counting_functions=counting_functions,
        energies=energies, use_weights=use_weights, temperature=temperature
    )
    if show_plot:
        fig = plt.figure(figsize=figsize)
        for i, ((elem1, elem2), (r, counting)) in enumerate(ensemble_countings.items(), 1):
            plt.subplot(3, 1, i)
            plt.plot(r, counting, label=label)
            plt.title(f'{elem1}-{elem2} Counting Function')
            plt.xlabel('Distance (Å)')
            plt.ylabel('N(r)')
            plt.grid(True)
            if label:
                plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    return ensemble_countings


def plot_multiple_counting_functions(*data_items, labels=None, temperature=1800,
                                      use_weights=False, figsize=(8, 10), save_path=None):
    if len(data_items) == 0:
        raise ValueError("At least one data item must be provided")
    all_cf_results = []
    all_cfs = {}
    for i, item in enumerate(data_items):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"
        if isinstance(item, (list, np.ndarray)):
            ensemble_cfs = calculate_ensemble_partial_counting_functions(
                struct_ids=item, temperature=temperature, use_weights=use_weights
            )
        elif isinstance(item, dict) and all(isinstance(k, tuple) and len(k) == 2
                                             for k in item.keys()):
            ensemble_cfs = item
        else:
            print(f"Warning: Unrecognized data type at index {i}, skipping")
            continue
        all_cf_results.append(ensemble_cfs)
        for pair, (r, cf) in ensemble_cfs.items():
            if pair not in all_cfs:
                all_cfs[pair] = []
            all_cfs[pair].append((r, cf, label))
    if not all_cfs:
        print("Warning: No valid CF data to plot")
        return all_cf_results
    plt.figure(figsize=figsize)
    for i, (pair, cf_list) in enumerate(all_cfs.items()):
        plt.subplot(3, 1, i+1)
        for r, cf, label in cf_list:
            plt.plot(r, cf, label=label, linewidth=2)
        plt.title(f'{pair[0]}-{pair[1]} Counting Function')
        plt.xlabel('Distance (Å)')
        plt.ylabel('N(r)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    return all_cf_results


def plot_single_rdf(struct_ids=None, rdfs=None, energies=None, use_weights=False,
                    temperature=1800, smoothed=True, figsize=(8, 10), label=None,
                    save_path=None, show_plot=True):
    print("Calculating ensemble partial RDFs...")
    ensemble_rdfs = calculate_ensemble_partial_rdfs(
        struct_ids=struct_ids, rdfs=rdfs, energies=energies,
        use_weights=use_weights, temperature=temperature, smoothed=smoothed
    )
    if show_plot:
        fig = plt.figure(figsize=figsize)
        title_suffix = " (KAMEL-LOBE Smoothed)" if smoothed else ""
        for i, ((elem1, elem2), (r, rdf)) in enumerate(ensemble_rdfs.items(), 1):
            plt.subplot(3, 1, i)
            plt.plot(r, rdf, label=label)
            plt.title(f'{elem1}-{elem2} Partial RDF{title_suffix}')
            plt.xlabel('Distance (Å)')
            plt.ylabel('g(r)')
            plt.grid(True)
            if label:
                plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show_plot:
            plt.show()
    return ensemble_rdfs


def plot_multiple_rdfs(*data_items, labels=None, temperature=1800, use_weights=False,
                       smoothed=True, figsize=(8, 10), save_path=None):
    if len(data_items) == 0:
        raise ValueError("At least one data item must be provided")
    all_rdf_results = []
    all_rdfs = {}
    for i, item in enumerate(data_items):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"
        if isinstance(item, (list, np.ndarray)):
            ensemble_rdfs = calculate_ensemble_partial_rdfs(
                struct_ids=item, temperature=temperature,
                use_weights=use_weights, smoothed=smoothed
            )
        elif isinstance(item, dict) and all(isinstance(k, tuple) and len(k) == 2
                                             for k in item.keys()):
            ensemble_rdfs = item
        else:
            print(f"Warning: Unrecognized data type at index {i}, skipping")
            continue
        all_rdf_results.append(ensemble_rdfs)
        for pair, (r, rdf) in ensemble_rdfs.items():
            if pair not in all_rdfs:
                all_rdfs[pair] = []
            all_rdfs[pair].append((r, rdf, label))
    if not all_rdfs:
        print("Warning: No valid RDF data to plot")
        return all_rdf_results
    title_suffix = " (KAMEL-LOBE Smoothed)" if smoothed else ""
    plt.figure(figsize=figsize)
    for i, (pair, rdf_list) in enumerate(all_rdfs.items()):
        plt.subplot(3, 1, i+1)
        for r, rdf, label in rdf_list:
            plt.plot(r, rdf, label=label, linewidth=2)
        plt.title(f'{pair[0]}-{pair[1]} Partial RDF{title_suffix}')
        plt.xlabel('Distance (Å)')
        plt.ylabel('g(r)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    return all_rdf_results


# =====================================================================
# CONVERSION FUNCTIONS BETWEEN RDF AND COUNTING FUNCTIONS
# =====================================================================

def convert_rdf_to_counting_function(ensemble_rdfs, ensemble_densities=None):
    ensemble_countings = {}
    for pair, (r_values, g_r) in ensemble_rdfs.items():
        dr = r_values[1] - r_values[0] if len(r_values) > 1 else 0.05
        if ensemble_densities is not None:
            elem1, elem2 = pair
            if elem1 == elem2:
                rho = ensemble_densities['partial_densities'].get(elem1, 0.066)
            else:
                rho1 = ensemble_densities['partial_densities'].get(elem1, 0.066)
                rho2 = ensemble_densities['partial_densities'].get(elem2, 0.066)
                rho = np.sqrt(rho1 * rho2)
        else:
            rho = 0.066
        pair_correlation = 4 * np.pi * r_values**2 * rho * (g_r - 1)
        counting_function = np.cumsum(pair_correlation) * dr
        uniform_background = (4/3) * np.pi * r_values**3 * rho
        counting_function = counting_function + uniform_background
        ensemble_countings[pair] = (r_values, counting_function)
    return ensemble_countings


def convert_counting_function_to_rdf(ensemble_countings, ensemble_densities=None):
    ensemble_rdfs = {}
    for pair, (r_values, counting_function) in ensemble_countings.items():
        dr = r_values[1] - r_values[0] if len(r_values) > 1 else 0.05
        if ensemble_densities is not None:
            elem1, elem2 = pair
            if elem1 == elem2:
                rho = ensemble_densities['partial_densities'].get(elem1, 0.066)
            else:
                rho1 = ensemble_densities['partial_densities'].get(elem1, 0.066)
                rho2 = ensemble_densities['partial_densities'].get(elem2, 0.066)
                rho = np.sqrt(rho1 * rho2)
        else:
            if len(counting_function) > 10:
                r_large = r_values[-5:]
                N_large = counting_function[-5:]
                rho_estimates = 3 * N_large / (4 * np.pi * r_large**3)
                rho = np.mean(rho_estimates[rho_estimates > 0])
            else:
                rho = 0.066
        pair_correlation = np.gradient(counting_function, dr)
        uniform_background_derivative = 4 * np.pi * r_values**2 * rho
        pair_correlation_excess = pair_correlation - uniform_background_derivative
        r_safe = np.where(r_values == 0, np.finfo(float).eps, r_values)
        g_r = 1 + pair_correlation_excess / (4 * np.pi * r_safe**2 * rho)
        g_r[:3] = 0.0
        ensemble_rdfs[pair] = (r_values, g_r)
    return ensemble_rdfs


# =====================================================================
# KAMEL-LOBE SMOOTHING (original name preserved)
# =====================================================================

def kamel_lobe_smooth(r_values, g_r, w=0.03):
    from scipy.stats import norm
    import scipy.sparse as sp
    r = np.array(r_values)
    RDF = np.array(g_r)
    Nbins = RDF.shape[0]
    delr = r[1] - r[0]
    m_KL = int(np.ceil(2*w/delr))
    if m_KL <= 1:
        print('w <= delr/2, no averaging is performed')
        return r, RDF
    T1 = np.zeros((Nbins, Nbins))
    for col in range(1, Nbins):
        value = (col * delr)**2
        T1[col, col] = value
        T1[col+1:, col] = 2 * value
    k_KL = 2 * m_KL - 1
    fractions = np.zeros((1, k_KL))
    A1_block = sp.identity(m_KL, format='csr')
    A2_block = sp.lil_matrix((m_KL, Nbins - m_KL))
    fractions[0, m_KL-1:] = (norm.cdf(((np.arange(0, m_KL) + 0.5) * delr), 0, w) -
                              norm.cdf(((np.arange(0, m_KL) - 0.5) * delr), 0, w))
    fractions[0, :m_KL-1] = np.flip(fractions[0, m_KL:2*m_KL-1])
    fractions[0, :] *= 1/np.sum(fractions)
    B_block = sp.diags(np.tile(fractions, (Nbins-2*m_KL, 1)).T,
                       np.arange(0, 2*(m_KL-1)+1),
                       shape=(Nbins-2*m_KL, Nbins))
    T2 = sp.vstack((sp.hstack((A1_block, A2_block)),
                    B_block,
                    sp.hstack((A2_block, A1_block))))
    T3 = np.zeros((Nbins, Nbins))
    constant = 1/(delr**2)
    for row in range(1, Nbins):
        T3[row, row] = constant/(row)**2
        factor = 2*constant/(row)**2
        sign = 1 - 2 * (row & 1)
        for col in range(row):
            T3[row, col] = sign * factor
            sign *= -1
    intermediate1 = T1 @ RDF
    intermediate2 = T2 @ intermediate1
    gr_convert = T3 @ intermediate2
    gr_tilde = gr_convert[:-m_KL]
    r_tilde = r[:-m_KL]
    gr_tilde = np.maximum(gr_tilde, 0.0)
    return r_tilde, gr_tilde