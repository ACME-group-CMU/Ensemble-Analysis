"""
Ring statistics analysis using King's criterion for glass structures.
Calculates ring size distributions and connectivity matrices.

Key features:
- PBC-correct via explicit supercell tiling: the structure is expanded into
  a supercell before graph construction so that rings can close through
  periodic images without algebraic image tracking. Statistics are normalized
  back to the base cell.
- Base cell atoms identified by fractional coordinates in [0, 1/na) x [0, 1/nb) x [0, 1/nc)
  — robust to pymatgen's internal atom ordering after make_supercell.
- 2-rings: Edge-sharing tetrahedra (2 Si sharing 2+ O atoms)
- 3+ rings: King's criterion (shortest path between neighbor pairs)
- Connectivity matrix: Co-occurrence of ring sizes at nodes
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm


# =====================================================================
# BOND CUTOFFS
# =====================================================================

def get_bond_cutoffs(structure, method='default', custom_cutoffs=None):
    """
    Determine bond cutoffs for each element pair.

    Parameters:
    -----------
    structure : pymatgen Structure
    method : str
        'default' or 'custom'
    custom_cutoffs : dict, optional

    Returns:
    --------
    dict : (element1, element2) -> cutoff_distance
    """
    if method == 'custom' and custom_cutoffs is not None:
        return custom_cutoffs

    return {
        ('Si', 'O'): 2.0,
        ('O', 'Si'): 2.0,
        ('Si', 'Si'): 3.3,
        ('O', 'O'): 3.0,
    }


# =====================================================================
# SUPERCELL GRAPH CONSTRUCTION
# =====================================================================

def structure_to_graph(structure, cutoffs, network_formers=['Si'], supercell=(3, 3, 3)):
    """
    Convert structure to a NetworkX MultiGraph by tiling into a supercell
    and building the Si-Si connectivity graph without PBC.

    Base cell atoms are identified by fractional coordinates falling in
    [0, 1/na) x [0, 1/nb) x [0, 1/nc) — the origin image. This is
    robust to pymatgen's internal atom ordering after make_supercell.

    Node attributes:
      'is_base_cell' : True for atoms in the origin image of the supercell

    Parameters:
    -----------
    structure : pymatgen Structure
    cutoffs : dict
    network_formers : list
    supercell : tuple of int
        (na, nb, nc) tiling. (3,3,3) is sufficient for most SiO2 cells.

    Returns:
    --------
    (networkx.MultiGraph, int)
        (graph, n_base_si) where n_base_si is the Si count in the base cell.
    """
    na, nb, nc = supercell
    network_former_set = set(network_formers)
    si_o_cutoff = cutoffs.get(('Si', 'O'), cutoffs.get(('O', 'Si'), 2.0))

    # Count base cell Si before tiling
    n_base_si = sum(1 for site in structure if site.species_string in network_former_set)

    # Build supercell
    sc = structure.copy()
    sc.make_supercell(list(supercell))

    # Identify Si and O in supercell
    si_indices_sc = [i for i, site in enumerate(sc)
                     if site.species_string in network_former_set]
    si_set_sc = set(si_indices_sc)

    def _is_base_cell(frac_coords):
        """True if fractional coords fall in the origin image."""
        return (0 <= frac_coords[0] < 1/na and
                0 <= frac_coords[1] < 1/nb and
                0 <= frac_coords[2] < 1/nc)

    G = nx.MultiGraph()

    for sc_idx in si_indices_sc:
        fc = sc[sc_idx].frac_coords
        G.add_node(sc_idx, is_base_cell=_is_base_cell(fc))

    # Build Si-O-Si edges (no PBC needed — supercell large enough)
    edges_added = set()

    for sc_idx_i in si_indices_sc:
        site_i = sc[sc_idx_i]
        o_neighbors = sc.get_neighbors(site_i, si_o_cutoff)

        for o_nbr in o_neighbors:
            o_sc_idx = o_nbr.index
            if sc[o_sc_idx].species_string in network_former_set:
                continue  # not oxygen

            o_site = sc[o_sc_idx]
            si_neighbors_of_o = sc.get_neighbors(o_site, si_o_cutoff)

            for si_nbr in si_neighbors_of_o:
                sc_idx_j = si_nbr.index
                if sc_idx_j not in si_set_sc:
                    continue
                if sc_idx_j == sc_idx_i:
                    continue

                edge_id = (min(sc_idx_i, sc_idx_j), max(sc_idx_i, sc_idx_j), o_sc_idx)
                if edge_id not in edges_added:
                    G.add_edge(sc_idx_i, sc_idx_j, bridge_atom=o_sc_idx)
                    edges_added.add(edge_id)

    return G, n_base_si


# =====================================================================
# RING DETECTION
# =====================================================================

def find_rings(graph, base_si_count, max_ring_size=20):
    """
    Find rings using King's criterion, searching only from base-cell nodes
    to avoid triple-counting from equivalent images.

    2-rings: pairs of Si sharing 2+ distinct O bridges.
    3+ rings: King's criterion shortest path.

    Parameters:
    -----------
    graph : networkx.MultiGraph
    base_si_count : int
        Number of Si in the original unit cell (for normalization)
    max_ring_size : int

    Returns:
    --------
    dict
    """
    rings_per_base_node = defaultdict(list)
    all_rings_set = set()

    base_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_base_cell')]

    # === 2-RINGS ===
    bridge_pairs = defaultdict(set)
    for u, v, data in graph.edges(data=True):
        key = (min(u, v), max(u, v))
        bridge_pairs[key].add(data['bridge_atom'])

    two_rings_found = set()
    for (u, v), bridges in bridge_pairs.items():
        if len(bridges) >= 2:
            u_base = graph.nodes[u]['is_base_cell']
            v_base = graph.nodes[v]['is_base_cell']
            if u_base or v_base:
                ring_key = frozenset([u, v])
                if ring_key not in two_rings_found:
                    two_rings_found.add(ring_key)
                    all_rings_set.add((2, ring_key))
                    if u_base:
                        rings_per_base_node[u].append(2)
                    if v_base:
                        rings_per_base_node[v].append(2)

    # === KING'S CRITERION FOR 3+ RINGS ===
    for node in base_nodes:
        neighbors = list(set(graph.neighbors(node)))
        if len(neighbors) < 2:
            continue

        subgraph = graph.copy()
        subgraph.remove_node(node)

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                n1, n2 = neighbors[i], neighbors[j]
                try:
                    path = nx.shortest_path(subgraph, source=n1, target=n2)
                    ring_size = len(path) + 1

                    if ring_size <= max_ring_size:
                        ring_key = frozenset([node] + path)
                        if len(ring_key) == ring_size:
                            rings_per_base_node[node].append(ring_size)
                            all_rings_set.add((ring_size, ring_key))

                except nx.NetworkXNoPath:
                    pass

    ring_size_counts = defaultdict(int)
    for sizes in rings_per_base_node.values():
        for size in sizes:
            ring_size_counts[size] += 1

    return {
        'rings_per_node': dict(rings_per_base_node),
        'all_rings': list(all_rings_set),
        'ring_size_counts': dict(ring_size_counts)
    }


# =====================================================================
# STATISTICS CALCULATION
# =====================================================================

def calculate_ring_statistics(ring_data, num_nodes):
    """
    Calculate RC, RN, and connectivity matrix normalized to base cell.

    Parameters:
    -----------
    ring_data : dict
        From find_rings()
    num_nodes : int
        Number of Si in the base cell

    Returns:
    --------
    dict : RC, RN, connectivity_matrix, ring_sizes
    """
    rings_per_node = ring_data['rings_per_node']
    all_rings = ring_data['all_rings']

    all_sizes = set()
    for sizes in rings_per_node.values():
        all_sizes.update(sizes)
    ring_sizes = sorted(all_sizes)

    if not ring_sizes:
        return {'RC': {}, 'RN': {}, 'connectivity_matrix': np.array([[]]), 'ring_sizes': []}

    unique_by_size = defaultdict(int)
    for ring_size, _ in all_rings:
        unique_by_size[ring_size] += 1
    RC = {size: count / num_nodes for size, count in unique_by_size.items()}

    total_count = defaultdict(int)
    for sizes in rings_per_node.values():
        for size in sizes:
            total_count[size] += 1
    RN = {size: count / num_nodes for size, count in total_count.items()}

    n_sizes = len(ring_sizes)
    connectivity_matrix = np.zeros((n_sizes, n_sizes))
    for sizes in rings_per_node.values():
        sizes_set = set(sizes)
        for i, si in enumerate(ring_sizes):
            for j, sj in enumerate(ring_sizes):
                if si in sizes_set and sj in sizes_set:
                    connectivity_matrix[i, j] += 1 / num_nodes

    return {
        'RC': RC,
        'RN': RN,
        'connectivity_matrix': connectivity_matrix,
        'ring_sizes': ring_sizes
    }


# =====================================================================
# MAIN INTERFACE
# =====================================================================

def calculate_ring_statistics_for_structure(structure, cutoffs=None, max_ring_size=20,
                                            supercell=(3, 3, 3)):
    """
    Calculate complete ring statistics for a structure.

    Parameters:
    -----------
    structure : pymatgen Structure
    cutoffs : dict, optional
    max_ring_size : int
    supercell : tuple
        Tiling for PBC handling. Increase for very elongated cells.

    Returns:
    --------
    dict
    """
    if cutoffs is None:
        cutoffs = get_bond_cutoffs(structure)

    graph, n_base_si = structure_to_graph(structure, cutoffs,
                                          network_formers=['Si'],
                                          supercell=supercell)
    ring_data = find_rings(graph, n_base_si, max_ring_size=max_ring_size)
    stats = calculate_ring_statistics(ring_data, n_base_si)

    return {
        'num_atoms': len(structure),
        'num_si': n_base_si,
        'num_rings_total': len(ring_data['all_rings']),
        'RC': stats['RC'],
        'RN': stats['RN'],
        'connectivity_matrix': stats['connectivity_matrix'],
        'ring_sizes': stats['ring_sizes'],
        'raw_ring_data': ring_data
    }


def calculate_ensemble_ring_statistics(structures, cutoffs=None, max_ring_size=20,
                                       supercell=(3, 3, 3), show_progress=True):
    """
    Calculate ring statistics for an ensemble of structures.

    Parameters:
    -----------
    structures : dict
        {structure_id: pymatgen Structure}
    cutoffs : dict, optional
    max_ring_size : int
    supercell : tuple
    show_progress : bool

    Returns:
    --------
    dict : {structure_id: ring_statistics}
    """
    ensemble_stats = {}
    iterator = tqdm(structures.items()) if show_progress else structures.items()

    for struct_id, structure in iterator:
        if show_progress:
            iterator.set_description(f"Structure {struct_id}")
        stats = calculate_ring_statistics_for_structure(
            structure, cutoffs=cutoffs, max_ring_size=max_ring_size, supercell=supercell
        )
        ensemble_stats[struct_id] = stats

    return ensemble_stats


# =====================================================================
# UTILITIES
# =====================================================================

def summarize_ring_statistics(ring_stats):
    """Print summary of ring statistics."""
    print(f"Total atoms: {ring_stats['num_atoms']}")
    print(f"Si atoms (base cell): {ring_stats['num_si']}")
    print(f"Total rings found: {ring_stats['num_rings_total']}")
    print(f"\nRing size distribution:")
    print(f"  Size  |  RC (rings/cell)  |  RN (rings/node)")
    print("-" * 50)
    for size in ring_stats['ring_sizes']:
        rc = ring_stats['RC'].get(size, 0)
        rn = ring_stats['RN'].get(size, 0)
        print(f"  {size:3d}   |  {rc:15.3f}  |  {rn:15.3f}")


def get_dominant_ring_sizes(ensemble_stats, top_n=5):
    """Find most common ring sizes in ensemble."""
    rc_totals = defaultdict(float)
    n = len(ensemble_stats)
    for stats in ensemble_stats.values():
        for size, rc in stats['RC'].items():
            rc_totals[size] += rc
    averages = sorted([(s, v/n) for s, v in rc_totals.items()], key=lambda x: -x[1])
    return averages[:top_n]


def average_connectivity_matrices(ensemble_stats):
    """Average connectivity matrices across ensemble."""
    all_sizes = set()
    for stats in ensemble_stats.values():
        all_sizes.update(stats['ring_sizes'])
    ring_sizes = sorted(all_sizes)
    size_to_idx = {s: i for i, s in enumerate(ring_sizes)}
    sum_matrix = np.zeros((len(ring_sizes), len(ring_sizes)))

    for stats in ensemble_stats.values():
        for i, si in enumerate(stats['ring_sizes']):
            for j, sj in enumerate(stats['ring_sizes']):
                sum_matrix[size_to_idx[si], size_to_idx[sj]] += stats['connectivity_matrix'][i, j]

    return {
        'average_matrix': sum_matrix / len(ensemble_stats),
        'ring_sizes': ring_sizes,
        'num_structures': len(ensemble_stats)
    }