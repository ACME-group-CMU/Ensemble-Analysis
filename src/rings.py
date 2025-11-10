"""
Ring statistics analysis using King's criterion for glass structures.
Calculates ring size distributions and connectivity matrices.

Key features:
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
        The structure to analyze
    method : str
        'default' or 'custom'
    custom_cutoffs : dict, optional
        Custom cutoff distances
        
    Returns:
    --------
    dict : (element1, element2) -> cutoff_distance
    """
    if method == 'custom' and custom_cutoffs is not None:
        return custom_cutoffs
    
    # Default cutoffs for SiO2 glass (Angstroms)
    return {
        ('Si', 'O'): 2.0,
        ('O', 'Si'): 2.0,
        ('Si', 'Si'): 3.3,
        ('O', 'O'): 3.0,
    }


# =====================================================================
# GRAPH CONSTRUCTION
# =====================================================================

def structure_to_graph(structure, cutoffs, network_formers=['Si']):
    """
    Convert structure to NetworkX MultiGraph for ring analysis.
    
    For SiO2: Only Si atoms are nodes, edges represent O bridges.
    MultiGraph allows multiple edges for edge-sharing tetrahedra.
    
    Parameters:
    -----------
    structure : pymatgen Structure
    cutoffs : dict
        Bond cutoff distances
    network_formers : list
        Elements to include as nodes (default ['Si'])
        
    Returns:
    --------
    networkx.MultiGraph
    """
    G = nx.MultiGraph()
    
    # Add network former nodes
    for i, site in enumerate(structure):
        if site.species_string in network_formers:
            G.add_node(i, element=site.species_string)
    
    network_former_indices = [i for i, site in enumerate(structure) 
                             if site.species_string in network_formers]
    
    # Build edges through bridging atoms (O)
    edges_added = set()
    
    for i in network_former_indices:
        site_i = structure[i]
        element_i = site_i.species_string
        
        # Get Si-O cutoff
        si_o_cutoff = cutoffs.get((element_i, 'O'), cutoffs.get(('O', element_i), 2.0))
        
        # Find O neighbors of this Si
        o_neighbors = structure.get_neighbors(site_i, si_o_cutoff)
        
        for o_neighbor in o_neighbors:
            o_idx = o_neighbor.index
            o_site = structure[o_idx]
            
            # Skip if not a bridging atom
            if o_site.species_string in network_formers:
                continue
            
            # Find other Si bonded to this O
            other_si_neighbors = structure.get_neighbors(o_site, si_o_cutoff)
            
            for other_si in other_si_neighbors:
                j = other_si.index
                
                if j in network_former_indices and i < j:
                    # Unique edge identifier
                    edge_id = (min(i, j), max(i, j), o_idx)
                    
                    if edge_id not in edges_added:
                        distance = o_neighbor.nn_distance + other_si.nn_distance
                        G.add_edge(i, j, distance=distance, bridge_atom=o_idx)
                        edges_added.add(edge_id)
    
    return G


# =====================================================================
# RING DETECTION
# =====================================================================

def find_rings(graph, max_ring_size=20):
    """
    Find rings using King's criterion with special 2-ring detection.
    
    2-rings: Pairs of nodes with 2+ edges (edge-sharing tetrahedra)
    3+ rings: King's criterion - shortest path between neighbor pairs
    
    Parameters:
    -----------
    graph : networkx.MultiGraph
    max_ring_size : int
        
    Returns:
    --------
    dict : {
        'rings_per_node': {node_id: [ring_sizes]},
        'all_rings': [(ring_size, ring_nodes_tuple)],
        'ring_size_counts': {ring_size: count}
    }
    """
    rings_per_node = defaultdict(list)
    all_rings_set = set()
    
    # === DETECT 2-RINGS (Edge-sharing tetrahedra) ===
    # Two Si sharing 2+ O atoms = multiple edges
    two_rings_found = set()
    
    for u, v in graph.edges():
        if u < v:
            num_edges = graph.number_of_edges(u, v)
            
            if num_edges >= 2:
                ring_nodes = tuple(sorted([u, v]))
                
                if ring_nodes not in two_rings_found:
                    two_rings_found.add(ring_nodes)
                    all_rings_set.add((2, ring_nodes))
                    
                    # Add to both nodes
                    rings_per_node[u].append(2)
                    rings_per_node[v].append(2)
    
    # === KING'S CRITERION FOR 3+ RINGS ===
    for node in graph.nodes():
        neighbors = list(set(graph.neighbors(node)))
        
        if len(neighbors) < 2:
            continue
        
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                n1, n2 = neighbors[i], neighbors[j]
                
                # Remove central node and find path
                subgraph = graph.copy()
                subgraph.remove_node(node)
                
                try:
                    path = nx.shortest_path(subgraph, source=n1, target=n2)
                    ring_size = len(path) + 1
                    
                    if ring_size <= max_ring_size:
                        rings_per_node[node].append(ring_size)
                        ring_nodes = tuple(sorted([node] + path))
                        all_rings_set.add((ring_size, ring_nodes))
                        
                except nx.NetworkXNoPath:
                    pass
    
    # Count ring sizes
    ring_size_counts = defaultdict(int)
    for node, sizes in rings_per_node.items():
        for size in sizes:
            ring_size_counts[size] += 1
    
    return {
        'rings_per_node': dict(rings_per_node),
        'all_rings': list(all_rings_set),
        'ring_size_counts': dict(ring_size_counts)
    }


# =====================================================================
# STATISTICS CALCULATION
# =====================================================================

def calculate_ring_statistics(ring_data, num_nodes):
    """
    Calculate RC, RN, and connectivity matrix.
    
    Parameters:
    -----------
    ring_data : dict
        From find_rings()
    num_nodes : int
        Total nodes in structure
        
    Returns:
    --------
    dict : RC, RN, connectivity_matrix, ring_sizes
    """
    rings_per_node = ring_data['rings_per_node']
    all_rings = ring_data['all_rings']
    
    # Get unique ring sizes
    all_sizes = set()
    for node, sizes in rings_per_node.items():
        all_sizes.update(sizes)
    ring_sizes = sorted(all_sizes)
    
    if len(ring_sizes) == 0:
        return {
            'RC': {},
            'RN': {},
            'connectivity_matrix': np.array([[]]),
            'ring_sizes': []
        }
    
    # RC: Unique rings per cell
    RC = defaultdict(float)
    unique_rings_by_size = defaultdict(int)
    for ring_size, ring_nodes in all_rings:
        unique_rings_by_size[ring_size] += 1
    
    for size, count in unique_rings_by_size.items():
        RC[size] = count / num_nodes
    
    # RN: Rings per node
    RN = defaultdict(float)
    total_ring_count = defaultdict(int)
    for node, sizes in rings_per_node.items():
        for size in sizes:
            total_ring_count[size] += 1
    
    for size, count in total_ring_count.items():
        RN[size] = count / num_nodes
    
    # Connectivity matrix
    n_sizes = len(ring_sizes)
    connectivity_matrix = np.zeros((n_sizes, n_sizes))
    
    for node, sizes in rings_per_node.items():
        sizes_set = set(sizes)
        
        for i, size_i in enumerate(ring_sizes):
            for j, size_j in enumerate(ring_sizes):
                if size_i in sizes_set and size_j in sizes_set:
                    connectivity_matrix[i, j] += 1 / num_nodes
    
    return {
        'RC': dict(RC),
        'RN': dict(RN),
        'connectivity_matrix': connectivity_matrix,
        'ring_sizes': ring_sizes
    }


# =====================================================================
# MAIN INTERFACE
# =====================================================================

def calculate_ring_statistics_for_structure(structure, cutoffs=None, max_ring_size=20):
    """
    Calculate complete ring statistics for a structure.
    
    Parameters:
    -----------
    structure : pymatgen Structure
    cutoffs : dict, optional
    max_ring_size : int
        
    Returns:
    --------
    dict : Complete ring statistics
    """
    if cutoffs is None:
        cutoffs = get_bond_cutoffs(structure)
    
    graph = structure_to_graph(structure, cutoffs, network_formers=['Si'])
    ring_data = find_rings(graph, max_ring_size=max_ring_size)
    
    num_si = len([s for s in structure if s.species_string == 'Si'])
    stats = calculate_ring_statistics(ring_data, num_si)
    
    return {
        'num_atoms': len(structure),
        'num_si': num_si,
        'num_rings_total': len(ring_data['all_rings']),
        'RC': stats['RC'],
        'RN': stats['RN'],
        'connectivity_matrix': stats['connectivity_matrix'],
        'ring_sizes': stats['ring_sizes'],
        'raw_ring_data': ring_data
    }


def calculate_ensemble_ring_statistics(structures, cutoffs=None, max_ring_size=20, 
                                       show_progress=True):
    """
    Calculate ring statistics for ensemble.
    
    Parameters:
    -----------
    structures : dict
        {structure_id: pymatgen Structure}
    cutoffs : dict, optional
    max_ring_size : int
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
            structure, cutoffs=cutoffs, max_ring_size=max_ring_size
        )
        ensemble_stats[struct_id] = stats
    
    return ensemble_stats


# =====================================================================
# UTILITIES
# =====================================================================

def summarize_ring_statistics(ring_stats):
    """Print summary of ring statistics."""
    print(f"Total atoms: {ring_stats['num_atoms']}")
    print(f"Si atoms: {ring_stats['num_si']}")
    print(f"Total rings: {ring_stats['num_rings_total']}")
    print(f"\nRing size distribution:")
    print(f"  Size  |  RC (rings/cell)  |  RN (rings/node)")
    print("-" * 50)
    
    for size in ring_stats['ring_sizes']:
        rc = ring_stats['RC'].get(size, 0)
        rn = ring_stats['RN'].get(size, 0)
        print(f"  {size:3d}   |  {rc:15.3f}  |  {rn:15.3f}")
    
    print(f"\nConnectivity matrix: {ring_stats['connectivity_matrix'].shape}")


def get_dominant_ring_sizes(ensemble_stats, top_n=5):
    """Find most common ring sizes in ensemble."""
    rc_totals = defaultdict(float)
    n = len(ensemble_stats)
    
    for stats in ensemble_stats.values():
        for size, rc in stats['RC'].items():
            rc_totals[size] += rc
    
    averages = [(size, rc / n) for size, rc in rc_totals.items()]
    averages.sort(key=lambda x: x[1], reverse=True)
    
    return averages[:top_n]


def average_connectivity_matrices(ensemble_stats):
    """Average connectivity matrices across ensemble."""
    # Get all ring sizes
    all_sizes = set()
    for stats in ensemble_stats.values():
        all_sizes.update(stats['ring_sizes'])
    
    ring_sizes = sorted(all_sizes)
    n_sizes = len(ring_sizes)
    size_to_idx = {size: idx for idx, size in enumerate(ring_sizes)}
    
    # Sum matrices
    sum_matrix = np.zeros((n_sizes, n_sizes))
    
    for stats in ensemble_stats.values():
        struct_sizes = stats['ring_sizes']
        struct_matrix = stats['connectivity_matrix']
        
        for i, size_i in enumerate(struct_sizes):
            for j, size_j in enumerate(struct_sizes):
                gi = size_to_idx[size_i]
                gj = size_to_idx[size_j]
                sum_matrix[gi, gj] += struct_matrix[i, j]
    
    # Average
    n_structures = len(ensemble_stats)
    avg_matrix = sum_matrix / n_structures
    
    return {
        'average_matrix': avg_matrix,
        'ring_sizes': ring_sizes,
        'num_structures': n_structures
    }