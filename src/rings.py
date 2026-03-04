"""
Ring statistics analysis using King's criterion for glass structures.
Calculates ring size distributions and connectivity matrices.

Key features:
- Image-tracking BFS: nodes are (base_index, image) tuples, so the graph is
  effectively infinite and supercell size has no effect on results.
- Focal node removal only removes (focal, (0,0,0)) — not its periodic images —
  so legitimate paths through other images are never blocked.
- BFS is bounded by max_ring_size (depth limit) and image_radius (spatial limit).
- 2-rings: Edge-sharing tetrahedra (2 Si sharing 2+ O bridges).
- 3+ rings: King's criterion (shortest path between neighbor pairs in image graph).
- Rings are deduplicated by canonical frozenset of (base_index, image) tuples.
- Base cell is centered at image (0,0,0); search extends to ±image_radius.
"""

import numpy as np
import networkx as nx
from collections import defaultdict, deque
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
# PERIODIC BOND TABLE
# =====================================================================

def build_bond_table(structure, cutoffs, network_formers=['Si']):
    """
    Build a table of Si-O-Si connections with their periodic image offsets.

    For each Si atom i, finds all Si atoms j connected through an O bridge,
    along with the image offset delta such that the connection goes from
    Si_i at image (0,0,0) to Si_j at image delta.

    Parameters:
    -----------
    structure : pymatgen Structure
    cutoffs : dict
    network_formers : list

    Returns:
    --------
    tuple : (bond_table, si_indices)
        bond_table : {si_base_index: [(si_j, delta_image, o_idx), ...]}
        si_indices : list of Si atom indices in base cell
    """
    network_former_set = set(network_formers)
    si_o_cutoff = cutoffs.get(('Si', 'O'), cutoffs.get(('O', 'Si'), 2.0))

    si_indices = [int(i) for i, s in enumerate(structure)
                  if s.species_string in network_former_set]
    si_set = set(si_indices)

    bond_table = defaultdict(list)
    edges_added = set()

    for i in si_indices:
        site_i = structure[i]
        o_neighbors = structure.get_neighbors(site_i, si_o_cutoff)

        for o_nbr in o_neighbors:
            o_idx = int(o_nbr.index)
            if structure[o_idx].species_string in network_former_set:
                continue

            # Image of O relative to Si_i at (0,0,0)
            o_image = tuple(int(x) for x in np.round(o_nbr.image).astype(int))

            o_site = structure[o_idx]
            si_of_o = structure.get_neighbors(o_site, si_o_cutoff)

            for si_nbr in si_of_o:
                j = int(si_nbr.index)
                if j not in si_set or j == i:
                    continue

                # Image of Si_j relative to Si_i
                j_image = tuple(int(x) for x in
                                (np.array(o_image) +
                                 np.round(si_nbr.image).astype(int)))

                # Canonical edge key — always (smaller, larger, ...)
                if i < j:
                    edge_key = (i, j, o_idx, o_image, j_image)
                else:
                    neg = tuple(-x for x in j_image)
                    edge_key = (j, i, o_idx, neg, tuple(-x for x in j_image))
                    continue  # only process each bond once (from lower index)

                if edge_key not in edges_added:
                    edges_added.add(edge_key)
                    bond_table[i].append((j, j_image, o_idx))

    # Build reverse directions in a second pass
    full_bond_table = defaultdict(list)
    for i, connections in bond_table.items():
        for (j, j_image, o_idx) in connections:
            full_bond_table[i].append((j, j_image, o_idx))
            neg_j_image = tuple(-x for x in j_image)
            full_bond_table[j].append((i, neg_j_image, o_idx))

    return dict(full_bond_table), si_indices


# =====================================================================
# IMAGE-TRACKING BFS
# =====================================================================

def _bfs_shortest_path(start_node, end_node, bond_table, focal_base,
                       max_depth, image_radius):
    """
    BFS from start_node = (base_idx, image) to end_node = (base_idx, image),
    excluding the focal node (focal_base, (0,0,0)).

    Returns the path as list of (base_idx, image) tuples, or None if not found.
    """
    queue = deque()
    queue.append((start_node, [start_node]))
    visited = {start_node}

    while queue:
        current, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        cur_base, cur_img = current

        for (nbr_base, delta_img, _) in bond_table.get(cur_base, []):
            nbr_img = tuple(cur_img[k] + delta_img[k] for k in range(3))

            if any(abs(nbr_img[k]) > image_radius for k in range(3)):
                continue

            nbr_node = (nbr_base, nbr_img)

            # Skip focal node at (0,0,0) only
            if nbr_base == focal_base and nbr_img == (0, 0, 0):
                continue

            if nbr_node in visited:
                continue

            new_path = path + [nbr_node]

            # Exact match on (base, image)
            if nbr_node == end_node:
                return new_path

            visited.add(nbr_node)
            queue.append((nbr_node, new_path))

    return None


# =====================================================================
# RING DETECTION
# =====================================================================

def canonicalize_ring(ring_nodes):
    """Shift all images so the lexicographically smallest image is (0,0,0)."""
    nodes = list(ring_nodes)
    # Find the minimum image offset
    min_img = min(img for _, img in nodes)
    # Shift all images by -min_img
    shifted = frozenset((b, tuple(i - m for i, m in zip(img, min_img)))
                        for b, img in nodes)
    return shifted

def find_rings(structure, cutoffs, network_formers=['Si'],
               max_ring_size=20, image_radius=3):
    """
    Find rings using King's criterion with image-tracking BFS.

    King's criterion is applied per pair of O bridges from each focal Si atom.
    Rings are canonicalized by shifting images so the minimum image is (0,0,0),
    so that translated copies of the same ring are deduplicated correctly.

    Parameters:
    -----------
    structure : pymatgen Structure
    cutoffs : dict
    network_formers : list
    max_ring_size : int
    image_radius : int

    Returns:
    --------
    dict
    """
    bond_table, si_indices = build_bond_table(structure, cutoffs, network_formers)

    rings_per_node = defaultdict(list)
    all_rings_set = set()
    seen_ring_keys = set()

    def canonicalize_ring(ring_nodes):
        """Shift all images so the lexicographically smallest image is (0,0,0)."""
        nodes = list(ring_nodes)
        min_img = min(img for _, img in nodes)
        return frozenset((b, tuple(i - m for i, m in zip(img, min_img)))
                         for b, img in nodes)

    for focal in si_indices:
        arms = bond_table.get(focal, [])

        if len(arms) < 2:
            continue

        seen_pairs = set()

        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                n1_base, n1_img, o1 = arms[i]
                n2_base, n2_img, o2 = arms[j]

                n1 = (n1_base, n1_img)
                n2 = (n2_base, n2_img)

                pair_key = (i, j)

                # 2-ring: both arms go to the same (base, image) neighbor
                if n1 == n2:
                    raw_key = frozenset([(focal, (0, 0, 0)), n1])
                    canon_key = canonicalize_ring(raw_key)
                    rings_per_node[focal].append(2)
                    if canon_key not in seen_ring_keys:
                        seen_ring_keys.add(canon_key)
                        all_rings_set.add((2, canon_key))
                    seen_pairs.add(pair_key)
                    continue

                path = _bfs_shortest_path(
                    start_node=n1,
                    end_node=n2,
                    bond_table=bond_table,
                    focal_base=focal,
                    max_depth=max_ring_size - 1,
                    image_radius=image_radius
                )

                if path is None:
                    continue

                ring_size = len(path) + 1

                if ring_size > max_ring_size:
                    continue

                raw_key = frozenset([(focal, (0, 0, 0))] + path)

                if len(raw_key) != ring_size:
                    continue

                canon_key = canonicalize_ring(raw_key)

                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    rings_per_node[focal].append(ring_size)

                if canon_key not in seen_ring_keys:
                    seen_ring_keys.add(canon_key)
                    all_rings_set.add((ring_size, canon_key))

    ring_size_counts = defaultdict(int)
    for sizes in rings_per_node.values():
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
        Number of Si atoms in base cell.

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
        return {'RC': {}, 'RN': {}, 'connectivity_matrix': np.array([[]]),
                'ring_sizes': []}

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

def calculate_ring_statistics_for_structure(structure, cutoffs=None,
                                             max_ring_size=20, image_radius=3):
    """
    Calculate complete ring statistics for a structure.

    Parameters:
    -----------
    structure : pymatgen Structure
    cutoffs : dict, optional
    max_ring_size : int
    image_radius : int
        Search radius in periodic images (default 3).

    Returns:
    --------
    dict
    """
    if cutoffs is None:
        cutoffs = get_bond_cutoffs(structure)

    ring_data = find_rings(structure, cutoffs,
                           network_formers=['Si'],
                           max_ring_size=max_ring_size,
                           image_radius=image_radius)

    num_si = sum(1 for s in structure if s.species_string == 'Si')
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


def calculate_ensemble_ring_statistics(structures, cutoffs=None,
                                        max_ring_size=20, image_radius=3,
                                        show_progress=True):
    """
    Calculate ring statistics for an ensemble of structures.

    Parameters:
    -----------
    structures : dict
        {structure_id: pymatgen Structure}
    cutoffs : dict, optional
    max_ring_size : int
    image_radius : int
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
            structure, cutoffs=cutoffs,
            max_ring_size=max_ring_size,
            image_radius=image_radius
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
                sum_matrix[size_to_idx[si], size_to_idx[sj]] += \
                    stats['connectivity_matrix'][i, j]

    return {
        'average_matrix': sum_matrix / len(ensemble_stats),
        'ring_sizes': ring_sizes,
        'num_structures': len(ensemble_stats)
    }