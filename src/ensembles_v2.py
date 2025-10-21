import random
import numpy as np
import networkx as nx
from src.data_management_v2 import load_energies, load_structures

def random_sample(struct_ids, sample_size=100):
    """
    Randomly sample a specified number of structure IDs from a list of IDs.
    
    Parameters:
    -----------
    struct_ids : list or array of int
        List of structure IDs to sample from
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs (sorted)
    """
    struct_ids = np.array(struct_ids)
    
    if sample_size > len(struct_ids):
        print(f"Warning: sample_size {sample_size} exceeds available structures {len(struct_ids)}. Using all available.")
        sample_size = len(struct_ids)
    
    sampled_indices = np.random.choice(len(struct_ids), size=sample_size, replace=False)
    sampled_ids = struct_ids[sampled_indices]
    sorted_sampled_ids = sorted(sampled_ids)
    
    return np.array(sorted_sampled_ids)

def energy_range_sample(struct_ids=None, energies=None, sample_size=100):
    """
    Sample structures evenly across the energy range.
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs to sample from (if energies not provided)
    energies : dict, optional
        Pre-loaded energy data {struct_id: energy}
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs sorted by energy
    """
    # Auto-load energies if not provided
    if energies is None:
        if struct_ids is None:
            raise ValueError("Must provide either struct_ids or energies")
        energies = load_energies(struct_ids)
    
    # Get all available structure IDs from energies
    available_ids = list(energies.keys())
    
    if sample_size > len(available_ids):
        print(f"Warning: sample_size {sample_size} exceeds available structures {len(available_ids)}. Using all available.")
        sample_size = len(available_ids)
    
    # Get energies for each structure and sort by energy
    struct_energies = [(struct_id, energies[struct_id]) for struct_id in available_ids]
    struct_energies.sort(key=lambda x: x[1])  # Sort by energy (lowest to highest)
    
    # Calculate indices for even sampling across energy range
    total_structures = len(struct_energies)
    # Create indices that are evenly spaced across the range
    indices = [int(i * (total_structures - 1) / (sample_size - 1)) for i in range(sample_size)]
    
    # Select the structures at these indices
    sampled_structs = [struct_energies[i][0] for i in indices]
    
    # Convert to numpy array and sort by ID (not energy) for consistency
    sampled_ids_array = sorted(np.array(sampled_structs))
    
    return np.array(sampled_ids_array)

def greedy_sample(struct_ids=None, distance_matrix=None, sample_size=100):
    """
    Greedy algorithm to select a diverse subset of structures that maximizes
    the minimum distance between any pair of selected structures.
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs corresponding to rows/columns in the distance matrix
    distance_matrix : numpy.ndarray
        Square matrix of pairwise distances between structures
    sample_size : int, optional
        Number of structures to select (default: 100)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs
    """
    if distance_matrix is None:
        raise ValueError("Distance matrix is required for greedy sampling")
    
    if struct_ids is None:
        # Assume struct_ids are just the indices if not provided
        struct_ids = np.arange(len(distance_matrix))
    
    struct_ids = np.array(struct_ids)
    distance_matrix = np.array(distance_matrix)
    
    # Limit sample size to available structures
    if sample_size > len(struct_ids):
        print(f"Warning: sample_size {sample_size} exceeds available structures {len(struct_ids)}. Adjusting to {len(struct_ids)}.")
        sample_size = min(sample_size, len(struct_ids))

    # Find the pair with maximum distance to start
    mask = np.triu(np.ones(distance_matrix.shape), k=1).astype(bool)
    i, j = np.unravel_index(np.argmax(distance_matrix * mask), distance_matrix.shape)
    
    # Initialize with the two most dissimilar structures
    selected_indices = [i, j]
    remaining_indices = list(range(len(struct_ids)))
    remaining_indices.remove(i)
    remaining_indices.remove(j)
    
    # Greedy selection for the rest
    for _ in range(sample_size - 2):  # -2 because we already selected 2 structures
        if not remaining_indices:
            break
            
        # For each remaining structure, find its minimum distance to any already selected structure
        min_distances = []
        for idx in remaining_indices:
            distances_to_selected = [distance_matrix[idx, sel_idx] for sel_idx in selected_indices]
            min_distances.append(min(distances_to_selected))
        
        # Choose the structure with maximum minimum distance
        max_min_idx = remaining_indices[np.argmax(min_distances)]
        selected_indices.append(max_min_idx)
        remaining_indices.remove(max_min_idx)
    
    # Convert selected indices to structure IDs
    selected_ids = struct_ids[selected_indices]
    
    # Sort by ID for consistency
    selected_ids_sorted = sorted(selected_ids)
    
    return np.array(selected_ids_sorted)

def louvain_central_sample(struct_ids=None, similarity_matrix=None, sample_size=100, resolution=1.0):
    """
    Sample structures using Louvain community detection, selecting points close 
    to the most representative point of each community.
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs corresponding to rows/columns in the similarity matrix
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
    resolution : float, optional
        Resolution parameter for Louvain community detection (default: 1.0)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs sorted by ID, or empty array if community detection fails
    """
    # Perform community detection and allocation
    G, communities, community_alloc, ids_array = _louvain_community_detection(
        similarity_matrix, struct_ids, sample_size, resolution
    )
    
    if communities is None:
        return np.array([])
    
    selected_indices = []
    
    # Process each community
    for i, community in enumerate(communities):
        alloc = community_alloc[i]
        if alloc <= 0:
            continue
        
        # Calculate centrality of each node
        centrality = _calculate_centrality(community, G)
        
        # Find the most central node
        most_central = max(centrality.keys(), key=lambda x: centrality[x])
        comm_selected = [most_central]
        
        # Find nodes most similar to the central node
        if alloc > 1:
            similarity_to_central = {}
            for node in community:
                if node != most_central:
                    edge = G.get_edge_data(node, most_central, default=None)
                    if edge:
                        similarity_to_central[node] = edge['similarity']
                    else:
                        similarity_to_central[node] = 0
            
            # Select nodes most similar to the central node
            sorted_by_similarity = sorted(similarity_to_central.keys(), 
                                        key=lambda x: similarity_to_central[x], reverse=True)
            additional = sorted_by_similarity[:alloc-1]
            comm_selected.extend(additional)
        
        selected_indices.extend(comm_selected)
    
    return _process_selections(selected_indices, ids_array)

def louvain_diverse_sample(struct_ids=None, similarity_matrix=None, sample_size=100, resolution=1.0):
    """
    Sample structures using Louvain community detection, selecting maximally diverse 
    representatives from each community.
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs corresponding to rows/columns in the similarity matrix
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
    resolution : float, optional
        Resolution parameter for Louvain community detection (default: 1.0)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs sorted by ID, or empty array if community detection fails
    """
    # Perform community detection and allocation
    G, communities, community_alloc, ids_array = _louvain_community_detection(
        similarity_matrix, struct_ids, sample_size, resolution
    )
    
    if communities is None:
        return np.array([])
    
    selected_indices = []
    
    # Process each community
    for i, community in enumerate(communities):
        alloc = community_alloc[i]
        if alloc <= 0:
            continue
        
        # Handle the simple case
        comm_list = list(community)
        if len(comm_list) <= alloc:
            selected_indices.extend(comm_list)
            continue
        
        # Start with a random point
        comm_selected = [random.choice(comm_list)]
        comm_list.remove(comm_selected[0])
        
        # Add remaining nodes greedily to maximize diversity
        while len(comm_selected) < alloc:
            best_node = None
            best_min_sim = float('inf')
            
            for node in comm_list:
                # Calculate minimum similarity to already selected nodes
                min_sim = float('inf')
                for selected in comm_selected:
                    edge = G.get_edge_data(node, selected, default=None)
                    if edge:
                        if edge['similarity'] < min_sim:
                            min_sim = edge['similarity']
                
                # Select node with minimum similarity to already selected nodes
                if min_sim < best_min_sim:
                    best_min_sim = min_sim
                    best_node = node
            
            if best_node is not None:
                comm_selected.append(best_node)
                comm_list.remove(best_node)
            else:
                # If no more valid nodes, fill with random selection
                remaining_needed = alloc - len(comm_selected)
                if comm_list and remaining_needed > 0:
                    additional = random.sample(comm_list, min(remaining_needed, len(comm_list)))
                    comm_selected.extend(additional)
                break
        
        selected_indices.extend(comm_selected)
    
    return _process_selections(selected_indices, ids_array)

def louvain_cell_tower_sample(struct_ids=None, similarity_matrix=None, sample_size=100, 
                             resolution=1.0, coverage_radius=0.5):
    """
    Sample structures using Louvain community detection with a minimal covering set approach,
    ensuring every point in each community is within a coverage radius of at least one selected point.
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs corresponding to rows/columns in the similarity matrix
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
    resolution : float, optional
        Resolution parameter for Louvain community detection (default: 1.0)
    coverage_radius : float, optional
        Similarity threshold used for coverage (default: 0.5)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs sorted by ID, or empty array if community detection fails
    """
    # Perform community detection and allocation
    G, communities, community_alloc, ids_array = _louvain_community_detection(
        similarity_matrix, struct_ids, sample_size, resolution
    )
    
    if communities is None:
        return np.array([])
    
    selected_indices = []
    
    # Process each community
    for i, community in enumerate(communities):
        alloc = community_alloc[i]
        if alloc <= 0:
            continue
        
        # Handle the simple case
        comm_list = list(community)
        if len(comm_list) <= alloc:
            selected_indices.extend(comm_list)
            continue
        
        # Create a copy of the community for tracking coverage
        all_nodes = list(community)
        remaining_nodes = list(community)  # Nodes that still need to be considered
        covered_nodes = set()  # Track which nodes are already covered
        comm_selected = []  # Selected nodes
        
        # First phase: Select points until all are covered or we've selected alloc points
        while len(comm_selected) < alloc and remaining_nodes and len(covered_nodes) < len(all_nodes):
            best_node = None
            best_coverage = -1
            
            # For each candidate node, count how many uncovered nodes it would cover
            for node in remaining_nodes:
                # Find nodes that would be covered by this node
                newly_covered = set()
                for other in all_nodes:
                    if other in covered_nodes:
                        continue  # Skip already covered nodes
                    
                    if other == node:
                        newly_covered.add(other)  # Node covers itself
                        continue
                    
                    edge = G.get_edge_data(node, other, default=None)
                    if edge and edge['similarity'] >= coverage_radius:
                        newly_covered.add(other)
                
                # Check if this node provides better coverage
                if len(newly_covered) > best_coverage:
                    best_coverage = len(newly_covered)
                    best_node = node
            
            if best_node is not None:
                comm_selected.append(best_node)
                remaining_nodes.remove(best_node)
                
                # Update covered nodes
                for other in all_nodes:
                    if other == best_node:
                        covered_nodes.add(other)
                        continue
                    
                    edge = G.get_edge_data(best_node, other, default=None)
                    if edge and edge['similarity'] >= coverage_radius:
                        covered_nodes.add(other)
            else:
                # If no more nodes can provide coverage, break this phase
                break
        
        # Second phase: If we still need more points after full coverage
        if len(comm_selected) < alloc:
            # Redefine remaining_nodes to exclude already selected
            remaining_nodes = [node for node in all_nodes if node not in comm_selected]
            
            # Continue selecting based on best additional coverage
            while len(comm_selected) < alloc and remaining_nodes:
                best_node = None
                best_score = -1
                
                for node in remaining_nodes:
                    # Calculate a coverage score based on how well this node covers others
                    # that are not already well-covered by existing selections
                    
                    coverage_score = 0
                    for other in all_nodes:
                        if other == node or other in comm_selected:
                            continue
                        
                        # Check how well other is already covered
                        best_current_sim = 0
                        for selected in comm_selected:
                            edge = G.get_edge_data(selected, other, default=None)
                            if edge:
                                best_current_sim = max(best_current_sim, edge['similarity'])
                        
                        # Check how well node would cover other
                        edge = G.get_edge_data(node, other, default=None)
                        if edge:
                            # Improvement in coverage
                            improvement = max(0, edge['similarity'] - best_current_sim)
                            coverage_score += improvement
                    
                    if coverage_score > best_score:
                        best_score = coverage_score
                        best_node = node
                
                if best_node is not None:
                    comm_selected.append(best_node)
                    remaining_nodes.remove(best_node)
                else:
                    # Fall back to random selection if no clear winner
                    additional = random.sample(remaining_nodes, min(alloc - len(comm_selected), len(remaining_nodes)))
                    comm_selected.extend(additional)
                    break
        
        selected_indices.extend(comm_selected)
    
    return _process_selections(selected_indices, ids_array)

def louvain_max_representation_sample(struct_ids=None, similarity_matrix=None, sample_size=100, resolution=1.0):
    """
    Sample structures using Louvain community detection, starting with the central point of each
    community, then iteratively adding points that maximize representation of poorly covered areas.
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs corresponding to rows/columns in the similarity matrix
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
    resolution : float, optional
        Resolution parameter for Louvain community detection (default: 1.0)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs sorted by ID, or empty array if community detection fails
    """
    # Perform community detection and allocation
    G, communities, community_alloc, ids_array = _louvain_community_detection(
        similarity_matrix, struct_ids, sample_size, resolution
    )
    
    if communities is None:
        return np.array([])
    
    selected_indices = []
    
    # Process each community
    for i, community in enumerate(communities):
        alloc = community_alloc[i]
        if alloc <= 0:
            continue
        
        # Calculate centrality within community
        centrality = _calculate_centrality(community, G)
        
        # Start with the most central node
        most_central = max(centrality.keys(), key=lambda x: centrality[x])
        comm_selected = [most_central]
        
        # Create a list of remaining candidates
        candidates = list(community)
        candidates.remove(most_central)
        
        # Calculate initial representation scores
        # For each node, how well is it represented by our selected nodes?
        representation = {}
        for node in community:
            # Initially, only the central node provides representation
            edge = G.get_edge_data(node, most_central, default=None)
            if edge:
                representation[node] = edge['similarity']
            else:
                representation[node] = 0
        
        # Iteratively add nodes that maximize representation
        while len(comm_selected) < alloc and candidates:
            best_node = None
            best_improvement = -1
            
            for node in candidates:
                # Calculate how much this node would improve representation
                improvement = 0
                for other in community:
                    if other == node:
                        continue
                    
                    # How well is 'other' currently represented?
                    current_rep = representation.get(other, 0)
                    
                    # How well would 'node' represent 'other'?
                    edge = G.get_edge_data(node, other, default=None)
                    if edge:
                        new_rep = max(current_rep, edge['similarity'])
                        improvement += (new_rep - current_rep)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_node = node
            
            if best_node is not None:
                comm_selected.append(best_node)
                candidates.remove(best_node)
                
                # Update representation scores
                for other in community:
                    edge = G.get_edge_data(best_node, other, default=None)
                    if edge:
                        representation[other] = max(representation.get(other, 0), 
                                                  edge['similarity'])
            else:
                # If no improvement possible, break
                break
        
        selected_indices.extend(comm_selected)
    
    return _process_selections(selected_indices, ids_array)

def louvain_random_sample(struct_ids=None, similarity_matrix=None, sample_size=100, resolution=1.0):
    """
    Sample structures using Louvain community detection, selecting random
    representatives from each community.
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs corresponding to rows/columns in the similarity matrix
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
    resolution : float, optional
        Resolution parameter for Louvain community detection (default: 1.0)
        
    Returns:
    --------
    tuple: (numpy.ndarray, list)
        Array of sampled structure IDs sorted by ID, list of communities
    """
    # Perform community detection and allocation
    G, communities, community_alloc, ids_array = _louvain_community_detection(
        similarity_matrix, struct_ids, sample_size, resolution
    )
    
    if communities is None:
        return np.array([]), []
    
    selected_indices = []
    
    # Process each community
    print(f"Found {len(communities)} communities")
    for i in range(len(communities)):
        print(f"Community {i}: {len(communities[i])} structures")
        
    for i, community in enumerate(communities):
        alloc = community_alloc[i]
        if alloc <= 0:
            continue
        
        # Simple random selection
        comm_list = list(community)
        if len(comm_list) <= alloc:
            comm_selected = comm_list
        else:
            comm_selected = random.sample(comm_list, alloc)
        
        selected_indices.extend(comm_selected)
    
    return _process_selections(selected_indices, ids_array), communities

##############################
###### HELPER FUNCTIONS ######
##############################

def _louvain_community_detection(similarity_matrix, struct_ids=None, sample_size=100, resolution=1.0):
    """
    Helper function that performs Louvain community detection and calculates allocation.
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    struct_ids : list of int, optional
        Structure IDs corresponding to rows/columns in the similarity matrix
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
    resolution : float, optional
        Resolution parameter for Louvain community detection (default: 1.0)
        
    Returns:
    --------
    tuple
        (graph, communities, community_alloc, ids_array)
    """
    if similarity_matrix is None:
        raise ValueError("Similarity matrix is required for Louvain community detection")
    
    # Ensure ids is a numpy array
    if struct_ids is None:
        ids_array = np.arange(len(similarity_matrix))
    else:
        ids_array = np.array(struct_ids)
    
    # Limit sample size to available structures
    sample_size = min(sample_size, len(ids_array))
    
    # Build graph from similarity matrix
    G = nx.Graph()
    G.add_nodes_from(range(len(ids_array)))
    
    # Add edges with weights based on similarities
    sim_matrix = np.array(similarity_matrix)
    if sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError("Similarity matrix must be square")
    
    for i in range(sim_matrix.shape[0]):
        for j in range(i+1, sim_matrix.shape[1]):
            similarity = sim_matrix[i, j]
            # Invert similarity for weight (lower similarity = higher weight)
            weight = 1.0 / (similarity + 1e-6)  # Avoid division by zero
            G.add_edge(i, j, weight=weight, similarity=similarity)
    
    # Find communities using Louvain algorithm
    try:
        communities = list(nx.community.louvain_communities(G, weight='weight', resolution=resolution))
    except Exception as e:
        print(f"Warning: Could not detect communities: {e}. Returning empty array.")
        return None, None, None, ids_array
    
    # Calculate sampling allocation per community (proportional to community size)
    total_structures = len(ids_array)
    community_alloc = {}
    
    # Calculate the proportional allocation for each community
    for i, comm in enumerate(communities):
        # Allocation is proportional to community size
        alloc = int(round(len(comm) / total_structures * sample_size))
        community_alloc[i] = alloc
    
    # Ensure we get exactly sample_size samples by adjusting allocations
    total_allocated = sum(community_alloc.values())
    adjustment = sample_size - total_allocated
    
    if adjustment != 0:
        # Sort communities by size, largest first
        sorted_comms = sorted(range(len(communities)), 
                             key=lambda i: len(communities[i]), reverse=True)
        
        if adjustment > 0:
            # If we need more samples, add them to largest communities first
            for i in sorted_comms:
                if adjustment <= 0:
                    break
                community_alloc[i] += 1
                adjustment -= 1
        else:
            # If we need fewer samples, remove from smallest communities first
            for i in reversed(sorted_comms):
                if adjustment >= 0:
                    break
                if community_alloc[i] > 0:
                    community_alloc[i] -= 1
                    adjustment += 1
    
    return G, communities, community_alloc, ids_array

def _calculate_centrality(community, G):
    """
    Helper function to calculate the average similarity of each node to all others in community.
    
    Parameters:
    -----------
    community : list
        List of node indices in the community
    G : networkx.Graph
        Graph with edges representing similarities
        
    Returns:
    --------
    dict
        Dictionary mapping node indices to centrality scores
    """
    centrality = {}
    for node in community:
        similarities = []
        for other in community:
            if node != other:
                edge = G.get_edge_data(node, other, default=None)
                if edge:
                    similarities.append(edge['similarity'])
        if similarities:
            centrality[node] = sum(similarities) / len(similarities)
        else:
            centrality[node] = 0
    return centrality

def _process_selections(selected_indices, ids_array):
    """
    Process selected indices and return sorted IDs.
    
    Parameters:
    -----------
    selected_indices : list
        List of selected node indices
    ids_array : numpy.ndarray
        Array of structure IDs
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs sorted by ID
    """
    # Convert node indices back to structure IDs
    selected_ids = ids_array[selected_indices]
    
    # Sort by ID for consistency
    selected_ids_sorted = sorted(selected_ids)
    
    return np.array(selected_ids_sorted)

# =====================================================================
# ERROR CALCULATION FUNCTIONS
# =====================================================================

def calculate_counting_errors(new, ref, method='rmse'):
    """
    Calculate errors between two RDF dictionaries (same format as your functions return)
    
    Parameters:
    -----------
    new, ref : dict
        Dictionaries with element pairs as keys and (r_values, g_r) as values
    method : str
        'rmse' or 'kl_divergence'
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and error values as values
    """
    errors = {}
    
    for pair in new.keys():
        if pair in ref:
            r1, rdf1 = new[pair]
            r2, rdf2 = ref[pair]
            
            # Ensure same r-range (interpolate if needed)
            if not np.allclose(r1, r2):
                from scipy.interpolate import interp1d
                f = interp1d(r2, rdf2, kind='linear', bounds_error=False, fill_value=0)
                rdf2 = f(r1)
            
            error = calculate_counting_error(rdf1, rdf2, r1, method=method)
            errors[pair] = error
    
    return errors

def calculate_counting_error(rdf_calc, rdf_ref, r_values, method='rmse'):
    """Calculate error between calculated and reference RDF"""
    if method == 'rmse':
        return np.sqrt(np.mean((rdf_calc - rdf_ref)**2))
    elif method == 'kl_divergence':
        # Fix interpolation artifacts that can create negative RDF values
        rdf_calc_fixed = np.maximum(rdf_calc, 0.0)
        rdf_ref_fixed = np.maximum(rdf_ref, 0.0)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        rdf_calc_fixed = rdf_calc_fixed + epsilon
        rdf_ref_fixed = rdf_ref_fixed + epsilon
        
        # Normalize to probability distributions
        rdf_calc_norm = rdf_calc_fixed / np.sum(rdf_calc_fixed)
        rdf_ref_norm = rdf_ref_fixed / np.sum(rdf_ref_fixed)
        
        # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
        kl_div = np.sum(rdf_ref_norm * np.log(rdf_ref_norm / rdf_calc_norm))
        
        return kl_div

# =====================================================================
# UTILITY AND ANALYSIS FUNCTIONS
# =====================================================================

def compare_sampling_methods(struct_ids, similarity_matrix=None, sample_size=100, random_seed=42):
    """
    Compare different sampling methods on the same structure set.
    
    Parameters:
    -----------
    struct_ids : list of int
        Structure IDs to sample from
    similarity_matrix : numpy.ndarray, optional
        Similarity matrix for structure-based sampling methods
    sample_size : int, optional
        Number of structures to sample with each method
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Dictionary with method names as keys and sampled IDs as values
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Load energies once for all methods
    energies = load_energies(struct_ids)
    
    results = {}
    
    # Basic sampling methods
    results['random'] = random_sample(struct_ids, sample_size)
    results['energy_range'] = energy_range_sample(energies=energies, sample_size=sample_size)
    
    # Structure-based methods (if similarity matrix provided)
    if similarity_matrix is not None:
        print("Running structure-based sampling methods...")
        
        results['greedy'] = greedy_sample(
            struct_ids=struct_ids, distance_matrix=similarity_matrix, sample_size=sample_size
        )
        
        results['louvain_central'] = louvain_central_sample(
            struct_ids=struct_ids, similarity_matrix=similarity_matrix, sample_size=sample_size
        )
        
        results['louvain_diverse'] = louvain_diverse_sample(
            struct_ids=struct_ids, similarity_matrix=similarity_matrix, sample_size=sample_size
        )
        
        results['louvain_cell_tower'] = louvain_cell_tower_sample(
            struct_ids=struct_ids, similarity_matrix=similarity_matrix, sample_size=sample_size
        )
        
        results['louvain_max_representation'] = louvain_max_representation_sample(
            struct_ids=struct_ids, similarity_matrix=similarity_matrix, sample_size=sample_size
        )
        
        results['louvain_random'], _ = louvain_random_sample(
            struct_ids=struct_ids, similarity_matrix=similarity_matrix, sample_size=sample_size
        )
    
    return results

def analyze_sample_properties(sampled_ids, energies=None, structures=None):
    """
    Analyze the properties of a sampled set of structures.
    
    Parameters:
    -----------
    sampled_ids : list or array of int
        Sampled structure IDs
    energies : dict, optional
        Energy data for analysis
    structures : dict, optional
        Structure data for analysis
        
    Returns:
    --------
    dict : Analysis results
    """
    if energies is None:
        energies = load_energies(sampled_ids)
    
    if structures is None:
        structures = load_structures(sampled_ids)
    
    # Get energies for sampled structures
    sample_energies = [energies[struct_id] for struct_id in sampled_ids if struct_id in energies]
    
    if not sample_energies:
        return {"error": "No energy data available for sampled structures"}
    
    sample_energies = np.array(sample_energies)
    
    # Get volumes for sampled structures
    sample_volumes = []
    for struct_id in sampled_ids:
        if struct_id in structures:
            sample_volumes.append(structures[struct_id].volume)
    
    analysis = {
        'n_structures': len(sampled_ids),
        'n_with_energies': len(sample_energies),
        'n_with_structures': len(sample_volumes),
        'energy_stats': {
            'min': np.min(sample_energies),
            'max': np.max(sample_energies),
            'mean': np.mean(sample_energies),
            'std': np.std(sample_energies),
            'median': np.median(sample_energies),
            'range': np.max(sample_energies) - np.min(sample_energies)
        }
    }
    
    if sample_volumes:
        sample_volumes = np.array(sample_volumes)
        analysis['volume_stats'] = {
            'min': np.min(sample_volumes),
            'max': np.max(sample_volumes),
            'mean': np.mean(sample_volumes),
            'std': np.std(sample_volumes),
            'median': np.median(sample_volumes),
            'range': np.max(sample_volumes) - np.min(sample_volumes)
        }
    
    return analysis

def plot_sample_comparison(comparison_results, energies=None, figsize=(15, 10)):
    """
    Plot comparison of different sampling methods.
    
    Parameters:
    -----------
    comparison_results : dict
        Results from compare_sampling_methods()
    energies : dict, optional
        Energy data for plotting
    figsize : tuple, optional
        Figure size
    """
    import matplotlib.pyplot as plt
    
    if energies is None:
        # Load energies for all structures in comparison
        all_ids = set()
        for sampled_ids in comparison_results.values():
            all_ids.update(sampled_ids)
        energies = load_energies(list(all_ids))
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot 1: Energy distributions
    ax = axes[0]
    for method, sampled_ids in comparison_results.items():
        sample_energies = [energies[sid] for sid in sampled_ids if sid in energies]
        if sample_energies:
            ax.hist(sample_energies, alpha=0.6, label=method, bins=20)
    ax.set_xlabel('Energy (Ry)')
    ax.set_ylabel('Count')
    ax.set_title('Energy Distributions by Sampling Method')
    ax.legend()
    
    # Plot 2: Energy statistics
    ax = axes[1]
    methods = list(comparison_results.keys())
    means = []
    stds = []
    for method in methods:
        sample_energies = [energies[sid] for sid in comparison_results[method] if sid in energies]
        if sample_energies:
            means.append(np.mean(sample_energies))
            stds.append(np.std(sample_energies))
        else:
            means.append(0)
            stds.append(0)
    
    x_pos = np.arange(len(methods))
    ax.bar(x_pos, means, yerr=stds, capsize=5)
    ax.set_xlabel('Sampling Method')
    ax.set_ylabel('Mean Energy (Ry)')
    ax.set_title('Mean Energy by Sampling Method')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45)
    
    # Plot 3: Sample size comparison
    ax = axes[2]
    sample_sizes = [len(sampled_ids) for sampled_ids in comparison_results.values()]
    ax.bar(methods, sample_sizes)
    ax.set_xlabel('Sampling Method')
    ax.set_ylabel('Number of Structures')
    ax.set_title('Sample Sizes by Method')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Plot 4: Energy range coverage
    ax = axes[3]
    for method, sampled_ids in comparison_results.items():
        sample_energies = [energies[sid] for sid in sampled_ids if sid in energies]
        if sample_energies:
            ax.scatter([method] * len(sample_energies), sample_energies, alpha=0.6, s=10)
    ax.set_xlabel('Sampling Method')
    ax.set_ylabel('Energy (Ry)')
    ax.set_title('Energy Range Coverage')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Plot 5: Energy spread (min-max range)
    ax = axes[4]
    energy_ranges = []
    for method in methods:
        sample_energies = [energies[sid] for sid in comparison_results[method] if sid in energies]
        if sample_energies:
            energy_ranges.append(np.max(sample_energies) - np.min(sample_energies))
        else:
            energy_ranges.append(0)
    
    ax.bar(methods, energy_ranges)
    ax.set_xlabel('Sampling Method')
    ax.set_ylabel('Energy Range (Ry)')
    ax.set_title('Energy Range Coverage by Method')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Plot 6: Cumulative energy distribution
    ax = axes[5]
    all_energies = list(energies.values())
    all_energies.sort()
    ax.plot(all_energies, np.arange(len(all_energies)) / len(all_energies), 
            'k-', linewidth=2, label='Full dataset')
    
    for method, sampled_ids in comparison_results.items():
        sample_energies = [energies[sid] for sid in sampled_ids if sid in energies]
        if sample_energies:
            sample_energies.sort()
            ax.plot(sample_energies, np.arange(len(sample_energies)) / len(sample_energies), 
                   alpha=0.7, label=method)
    
    ax.set_xlabel('Energy (Ry)')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Energy Distributions')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_ensemble_diversity(sampled_ids, similarity_matrix=None, metric='average_distance'):
    """
    Evaluate the diversity of a sampled ensemble.
    
    Parameters:
    -----------
    sampled_ids : list or array of int
        Sampled structure IDs
    similarity_matrix : numpy.ndarray, optional
        Similarity matrix for the full dataset
    metric : str, optional
        Diversity metric to use ('average_distance', 'min_distance', 'max_distance')
        
    Returns:
    --------
    float : Diversity score
    """
    if similarity_matrix is None:
        print("Warning: No similarity matrix provided, cannot calculate structural diversity")
        return None
    
    # Get indices of sampled structures
    sampled_indices = []
    for struct_id in sampled_ids:
        if struct_id < len(similarity_matrix):
            sampled_indices.append(struct_id)
    
    if len(sampled_indices) < 2:
        return 0.0
    
    # Extract submatrix for sampled structures
    sub_matrix = similarity_matrix[np.ix_(sampled_indices, sampled_indices)]
    
    # Calculate diversity based on chosen metric
    if metric == 'average_distance':
        # Average pairwise distance (excluding diagonal)
        mask = np.triu(np.ones_like(sub_matrix, dtype=bool), k=1)
        distances = sub_matrix[mask]
        return np.mean(distances)
    
    elif metric == 'min_distance':
        # Minimum pairwise distance (excluding diagonal)
        mask = np.triu(np.ones_like(sub_matrix, dtype=bool), k=1)
        distances = sub_matrix[mask]
        return np.min(distances)
    
    elif metric == 'max_distance':
        # Maximum pairwise distance (excluding diagonal)
        mask = np.triu(np.ones_like(sub_matrix, dtype=bool), k=1)
        distances = sub_matrix[mask]
        return np.max(distances)
    
    else:
        raise ValueError(f"Unknown diversity metric: {metric}")

def bootstrap_ensemble_properties(struct_ids, energies=None, n_bootstrap=100, sample_size=100, 
                                 random_seed=42):
    """
    Bootstrap analysis of ensemble properties to estimate uncertainty.
    
    Parameters:
    -----------
    struct_ids : list of int
        Full set of structure IDs to bootstrap from
    energies : dict, optional
        Energy data
    n_bootstrap : int, optional
        Number of bootstrap samples
    sample_size : int, optional
        Size of each bootstrap sample
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Bootstrap statistics
    """
    np.random.seed(random_seed)
    
    if energies is None:
        energies = load_energies(struct_ids)
    
    bootstrap_stats = {
        'mean_energies': [],
        'std_energies': [],
        'min_energies': [],
        'max_energies': []
    }
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_sample = random_sample(struct_ids, sample_size)
        
        # Calculate properties
        sample_energies = [energies[sid] for sid in bootstrap_sample if sid in energies]
        
        if sample_energies:
            bootstrap_stats['mean_energies'].append(np.mean(sample_energies))
            bootstrap_stats['std_energies'].append(np.std(sample_energies))
            bootstrap_stats['min_energies'].append(np.min(sample_energies))
            bootstrap_stats['max_energies'].append(np.max(sample_energies))
    
    # Calculate bootstrap statistics
    results = {}
    for key, values in bootstrap_stats.items():
        if values:
            results[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'percentile_2.5': np.percentile(values, 2.5),
                'percentile_97.5': np.percentile(values, 97.5)
            }
    
    return results

# =====================================================================
# ADVANCED ENSEMBLE FUNCTIONS
# =====================================================================

def adaptive_sampling(struct_ids, initial_sample_size=50, max_iterations=10, 
                     convergence_threshold=0.01, property_function=None):
    """
    Adaptive sampling that iteratively adds structures until convergence.
    
    Parameters:
    -----------
    struct_ids : list of int
        Full set of structure IDs to sample from
    initial_sample_size : int, optional
        Initial sample size
    max_iterations : int, optional
        Maximum number of iterations
    convergence_threshold : float, optional
        Convergence threshold for property changes
    property_function : callable, optional
        Function to calculate property of interest for convergence checking
        
    Returns:
    --------
    dict : Results including final sample and convergence history
    """
    if property_function is None:
        # Default: use mean energy as convergence property
        def property_function(sample_ids):
            energies = load_energies(sample_ids)
            sample_energies = [energies[sid] for sid in sample_ids if sid in energies]
            return np.mean(sample_energies) if sample_energies else 0
    
    # Start with initial random sample
    current_sample = random_sample(struct_ids, initial_sample_size)
    remaining_ids = [sid for sid in struct_ids if sid not in current_sample]
    
    convergence_history = []
    current_property = property_function(current_sample)
    convergence_history.append(current_property)
    
    for iteration in range(max_iterations):
        if not remaining_ids:
            break
        
        # Add more structures (e.g., 10% of current sample size)
        add_size = max(1, len(current_sample) // 10)
        add_size = min(add_size, len(remaining_ids))
        
        new_structures = random_sample(remaining_ids, add_size)
        current_sample = np.concatenate([current_sample, new_structures])
        
        # Remove added structures from remaining
        remaining_ids = [sid for sid in remaining_ids if sid not in new_structures]
        
        # Check convergence
        new_property = property_function(current_sample)
        property_change = abs(new_property - current_property) / abs(current_property)
        
        convergence_history.append(new_property)
        
        if property_change < convergence_threshold:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        current_property = new_property
    
    return {
        'final_sample': current_sample,
        'convergence_history': convergence_history,
        'n_iterations': len(convergence_history) - 1,
        'final_sample_size': len(current_sample)
    }

def multi_objective_sampling(struct_ids, objectives=None, weights=None, sample_size=100):
    """
    Multi-objective sampling that balances multiple criteria.
    
    Parameters:
    -----------
    struct_ids : list of int
        Structure IDs to sample from
    objectives : list of callable, optional
        List of objective functions to optimize
    weights : list of float, optional
        Weights for each objective
    sample_size : int, optional
        Number of structures to sample
        
    Returns:
    --------
    numpy.ndarray : Sampled structure IDs
    """
    if objectives is None:
        # Default objectives: energy diversity and structural diversity
        energies = load_energies(struct_ids)
        
        def energy_diversity_obj(sample_ids):
            sample_energies = [energies[sid] for sid in sample_ids if sid in energies]
            return np.std(sample_energies) if len(sample_energies) > 1 else 0
        
        def energy_range_obj(sample_ids):
            sample_energies = [energies[sid] for sid in sample_ids if sid in energies]
            return np.max(sample_energies) - np.min(sample_energies) if len(sample_energies) > 1 else 0
        
        objectives = [energy_diversity_obj, energy_range_obj]
    
    if weights is None:
        weights = [1.0] * len(objectives)
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    # Simple greedy multi-objective optimization
    selected_ids = []
    remaining_ids = list(struct_ids)
    
    # Start with random structure
    first_id = random.choice(remaining_ids)
    selected_ids.append(first_id)
    remaining_ids.remove(first_id)
    
    # Greedily add structures that optimize the weighted sum of objectives
    for _ in range(sample_size - 1):
        if not remaining_ids:
            break
        
        best_score = -float('inf')
        best_id = None
        
        for candidate_id in remaining_ids:
            test_sample = selected_ids + [candidate_id]
            
            # Calculate weighted objective score
            score = 0
            for obj_func, weight in zip(objectives, weights):
                try:
                    obj_value = obj_func(test_sample)
                    score += weight * obj_value
                except:
                    score += 0  # Handle errors gracefully
            
            if score > best_score:
                best_score = score
                best_id = candidate_id
        
        if best_id is not None:
            selected_ids.append(best_id)
            remaining_ids.remove(best_id)
    
    return np.array(sorted(selected_ids))

