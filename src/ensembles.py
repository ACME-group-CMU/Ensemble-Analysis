import random
import numpy as np
import networkx as nx
import src.data_management as dm
import src.structure_manager as sm

def random_sample(ids, sample_size=100):
    """
    Randomly sample a specified number of structure IDs from a structure dictionary.
    
    Parameters:
    -----------
    structure_dict : list/array
        list of ID
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs
    """

    sampled_indices = np.random.choice(len(ids), size=sample_size, replace=False)
    sampled_ids = ids[sampled_indices]
    sorted_sampled_ids = sorted(sampled_ids, key=int)
    
    return sorted_sampled_ids

def energy_range_sample(structure_dict, sample_size=100):
    """
    Sample structures evenly across the energy range.
    
    Parameters:
    -----------
    structure_dict : dict or OrderedDict
        Dictionary containing structure IDs as keys, with each entry containing
        a 'structure' and 'Energy (Ry)' key
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs sorted by energy
    """
    # Get all available structure IDs
    available_ids = list(structure_dict.keys())
    
    # Get energies for each structure
    struct_energies = [(id_str, structure_dict[id_str]["Energy (Ry)"]) for id_str in available_ids]
    
    # Sort by energy (lowest to highest)
    struct_energies.sort(key=lambda x: x[1])
    
    # Calculate indices for even sampling
    total_structures = len(struct_energies)
    # Create indices that are evenly spaced across the range
    indices = [int(i * (total_structures - 1) / (sample_size - 1)) for i in range(sample_size)]
    
    # Select the structures at these indices
    sampled_structs = [struct_energies[i][0] for i in indices]
    
    # Convert to numpy array (similar to what random_structure_sample does)
    sampled_ids_array = sorted(np.array(sampled_structs), key=int)
    
    return sampled_ids_array

def greedy_sample(distance_matrix, ids, sample_size=100):
    """
    Greedy algorithm to select a diverse subset of structures that maximizes
    the minimum distance between any pair of selected structures.
    
    Parameters:
    -----------
    distance_matrix : numpy.ndarray
        Square matrix of pairwise distances between structures
    ids : list or numpy.ndarray
        List of structure IDs corresponding to rows/columns in the distance matrix
    sample_size : int, optional
        Number of structures to select (default: 100)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs
    """
    import numpy as np
    
    # Ensure inputs are numpy arrays
    distance_matrix = np.array(distance_matrix)
    ids = np.array(ids)
    
    # Limit sample size to available structures
    if sample_size > len(ids):
        print(f"Warning: sample_size {sample_size} exceeds available structures {len(ids)}. Adjusting to {len(ids)}.")
    sample_size = min(sample_size, len(ids))

    mask = np.triu(np.ones(distance_matrix.shape), k=1).astype(bool)
    
    # Get the indices of the maximum value
    i, j = np.unravel_index(np.argmax(distance_matrix * mask), distance_matrix.shape)
    
    # Initialize with the two most dissimilar structures
    selected_indices = [i, j]
    remaining_indices = list(range(len(ids)))
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
    selected_ids = ids[selected_indices]
    
    # Sort by ID for consistency
    selected_ids_sorted = sorted(selected_ids, key=int)
    
    return np.array(selected_ids_sorted)

def louvain_central_sample(similarity_matrix, ids, sample_size=100, resolution=1.0):
    """
    Sample structures using Louvain community detection, selecting points close 
    to the most representative point of each community.
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    ids : list or numpy.ndarray
        List of structure IDs corresponding to rows/columns in the similarity matrix
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
        similarity_matrix, ids, sample_size, resolution
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

def louvain_diverse_sample(similarity_matrix, ids, sample_size=100, resolution=1.0):
    """
    Sample structures using Louvain community detection, selecting maximally diverse 
    representatives from each community.
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    ids : list or numpy.ndarray
        List of structure IDs corresponding to rows/columns in the similarity matrix
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
        similarity_matrix, ids, sample_size, resolution
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

def louvain_cell_tower_sample(similarity_matrix, ids, sample_size=100, resolution=1.0, coverage_radius=0.5):
    """
    Sample structures using Louvain community detection with a minimal covering set approach,
    ensuring every point in each community is within a coverage radius of at least one selected point.
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    ids : list or numpy.ndarray
        List of structure IDs corresponding to rows/columns in the similarity matrix
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
        similarity_matrix, ids, sample_size, resolution
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

def louvain_max_representation_sample(similarity_matrix, ids, sample_size=100, resolution=1.0):
    """
    Sample structures using Louvain community detection, starting with the central point of each
    community, then iteratively adding points that maximize representation of poorly covered areas.
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    ids : list or numpy.ndarray
        List of structure IDs corresponding to rows/columns in the similarity matrix
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
        similarity_matrix, ids, sample_size, resolution
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

def louvain_random_sample(similarity_matrix, ids, sample_size=100, resolution=1.0):
    """
    Sample structures using Louvain community detection, selecting random
    representatives from each community.
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    ids : list or numpy.ndarray
        List of structure IDs corresponding to rows/columns in the similarity matrix
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
        similarity_matrix, ids, sample_size, resolution
    )
    
    if communities is None:
        return np.array([])
    
    selected_indices = []
    
    # Process each community
    print(len(communities))
    for i in range(len(communities)):
        print(len(communities[i]))
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

def _louvain_community_detection(similarity_matrix, ids, sample_size=100, resolution=1.0):
    """
    Helper function that performs Louvain community detection and calculates allocation.
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Square matrix of pairwise similarities between structures
    ids : list or numpy.ndarray
        List of structure IDs corresponding to rows/columns in the similarity matrix
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
    resolution : float, optional
        Resolution parameter for Louvain community detection (default: 1.0)
        
    Returns:
    --------
    tuple
        (graph, communities, community_alloc, ids_array)
    """
    # Ensure ids is a numpy array
    ids_array = np.array(ids)
    
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
    selected_ids_sorted = sorted(selected_ids, key=int)
    
    return np.array(selected_ids_sorted)


