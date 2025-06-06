import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import NeighborList
import src.data_management as dm

# =====================================================================
# RDF Calculation and Analysis
# =====================================================================

def calculate_rdf(structure, r_range, bins, element_pairs=None, periodic=True):
    """
    Calculate radial distribution function with proper normalization.
    
    Input:
    structure : pymatgen.core.structure.Structure object
    r_range : tuple of floats - (r_min, r_max) in Angstroms
    bins : int - number of bins
    element_pairs : list of tuples, optional e.g., [('Si', 'O')])
    periodic : bool
        
    Output:
    r_values : numpy.ndarray
    g_r : numpy.ndarray
    """
    # Convert pymatgen structure to ASE atoms for using the NeighborList
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    
    # Set up bins for the histogram
    r_min, r_max = r_range
    r_edges = np.linspace(r_min, r_max, bins + 1)
    r_values = 0.5 * (r_edges[1:] + r_edges[:-1])
    dr = r_edges[1] - r_edges[0]  # Bin width
    
    # Get atomic positions and cell information
    cell = atoms.get_cell()
    volume = atoms.get_volume()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)
    
    # Create a count of atoms by element type
    elements = np.unique(symbols)
    elem_count = {elem: symbols.count(elem) for elem in elements}
    
    # Process element pairs
    if element_pairs is None:
        # If no element pairs are specified, consider all pairs
        use_all_pairs = True
    else:
        use_all_pairs = False
        # Create a lookup table for valid element pairs
        pair_table = {}
        for elem1, elem2 in element_pairs:
            pair_table[(elem1, elem2)] = True
            pair_table[(elem2, elem1)] = True  # Include reverse pairs too
    
    # Set up the NeighborList with the maximum distance
    cutoffs = [r_max/2] * n_atoms
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)
    
    # Count pairs in each distance bin
    hist = np.zeros(bins)
    
    # Keep track of central and target atom counts for normalization
    if element_pairs is not None and len(element_pairs) == 1:
        elem1, elem2 = element_pairs[0]
        n_center = elem_count.get(elem1, 0)
        n_target = elem_count.get(elem2, 0)
    else:
        n_center = n_atoms
        n_target = n_atoms
    
    # Count all valid pairs
    pair_count = 0
    
    for i in range(n_atoms):
        elem_i = symbols[i]
        
        # Skip if not the central element type for a specific element pair
        if not use_all_pairs:
            if all(elem_i != pair[0] for pair in element_pairs):
                continue
        
        indices, offsets = nl.get_neighbors(i)
        
        for j, offset in zip(indices, offsets):
            elem_j = symbols[j]
            
            # Skip if not in the requested element pairs
            if not use_all_pairs:
                if (elem_i, elem_j) not in pair_table:
                    continue
            
            # Calculate distance with PBC
            pos_i = positions[i]
            pos_j = positions[j] + np.dot(offset, cell)
            distance = np.linalg.norm(pos_j - pos_i)
            
            # Add to histogram if in range
            if r_min <= distance <= r_max:
                bin_index = int((distance - r_min) / (r_max - r_min) * bins)
                if 0 <= bin_index < bins:
                    hist[bin_index] += 1
                    pair_count += 1
    
    # Compute g(r)
    g_r = np.zeros_like(hist)
    for i in range(bins):
        # Volume of the shell
        shell_volume = 4 * np.pi * r_values[i]**2 * dr
        
        # Number density of target atoms
        if use_all_pairs:
            rho = (n_atoms - 1) / volume  # Exclude self
        else:
            # For specific element pairs
            if element_pairs[0][0] == element_pairs[0][1]:  # Same element
                rho = (n_target - 1) / volume  # Exclude self for same-element correlations
            else:
                rho = n_target / volume
        
        # Number of central atoms
        n_central = n_center
        
        # Normalize
        if shell_volume > 0 and rho > 0 and n_central > 0:
            g_r[i] = hist[i] / (n_central * shell_volume * rho)
    
    return r_values, g_r

def calculate_partial_rdfs(structure, r_range, bins, element_pairs=None, periodic=True):
    """
    Calculate partial radial distribution functions for specific element pairs.
    
    Input:
    structure : pymatgen.core.structure.Structure object
    r_range : tuple of floats
    bins : int
    element_pairs : list of tuples, optional e.g., [('Si', 'O')])
    periodic : bool
        
    Output:
    dict : Dictionary with element pairs as keys and (r_values, g_r) as values
    """
    if element_pairs is None:
        # Get unique element types and create all possible pairs
        elements = np.unique([site.specie.symbol for site in structure])
        element_pairs = []
        for i, elem1 in enumerate(elements):
            for elem2 in elements[i:]:  # Start from i to avoid duplicates
                element_pairs.append((elem1, elem2))
    
    # Calculate RDF for each element pair
    results = {}
    for pair in element_pairs:
        r_values, g_r = calculate_rdf(structure, r_range, bins, [pair], periodic)
        results[pair] = (r_values, g_r)
    
    return results

# =====================================================================
# RDF Ensemble Extension
# =====================================================================

def calculate_weights(structures_dict, temperature):
    """
    Calculate Boltzmann weights based on energies in the structures dictionary.
    
    Input:
    structures_dict : dict : Dictionary mapping structure IDs to structure-energy pairs
    temperature : float : Temperature in Kelvin
        
    Output:
    dict : Dictionary mapping structure IDs to weights
    """
    # Convert temperature to Ry (1 Ry = 13.605693009 eV)
    kB_Ry = 6.33362e-6  # Ry/K
    kT = kB_Ry * temperature
    
    # Extract energies
    energies = {}
    for struct_id, struct_data in structures_dict.items():
        if "Energy (Ry)" in struct_data:
            energies[struct_id] = struct_data["Energy (Ry)"]
    
    # Find the minimum energy
    if not energies:
        raise ValueError("No energy values found in structures dictionary")
    
    min_energy = min(energies.values())
    
    # Calculate unnormalized weights
    weights = {}
    for struct_id, energy in energies.items():
        rel_energy = energy - min_energy
        # Using the Boltzmann factor for energy in Ry units
        # Note: The factor of 24 seems specific to your application - keeping it as is
        weights[struct_id] = np.exp(-rel_energy / (24*kT))
    
    # Normalize weights
    total_weight = sum(weights.values())
    for struct_id in weights:
        weights[struct_id] /= total_weight
    
    return weights

def calculate_ensemble_rdf(structures_dict, weights, r_range=(0, 10), bins=200):
    """
    Calculate ensemble-averaged radial distribution function with proper density normalization.
    
    This implementation follows Equation 4 from the paper:
    g^(2)(r) = ∑ₐ (nₐ/⟨n⟩)² Pₐ g^(2)ₐ(r)
    
    Parameters:
    -----------
    structures_dict : dict
        Dictionary mapping structure IDs to structure-energy pairs
    weights : dict
        Dictionary mapping structure IDs to Boltzmann weights (Pₐ)
    r_range : tuple, optional
        (rmin, rmax) range for the RDF in Angstroms
    bins : int, optional
        Number of bins for the histogram
        
    Returns:
    --------
    tuple: (r_values, ensemble_rdf)
        r_values: array of r values
        ensemble_rdf: array of g(r) values
    """
    ensemble_rdf = None
    r_values = None
    
    # First pass: calculate average number density
    total_density = 0.0
    total_weight = 0.0
    
    for struct_id, struct_data in tqdm(structures_dict.items(), desc="Calculating average density"):
        if struct_id not in weights:
            print(f"Warning: No weight found for structure {struct_id}, skipping")
            continue
        
        weight = weights[struct_id]
        structure = struct_data["structure"]
        
        try:
            # Calculate number density (atoms per volume)
            n_atoms = len(structure)
            volume = structure.volume
            number_density = n_atoms / volume
            
            # Add to weighted average
            total_density += weight * number_density
            total_weight += weight
            
        except Exception as e:
            print(f"Error processing structure {struct_id} (density calculation): {e}")
    
    # Calculate average number density
    if total_weight > 0:
        avg_number_density = total_density / total_weight
    else:
        print("Warning: Total weight is zero, cannot calculate average density")
        return None, None
    
    # Second pass: calculate ensemble RDF with density normalization
    for struct_id, struct_data in tqdm(structures_dict.items(), desc="Calculating ensemble RDF"):
        if struct_id not in weights:
            continue  # Already warned in first pass
        
        weight = weights[struct_id]
        structure = struct_data["structure"]
        
        try:
            # Calculate number density (atoms per volume)
            n_atoms = len(structure)
            volume = structure.volume
            number_density = n_atoms / volume
            
            # Calculate the density normalization factor (nₐ/⟨n⟩)²
            density_factor = (number_density / avg_number_density) ** 2
            
            # Calculate the RDF
            r, rdf = calculate_rdf(structure, r_range, bins)
            
            # Store r values from first calculation
            if r_values is None:
                r_values = r
                ensemble_rdf = np.zeros_like(rdf)
            
            # Add weighted contribution to ensemble RDF with density normalization
            ensemble_rdf += weight * density_factor * rdf
            
        except Exception as e:
            print(f"Error processing structure {struct_id} (RDF calculation): {e}")
    
    return r_values, ensemble_rdf

def calculate_ensemble_partial_rdfs(structures_dict, weights, r_range, bins, element_pairs):
    """
    Calculate ensemble average RDFs for specific element pairs with density normalization.
    
    This implementation follows Equation 4 from the paper:
    g^(2)(r) = ∑ₐ (nₐ/⟨n⟩)² Pₐ g^(2)ₐ(r)
    
    For each element pair, the proper density normalization is applied.
    
    Parameters:
    -----------
    structures_dict : dict
        Dictionary mapping structure IDs to structure-energy pairs
    weights : dict
        Dictionary mapping structure IDs to weights
    r_range : tuple
        (rmin, rmax) range for the RDF
    bins : int
        Number of bins for the histogram
    element_pairs : list of tuples
        List of element pairs to consider
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, g_r) as values
    """
    # First pass: calculate average number density
    total_density = 0.0
    total_weight = 0.0
    
    for struct_id, struct_data in tqdm(structures_dict.items(), desc="Calculating average density"):
        if struct_id not in weights:
            print(f"Warning: No weight found for structure {struct_id}, skipping")
            continue
        
        weight = weights[struct_id]
        structure = struct_data["structure"]
        
        try:
            # Calculate number density (atoms per volume)
            n_atoms = len(structure)
            volume = structure.volume
            number_density = n_atoms / volume
            
            # Add to weighted average
            total_density += weight * number_density
            total_weight += weight
            
        except Exception as e:
            print(f"Error processing structure {struct_id} (density calculation): {e}")
    
    # Calculate average number density
    if total_weight > 0:
        avg_number_density = total_density / total_weight
    else:
        print("Warning: Total weight is zero, cannot calculate average density")
        return {}
    
    # Initialize dictionaries to store ensemble RDFs
    ensemble_rdfs = {pair: None for pair in element_pairs}
    r_values = None
    
    # Second pass: calculate ensemble partial RDFs with density normalization
    for struct_id, struct_data in tqdm(structures_dict.items(), desc="Calculating ensemble partial RDFs"):
        if struct_id not in weights:
            continue  # Already warned in first pass
        
        weight = weights[struct_id]
        structure = struct_data["structure"]
        
        try:
            # Calculate number density (atoms per volume)
            n_atoms = len(structure)
            volume = structure.volume
            number_density = n_atoms / volume
            
            # Calculate the density normalization factor (nₐ/⟨n⟩)²
            density_factor = (number_density / avg_number_density) ** 2
            
            # Calculate RDFs for each element pair
            partial_rdfs = calculate_partial_rdfs(structure, r_range, bins, element_pairs)
            
            # Initialize ensemble RDFs on first iteration
            if r_values is None:
                r_values = partial_rdfs[element_pairs[0]][0]
                for pair in element_pairs:
                    ensemble_rdfs[pair] = np.zeros_like(partial_rdfs[pair][1])
            
            # Add weighted contributions to ensemble RDFs with density normalization
            for pair in element_pairs:
                ensemble_rdfs[pair] += weight * density_factor * partial_rdfs[pair][1]
                
        except Exception as e:
            print(f"Error processing structure {struct_id}: {e}")
    
    # Create results dictionary with (r_values, g_r) tuples
    results = {pair: (r_values, ensemble_rdfs[pair]) for pair in element_pairs}
    
    return results

def plot_rdf_statistics(rdf_results, output_file=None, plot_all_samples=False):
    """
    Plot RDF statistics from the results of calculate_random_samples_rdf.
    
    Parameters:
    -----------
    rdf_results : dict
        Results dictionary from calculate_random_samples_rdf
    output_file : str, optional
        Path to save the plot, if provided
    plot_all_samples : bool
        Whether to plot all individual samples (can be messy with many samples)
    """
    plt.figure(figsize=(15, 10))
    
    element_pairs = list(rdf_results['reference'].keys())
    
    for i, pair in enumerate(element_pairs, 1):
        plt.subplot(2, 2, i)
        
        # Get data
        r_values, mean_rdf = rdf_results['reference'][pair]
        std_rdf = rdf_results['statistics'][pair]['std']
        min_rdf = rdf_results['statistics'][pair]['min']
        max_rdf = rdf_results['statistics'][pair]['max']
        
        # Plot individual samples if requested
        if plot_all_samples:
            for sample in rdf_results['samples']:
                _, sample_rdf = sample[pair]
                plt.plot(r_values, sample_rdf, color='gray', alpha=0.1)
        
        # Plot mean and std
        plt.plot(r_values, mean_rdf, color='blue', linewidth=2, label='Mean')
        plt.fill_between(r_values, mean_rdf - std_rdf, mean_rdf + std_rdf, 
                         color='blue', alpha=0.3, label='±1σ')
        
        # Plot min/max range
        plt.fill_between(r_values, min_rdf, max_rdf, 
                         color='lightblue', alpha=0.2, label='Min-Max Range')
        
        # Labels and formatting
        plt.title(f'{pair[0]}-{pair[1]} Partial RDF')
        plt.xlabel('Distance (Å)')
        plt.ylabel('g(r)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    plt.show()

def plot_single_rdf(struct_ids, structure_dict, r_range=(0, 10), bins=100, temperature=1800, 
                    figsize=(8, 10), label=None, save_path=None, show_plot =True):
    """
    Plot RDF for structures specified by IDs from a structure dictionary.
    
    Parameters:
    -----------
    struct_ids : numpy.ndarray
        Array of structure IDs to use for RDF calculation
    structure_dict : dict
        Dictionary mapping structure IDs to structure-energy pairs
    r_range : tuple, optional
        (rmin, rmax) range for the RDF in Angstroms
    bins : int, optional
        Number of bins for the histogram
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting
    figsize : tuple, optional
        Figure size (width, height) in inches
    label : str, optional
        Label for the plot legend
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, g_r) values
    """
    
    # Extract the subset of structures specified by struct_ids
    subset_dict = {}
    for id_str in struct_ids:
        id_str = str(id_str)  # Ensure string format for dictionary key
        if id_str in structure_dict:
            subset_dict[id_str] = structure_dict[id_str]
        else:
            print(f"Warning: ID {id_str} not found in structure dictionary")
    
    if not subset_dict:
        print("Error: No valid structures found for the provided IDs")
        return {}
    
    elements = ['Si', 'O']  # Specify the elements of interest
    print(f"Elements in structures: {elements}")

    # Set up element pairs for partial RDFs
    element_pairs = []
    for i, elem1 in enumerate(elements):
        for j in range(i, len(elements)):
            elem2 = elements[j]
            element_pairs.append((elem1, elem2))
    
    # Calculate Boltzmann weights at the specified temperature
    weights = calculate_weights(subset_dict, temperature)
    print(f"Calculated Boltzmann weights at {temperature} K")

    # Calculate ensemble partial RDFs
    print("Calculating ensemble partial RDFs...")
    ensemble_rdfs = calculate_ensemble_partial_rdfs(
        subset_dict, 
        weights, 
        r_range, 
        bins, 
        element_pairs
    )

    if show_plot:
        # Plot the partial RDFs
        fig = plt.figure(figsize=figsize)
        
        # Iterate through the partial RDFs
        for i, ((elem1, elem2), (r, rdf)) in enumerate(ensemble_rdfs.items(), 1):
            plt.subplot(3, 1, i)
            plt.plot(r, rdf, label=label)
            plt.title(f'{elem1}-{elem2} Partial RDF')
            plt.xlabel('Distance (Å)')
            plt.ylabel('g(r)')
            plt.grid(True)
            if label:
                plt.legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Figure saved to {save_path}")
        
        plt.show()

    return ensemble_rdfs

def plot_saved_rdf(ensemble_rdfs, figsize=(8, 10), label=None, save_path=None):
    """
    Plot previously calculated RDF data.
    
    Parameters:
    -----------
    ensemble_rdfs : dict
        Dictionary with element pairs as keys and (r_values, g_r) as values
    figsize : tuple, optional
        Figure size (width, height) in inches
    label : str, optional
        Label for the plot legend
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    None
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=figsize)
    
    # Iterate through the partial RDFs
    for i, ((elem1, elem2), (r, rdf)) in enumerate(ensemble_rdfs.items(), 1):
        plt.subplot(3, 1, i)
        plt.plot(r, rdf, label=label)
        plt.title(f'{elem1}-{elem2} Partial RDF')
        plt.xlabel('Distance (Å)')
        plt.ylabel('g(r)')
        plt.grid(True)
        if label:
            plt.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_multiple_rdfs(*data_items, structure_dict=None, labels=None, r_range=(0, 10), bins=100, 
                       temperature=1800, figsize=(8, 10), save_path=None):
    """
    Plot RDFs from different sources: structure IDs, pre-calculated ensemble RDFs, or file paths.
    
    Parameters:
    -----------
    *data_items : variable number of items, each can be:
        - numpy.ndarray or list of structure IDs (requires structure_dict)
        - dict containing pre-calculated ensemble RDFs
        - str path to a file containing saved ensemble RDFs
    structure_dict : dict, optional
        Dictionary mapping structure IDs to structure-energy pairs
        (Required if any data_item is an array or list of structure IDs)
    labels : list of str, optional
        Labels for each item in the legend
    r_range : tuple, optional
        (rmin, rmax) range for the RDF in Angstroms (for ID-based calculations)
    bins : int, optional
        Number of bins for the histogram (for ID-based calculations)
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting (for ID-based calculations)
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the combined figure. If None, figure is not saved
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and lists of (r_values, g_r, idx) tuples as values
    
    Examples:
    --------
    # With structure IDs only:
    plot_multiple_rdfs(random_ids, energy_ids, maxmin_ids, 
                      structure_dict=structure_dict,
                      labels=["Random", "Energy", "MaxMin"])
                      
    # With pre-calculated ensemble RDFs only:
    plot_multiple_rdfs(random_rdf, energy_rdf, maxmin_rdf,
                      labels=["Random", "Energy", "MaxMin"])
                      
    # With file paths:
    plot_multiple_rdfs("random_rdf.pkl", "energy_rdf.pkl", "maxmin_rdf.pkl",
                      labels=["Random", "Energy", "MaxMin"])
                      
    # With mixed sources:
    plot_multiple_rdfs(random_ids, energy_rdf, "maxmin_rdf.pkl",
                      structure_dict=structure_dict,
                      labels=["Random", "Energy", "MaxMin"])
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Check if any data items were provided
    if len(data_items) == 0:
        raise ValueError("At least one data item must be provided")
    
    # Initialize dictionary to store all RDFs
    all_rdfs = {}
    
    # Process each data item
    for i, item in enumerate(data_items):
        label = labels[i] if labels and i < len(labels) else f"Set {i+1}"
        
        # Get ensemble RDFs based on the item type
        if isinstance(item, (list, np.ndarray)):
            # Item is an array of structure IDs
            if structure_dict is None:
                raise ValueError("structure_dict must be provided when using structure ID arrays")
            
            # Extract the subset of structures specified by struct_ids
            subset_dict = {}
            for id_str in item:
                id_str = str(id_str)  # Ensure string format for dictionary key
                if id_str in structure_dict:
                    subset_dict[id_str] = structure_dict[id_str]
                else:
                    print(f"Warning: ID {id_str} not found in structure dictionary")
            
            if not subset_dict:
                print(f"Warning: No valid structures found for item {i}, skipping")
                continue
            
            # Get elements
            elements = ['Si', 'O']  # Specify the elements of interest
            
            # Set up element pairs for partial RDFs
            element_pairs = []
            for i_elem, elem1 in enumerate(elements):
                for j_elem in range(i_elem, len(elements)):
                    elem2 = elements[j_elem]
                    element_pairs.append((elem1, elem2))
            
            # Calculate Boltzmann weights and ensemble RDFs
            weights = calculate_weights(subset_dict, temperature)
            ensemble_rdfs = calculate_ensemble_partial_rdfs(
                subset_dict, weights, r_range, bins, element_pairs
            )
            
        elif isinstance(item, dict) and all(isinstance(k, tuple) and len(k) == 2 for k in item.keys()):
            # Item is a pre-calculated ensemble RDFs dictionary
            ensemble_rdfs = item
            
        elif isinstance(item, str) and os.path.exists(item):
            # Item is a file path to saved ensemble RDFs
            ensemble_rdfs = dm.load_rdf_data(item)
            if ensemble_rdfs is None:
                print(f"Warning: Failed to load RDF data from {item}, skipping")
                continue
                
        else:
            print(f"Warning: Unrecognized data item type at index {i}, skipping")
            continue
        
        # Store in the all_rdfs dictionary
        for pair, (r, rdf) in ensemble_rdfs.items():
            if pair not in all_rdfs:
                all_rdfs[pair] = []
            all_rdfs[pair].append((r, rdf, i))  # Store r, rdf, and the set index
    
    # Create the combined plots
    # Plot all RDFs for each element pair on the same figure
    n_pairs = len(all_rdfs)
    if n_pairs == 0:
        print("Warning: No valid RDF data to plot")
        return all_rdfs
        
    n_rows = min(3, n_pairs)
    n_cols = (n_pairs + n_rows - 1) // n_rows
    
    plt.figure(figsize=figsize)
    
    for i, (pair, rdf_list) in enumerate(all_rdfs.items()):
        plt.subplot(n_rows, n_cols, i+1)
        
        for r, rdf, set_idx in rdf_list:
            if labels and set_idx < len(labels):
                plt.plot(r, rdf, label=labels[set_idx])
            else:
                plt.plot(r, rdf, label=f"Set {set_idx+1}")
        
        plt.title(f'{pair[0]}-{pair[1]} Partial RDF')
        plt.xlabel('Distance (Å)')
        plt.ylabel('g(r)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Combined figure saved to {save_path}")
    
    plt.show()
    
    return all_rdfs

# =====================================================================
# Utility Functions for Converting Between RDF and Counting Functions
# =====================================================================

def simple_counting_function(struct_id, structure_dict, element_pair, r_range=(0, 10), bins=100):
    """
    Calculate counting function for a single structure and element pair.
    Includes all periodic images.
    
    Parameters:
    -----------
    struct_id : str
        ID of the structure to analyze
    structure_dict : dict
        Dictionary mapping structure IDs to structure-energy pairs
    element_pair : tuple
        (center_element, neighbor_element) e.g., ('Si', 'O')
    r_range : tuple, optional
        (r_min, r_max) in Angstroms
    bins : int, optional
        Number of bins
        
    Returns:
    --------
    r_values : numpy.ndarray
        Distance values
    counting_function : numpy.ndarray
        Cumulative coordination numbers
    """
    
    # Get the structure
    structure = structure_dict[str(struct_id)]["structure"]
    
    # Extract element types
    center_element, neighbor_element = element_pair
    
    # Get all atomic sites
    sites = structure.sites
    
    # Find center atoms
    center_sites = [site for site in sites if site.specie.symbol == center_element]
    
    # Collect all distances between center atoms and neighbor atoms
    distances = []
    r_min, r_max = r_range
    
    for center_site in center_sites:
        # Get ALL neighbors within r_max, including periodic images
        neighbors = structure.get_neighbors(center_site, r_max)
        
        for neighbor_info in neighbors:
            neighbor_site = neighbor_info[0]  # The actual site
            distance = neighbor_info[1]
            # Check if this neighbor is the right element type
            if neighbor_site.specie.symbol == neighbor_element:
                # Skip self-interaction for same-element pairs
                if (center_element == neighbor_element and 
                    np.allclose(center_site.coords, neighbor_site.coords, atol=1e-3)):
                    continue
                    
                # Only keep distances in our range
                if r_min <= distance <= r_max:
                    distances.append(distance)
    
    # Create r_values (bin centers)
    r_values = np.linspace(r_min, r_max, bins)
    
    # Create histogram
    hist, bin_edges = np.histogram(distances, bins=bins, range=r_range)
    
    # Convert to cumulative (counting function)
    cumulative = np.cumsum(hist)
    
    # Normalize by number of center atoms
    counting_function = cumulative / len(center_sites)
    
    return r_values, counting_function

def calculate_partial_counting_functions(struct_id, structure_dict, element_pairs=None, r_range=(0, 10), bins=100):
    """
    Calculate counting functions for multiple element pairs using the simple base function.
    
    Parameters:
    -----------
    struct_id : str
        ID of the structure to analyze
    structure_dict : dict
        Dictionary mapping structure IDs to structure-energy pairs
    element_pairs : list of tuples, optional
        List of element pairs, e.g., [('Si', 'O'), ('Si', 'Si'), ('O', 'O')]
    r_range : tuple, optional
        (r_min, r_max) in Angstroms
    bins : int, optional
        Number of bins
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, counting_function) as values
    """
    
    if element_pairs is None:
        # Get unique element types and create all possible pairs
        structure = structure_dict[str(struct_id)]["structure"]
        elements = list(set([site.specie.symbol for site in structure.sites]))
        element_pairs = []
        for i, elem1 in enumerate(elements):
            for j in range(i, len(elements)):
                elem2 = elements[j]
                element_pairs.append((elem1, elem2))
    
    # Calculate counting function for each element pair
    results = {}
    for pair in element_pairs:
        r_values, counting_func = simple_counting_function(struct_id, structure_dict, pair, r_range, bins)
        results[pair] = (r_values, counting_func)
    
    return results

# =====================================================================
# Ensemble Averaging Functions
# =====================================================================

def calculate_ensemble_partial_counting_functions(structures_dict, weights, r_range=(0, 10), bins=100, element_pairs=None):
   """
   Calculate ensemble-averaged counting functions using the simple base function.
   
   Parameters:
   -----------
   structures_dict : dict
       Dictionary mapping structure IDs to structure-energy pairs
   weights : dict
       Dictionary mapping structure IDs to weights
   r_range : tuple, optional
       (r_min, r_max) range for the counting function in Angstroms
   bins : int, optional
       Number of bins for the histogram
   element_pairs : list of tuples, optional
       List of element pairs to consider, e.g., [('Si', 'O'), ('Si', 'Si'), ('O', 'O')]
       
   Returns:
   --------
   dict : Dictionary with element pairs as keys and (r_values, counting_function) as values
   """
   
   if element_pairs is None:
       # Get elements from first structure
       first_struct = list(structures_dict.values())[0]["structure"]
       elements = list(set([site.specie.symbol for site in first_struct.sites]))
       element_pairs = []
       for i, elem1 in enumerate(elements):
           for j in range(i, len(elements)):
               elem2 = elements[j]
               element_pairs.append((elem1, elem2))
   
   # First pass: calculate average number density
   total_density = 0.0
   total_weight = 0.0
   
   for struct_id, struct_data in tqdm(structures_dict.items(), desc="Calculating average density"):
       if struct_id not in weights:
           print(f"Warning: No weight found for structure {struct_id}, skipping")
           continue
       
       weight = weights[struct_id]
       structure = struct_data["structure"]
       
       try:
           # Calculate number density (atoms per volume)
           n_atoms = len(structure)
           volume = structure.volume
           number_density = n_atoms / volume
           
           # Add to weighted average
           total_density += weight * number_density
           total_weight += weight
           
       except Exception as e:
           print(f"Error processing structure {struct_id} (density calculation): {e}")
   
   # Calculate average number density
   if total_weight > 0:
       avg_number_density = total_density / total_weight
   else:
       print("Warning: Total weight is zero, cannot calculate average density")
       return {}
   
   # Initialize ensemble counting functions
   ensemble_countings = {pair: None for pair in element_pairs}
   r_values = None
   
   # Second pass: calculate ensemble averages with density normalization
   for struct_id, struct_data in tqdm(structures_dict.items(), desc="Calculating ensemble counting functions"):
       if struct_id not in weights:
           continue
       
       weight = weights[struct_id]
       structure = struct_data["structure"]
       
       try:
           # Calculate number density (atoms per volume)
           n_atoms = len(structure)
           volume = structure.volume
           number_density = n_atoms / volume
           
           # Calculate the density normalization factor (nₐ/⟨n⟩)²
           density_factor = (number_density / avg_number_density) ** 2
           
           # Calculate counting functions for all element pairs for this structure
           partial_countings = calculate_partial_counting_functions(
               struct_id, {struct_id: struct_data}, element_pairs, r_range, bins
           )
           
           # Initialize ensemble counting functions on first iteration
           if r_values is None:
               r_values = partial_countings[element_pairs[0]][0]
               for pair in element_pairs:
                   ensemble_countings[pair] = np.zeros_like(partial_countings[pair][1])
           
           # Add weighted contributions with density normalization
           for pair in element_pairs:
               ensemble_countings[pair] += weight * density_factor * partial_countings[pair][1]
               
       except Exception as e:
           print(f"Error processing structure {struct_id}: {e}")
   
   # Create results dictionary
   results = {pair: (r_values, ensemble_countings[pair]) for pair in element_pairs}
   
   return results

# =====================================================================
# Plotting Functions
# =====================================================================

def plot_single_counting_function(struct_ids, structure_dict, r_range=(0, 10), bins=100, temperature=1800, 
                                 figsize=(8, 10), label=None, save_path=None, show_plot=True):
    """
    Plot counting functions for structures specified by IDs from a structure dictionary.
    
    Parameters:
    -----------
    struct_ids : numpy.ndarray or list
        Array of structure IDs to use for counting function calculation
    structure_dict : dict
        Dictionary mapping structure IDs to structure-energy pairs
    r_range : tuple, optional
        (rmin, rmax) range for the counting function in Angstroms
    bins : int, optional
        Number of bins for the histogram
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting
    figsize : tuple, optional
        Figure size (width, height) in inches
    label : str, optional
        Label for the plot legend
    save_path : str, optional
        Path to save the figure
    show_plot : bool, optional
        Whether to display the plot
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, counting_function) values
    """
    # Extract the subset of structures
    subset_dict = {}
    for id_str in struct_ids:
        id_str = str(id_str)
        if id_str in structure_dict:
            subset_dict[id_str] = structure_dict[id_str]
        else:
            print(f"Warning: ID {id_str} not found in structure dictionary")
    
    if not subset_dict:
        print("Error: No valid structures found for the provided IDs")
        return {}
    
    # Default element pairs for SiO2
    element_pairs = [('Si', 'Si'), ('Si', 'O'), ('O', 'O')]
    
    # Calculate Boltzmann weights
    weights = calculate_weights(subset_dict, temperature)
    print(f"Calculated Boltzmann weights at {temperature} K")
    
    # Calculate ensemble partial counting functions
    print("Calculating ensemble partial counting functions...")
    ensemble_countings = calculate_ensemble_partial_counting_functions(
        subset_dict, weights, r_range, bins, element_pairs
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
            print(f"Figure saved to {save_path}")
        
        plt.show()

    return ensemble_countings

def plot_multiple_counting_functions(*data_items, structure_dict=None, labels=None, r_range=(0, 10), bins=100, 
                                   temperature=1800, figsize=(8, 10), save_path=None):
    """
    Plot counting functions from different sources: structure IDs, pre-calculated ensemble counting functions, or file paths.
    
    Parameters:
    -----------
    *data_items : variable number of items, each can be:
        - numpy.ndarray or list of structure IDs (requires structure_dict)
        - dict containing pre-calculated ensemble counting functions
        - str path to a file containing saved ensemble counting functions
    structure_dict : dict, optional
        Dictionary mapping structure IDs to structure-energy pairs
        (Required if any data_item is an array or list of structure IDs)
    labels : list of str, optional
        Labels for each item in the legend
    r_range : tuple, optional
        (rmin, rmax) range for the counting function in Angstroms (for ID-based calculations)
    bins : int, optional
        Number of bins for the histogram (for ID-based calculations)
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting (for ID-based calculations)
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the combined figure. If None, figure is not saved
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and lists of (r_values, counting_function, idx) tuples as values
    """
    import numpy as np
    import os
    
    # Check if any data items were provided
    if len(data_items) == 0:
        raise ValueError("At least one data item must be provided")
    
    # Initialize dictionary to store all counting functions
    all_countings = {}
    
    # Process each data item
    for i, item in enumerate(data_items):
        label = labels[i] if labels and i < len(labels) else f"Set {i+1}"
        
        # Get ensemble counting functions based on the item type
        if isinstance(item, (list, np.ndarray)):
            # Item is an array of structure IDs
            if structure_dict is None:
                raise ValueError("structure_dict must be provided when using structure ID arrays")
            
            ensemble_countings = plot_single_counting_function(
                item, structure_dict, r_range, bins, temperature, show_plot=False
            )
            
        elif isinstance(item, dict) and all(isinstance(k, tuple) and len(k) == 2 for k in item.keys()):
            # Item is a pre-calculated ensemble counting functions dictionary
            ensemble_countings = item
            
        elif isinstance(item, str) and os.path.exists(item):
            # Item is a file path to saved ensemble counting functions
            ensemble_countings = dm.load_rdf_data(item)  # Assumes this can load counting functions
            if ensemble_countings is None:
                print(f"Warning: Failed to load counting function data from {item}, skipping")
                continue
                
        else:
            print(f"Warning: Unrecognized data item type at index {i}, skipping")
            continue
        
        # Store in the all_countings dictionary
        for pair, (r, counting) in ensemble_countings.items():
            if pair not in all_countings:
                all_countings[pair] = []
            all_countings[pair].append((r, counting, i))
    
    # Create the combined plots
    n_pairs = len(all_countings)
    if n_pairs == 0:
        print("Warning: No valid counting function data to plot")
        return all_countings
        
    n_rows = min(3, n_pairs)
    n_cols = (n_pairs + n_rows - 1) // n_rows
    
    plt.figure(figsize=figsize)
    
    for i, (pair, counting_list) in enumerate(all_countings.items()):
        plt.subplot(n_rows, n_cols, i+1)
        
        for r, counting, set_idx in counting_list:
            if labels and set_idx < len(labels):
                plt.plot(r, counting, label=labels[set_idx])
            else:
                plt.plot(r, counting, label=f"Set {set_idx+1}")
        
        plt.title(f'{pair[0]}-{pair[1]} Counting Function')
        plt.xlabel('Distance (Å)')
        plt.ylabel('N(r)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Combined figure saved to {save_path}")
    
    plt.show()
    
    return all_countings

# =====================================================================
# Conversion Functions Between RDF and Counting Functions
# =====================================================================

def convert_rdf_to_counting_function(ensemble_rdfs, number_density=None):
    """
    Convert ensemble RDF dictionary to counting function dictionary.
    Takes the same format as plot_saved_rdf and outputs format for plot_multiple_counting_functions.
    
    Parameters:
    -----------
    ensemble_rdfs : dict
        Dictionary with element pairs as keys and (r_values, g_r) as values
        (same format as returned by plot_single_rdf or loaded RDF data)
    number_density : float, optional
        Number density of the system (atoms per volume)
        If None, will use a default that gives reasonable relative values
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, counting_function) as values
        (same format as expected by plot_multiple_counting_functions)
    """
    ensemble_countings = {}
    
    for pair, (r_values, g_r) in ensemble_rdfs.items():
        # Calculate counting function from RDF
        dr = r_values[1] - r_values[0] if len(r_values) > 1 else 0.05
        
        # Use default density if not provided
        if number_density is None:
            # Use a reasonable default density for SiO2 (atoms per Å³)
            rho = 0.066  # approximately correct for SiO2
        else:
            rho = number_density
        
        # Convert g(r) to pair correlation: 4πr²ρ[g(r)-1] + 4πr²ρ
        # The -1 part integrates to give relative coordination
        # The +1 part gives the uniform background contribution
        pair_correlation = 4 * np.pi * r_values**2 * rho * (g_r - 1)
        
        # Integrate to get counting function
        counting_function = np.cumsum(pair_correlation) * dr
        
        # Add the uniform background contribution
        # This gives absolute coordination numbers
        uniform_background = (4/3) * np.pi * r_values**3 * rho
        counting_function = counting_function + uniform_background
        
        ensemble_countings[pair] = (r_values, counting_function)
    
    return ensemble_countings

def convert_counting_function_to_rdf(ensemble_countings, number_density=None):
    """
    Convert ensemble counting function dictionary back to RDF dictionary.
    Takes output from counting functions and converts to format for plot_multiple_rdfs.
    
    Parameters:
    -----------
    ensemble_countings : dict
        Dictionary with element pairs as keys and (r_values, counting_function) as values
        (same format as returned by counting function calculations)
    number_density : float, optional
        Number density of the system (atoms per volume)
        If None, will estimate from the data
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, g_r) as values
        (same format as expected by plot_multiple_rdfs)
    """
    ensemble_rdfs = {}
    
    for pair, (r_values, counting_function) in ensemble_countings.items():
        # Calculate RDF from counting function
        dr = r_values[1] - r_values[0] if len(r_values) > 1 else 0.05
        
        # Estimate density if not provided
        if number_density is None:
            # Estimate from asymptotic behavior of counting function
            # At large r, N(r) ≈ (4/3)πr³ρ, so ρ ≈ 3N(r)/(4πr³)
            if len(counting_function) > 10:
                r_large = r_values[-5:]  # Use last few points
                N_large = counting_function[-5:]
                rho_estimates = 3 * N_large / (4 * np.pi * r_large**3)
                rho = np.mean(rho_estimates[rho_estimates > 0])  # Avoid division issues
            else:
                rho = 0.066  # Default for SiO2
        else:
            rho = number_density
        
        # Take derivative to get pair correlation
        pair_correlation = np.gradient(counting_function, dr)
        
        # Remove uniform background: dN/dr = 4πr²ρ for uniform distribution
        uniform_background_derivative = 4 * np.pi * r_values**2 * rho
        pair_correlation_excess = pair_correlation - uniform_background_derivative
        
        # Convert back to g(r): g(r) = 1 + excess/(4πr²ρ)
        # Avoid division by zero at r=0
        r_safe = np.where(r_values == 0, np.finfo(float).eps, r_values)
        g_r = 1 + pair_correlation_excess / (4 * np.pi * r_safe**2 * rho)
        
        g_r[:3] = 0.0
        ensemble_rdfs[pair] = (r_values, g_r)
    
    return ensemble_rdfs
