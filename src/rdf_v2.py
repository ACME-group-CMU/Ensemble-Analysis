import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import NeighborList
from src.data_management_v2 import load_energies, load_rdfs, load_counting_functions


# =====================================================================
# CORE RDF CALCULATION FUNCTIONS
# =====================================================================

def calculate_rdf(structure, r_range, bins, element_pairs=None, periodic=True):
    """
    Calculate radial distribution function with proper normalization.
    
    Parameters:
    -----------
    structure : pymatgen.core.structure.Structure object
    r_range : tuple of floats - (r_min, r_max) in Angstroms
    bins : int - number of bins
    element_pairs : list of tuples, optional e.g., [('Si', 'O')])
    periodic : bool
        
    Returns:
    --------
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
        
        # Use consistent partial densities for proper mathematical normalization
        if use_all_pairs:
            rho = n_atoms / volume
            n_central = n_atoms
        else:
            # For partial RDFs, use consistent element-specific densities
            elem1, elem2 = element_pairs[0]
            n_central = elem_count.get(elem1, 0)  # Central atoms (Si for Si-Si, Si-O)
            
            if elem1 == elem2:
                # Same element pair (Si-Si, O-O): use that element's density
                rho = elem_count.get(elem1, 0) / volume
            else:
                # Different elements (Si-O): use target element's density
                rho = elem_count.get(elem2, 0) / volume
        
        # Number of central atoms
        n_central = n_center
        
        # Normalize
        if shell_volume > 0 and rho > 0 and n_central > 0:
            g_r[i] = hist[i] / (n_central * shell_volume * rho)
    
    return r_values, g_r

def calculate_partial_rdfs(structure, r_range, bins, element_pairs=None, periodic=True):
    """
    Calculate partial radial distribution functions for specific element pairs.
    
    Parameters:
    -----------
    structure : pymatgen.core.structure.Structure object
    r_range : tuple of floats
    bins : int
    element_pairs : list of tuples, optional e.g., [('Si', 'O')])
    periodic : bool
        
    Returns:
    --------
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
# WEIGHT CALCULATION
# =====================================================================

def calculate_weights(structures_dict, temperature):
    """
    Calculate Boltzmann weights based on energies in the structures dictionary.
    
    Parameters:
    -----------
    structures_dict : dict : Dictionary mapping structure IDs to structure-energy pairs
    temperature : float : Temperature in Kelvin
        
    Returns:
    --------
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

# =====================================================================
# ENSEMBLE AVERAGING FUNCTIONS
# =====================================================================

def calculate_ensemble_rdf(struct_ids=None, rdfs=None, energies=None, use_weights = False,temperature=1800, 
                          r_range=(0, 10), bins=200):
    """
    Calculate ensemble-averaged radial distribution function with proper density normalization.
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs to use (if rdfs/energies not provided)
    rdfs : dict, optional
        Pre-loaded RDF data {struct_id: (r_values, g_r)}
    energies : dict, optional
        Pre-loaded energy data {struct_id: energy}
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting
    r_range : tuple, optional
        (rmin, rmax) range for the RDF in Angstroms
    bins : int, optional
        Number of bins for the histogram
        
    Returns:
    --------
    tuple: (r_values, ensemble_rdf)
    """
    # Auto-load if not provided
    if rdfs is None:
        if struct_ids is None:
            raise ValueError("Must provide either struct_ids or rdfs")
        rdfs = load_rdfs(struct_ids, pairs='total')
    
    if energies is None:
        if struct_ids is None:
            # Try to infer struct_ids from rdfs
            struct_ids = list(rdfs.keys())
        energies = load_energies(struct_ids)
    
    # Create legacy format for weight calculation
    legacy_dict = {}
    for struct_id in rdfs.keys():
        if struct_id in energies:
            legacy_dict[str(struct_id)] = {"Energy (Ry)": energies[struct_id]}
    
    if use_weights:
        weights = calculate_weights(legacy_dict, temperature)
    else:
        # Equal weights for all structures
        weights = {str(k): 1.0/len(legacy_dict) for k in legacy_dict.keys()}

    ensemble_rdf = None
    r_values = None

    # Calculate average density (simplified)
    avg_number_density = 0.066  # Default for SiO2

    # Calculate ensemble RDF with density normalization
    for struct_id, (r, rdf) in tqdm(rdfs.items(), desc="Calculating ensemble RDF"):
        if struct_id not in weights:
            continue

        weight = weights[struct_id]
        
        # Calculate the density normalization factor (nₐ/⟨n⟩)²
        number_density = 0.066  # Default for SiO2
        density_factor = (number_density / avg_number_density) ** 2
        
        # Store r values from first calculation
        if r_values is None:
            r_values = r
            ensemble_rdf = np.zeros_like(rdf)
        
        # Add weighted contribution to ensemble RDF with density normalization
        ensemble_rdf += weight * density_factor * rdf
    
    return r_values, ensemble_rdf

def calculate_ensemble_partial_rdfs(struct_ids=None, rdfs=None, energies=None, 
                                   densities=None, use_weights=False, temperature=1800, 
                                   element_pairs=None, smoothed=True):
    """
    Calculate ensemble average partial RDFs with proper density normalization.
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs to use
    rdfs : dict, optional
        Pre-loaded RDF data {struct_id: {pair: (r_values, g_r)}}
    energies : dict, optional
        Pre-loaded energy data {struct_id: energy}
    densities : dict, optional
        Pre-loaded density data {struct_id: density_data}
    use_weights : bool, optional
        Whether to use Boltzmann weighting (default: False)
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting
    element_pairs : list of tuples, optional
        List of element pairs to consider
    smoothed : bool, optional
        Whether to load smoothed RDFs
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, g_r) as values
    """
    if element_pairs is None:
        element_pairs = [('Si', 'Si'), ('Si', 'O'), ('O', 'O')]
    
    # Auto-load if not provided
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
        # Import here to avoid circular imports
        from src.data_management_v2 import load_densities
        densities = load_densities(struct_ids)
    
    # Create legacy format for weight calculation
    legacy_dict = {}
    for struct_id in rdfs.keys():
        if struct_id in energies:
            legacy_dict[str(struct_id)] = {"Energy (Ry)": energies[struct_id]}
    
    # Calculate weights based on use_weights parameter
    if use_weights:
        weights = calculate_weights(legacy_dict, temperature)
    else:
        # Equal weights for all structures
        weights = {str(k): 1.0/len(legacy_dict) for k in legacy_dict.keys()}

    # Calculate ensemble average densities for normalization
    from src.data_management_v2 import calculate_ensemble_average_density
    avg_densities = calculate_ensemble_average_density(list(rdfs.keys()), energies, temperature)

    # Initialize ensemble RDFs
    ensemble_rdfs = {pair: None for pair in element_pairs}
    r_values = None

    # Calculate ensemble partial RDFs with proper density normalization
    for struct_id, struct_rdfs in tqdm(rdfs.items(), desc="Calculating ensemble partial RDFs"):
        if struct_id not in weights or struct_id not in densities:
            continue
        
        weight = weights[struct_id]
        density_data = densities[struct_id]
        
        # Initialize ensemble RDFs on first iteration
        if r_values is None and element_pairs[0] in struct_rdfs:
            r_values, _ = struct_rdfs[element_pairs[0]]
            for pair in element_pairs:
                if pair in struct_rdfs:
                    ensemble_rdfs[pair] = np.zeros_like(struct_rdfs[pair][1])
        
        # Process each element pair
        for pair in element_pairs:
            if pair not in struct_rdfs:
                continue
            
            r, g_r = struct_rdfs[pair]
            
            # Calculate density normalization factor for this pair
            elem1, elem2 = pair
            struct_density1 = density_data['partial_densities'].get(elem1, 0)
            struct_density2 = density_data['partial_densities'].get(elem2, 0)
            avg_density1 = avg_densities['partial_densities'].get(elem1, 0)
            avg_density2 = avg_densities['partial_densities'].get(elem2, 0)
            
            # For same-element pairs, use single density
            if elem1 == elem2:
                if avg_density1 > 0:
                    density_factor = struct_density1 / avg_density1
                else:
                    density_factor = 1.0
            else:
                # Use total density for mixed pairs (simplified approach)
                struct_total = density_data['total_density'] 
                avg_total = avg_densities['total_density']
                if avg_total > 0:
                    density_factor = struct_total / avg_total
                else:
                    density_factor = 1.0
            
            # Normalize the RDF: g_normalized(r) = (g(r) - 1) * density_factor + 1
            g_normalized = (g_r - 1) * density_factor + 1
            
            # Add weighted contribution to ensemble RDF
            if ensemble_rdfs[pair] is not None:
                ensemble_rdfs[pair] += weight * g_normalized
    
    # Convert back to (r_values, g_r) format
    final_ensemble_rdfs = {}
    for pair in element_pairs:
        if ensemble_rdfs[pair] is not None:
            final_ensemble_rdfs[pair] = (r_values, ensemble_rdfs[pair])
    
    return final_ensemble_rdfs

# =====================================================================
# COUNTING FUNCTION IMPLEMENTATIONS
# =====================================================================

def simple_counting_function(struct_id, structure_dict, element_pair, r_range=(0, 10), bins=100):
    """
    Calculate counting function for a single structure and element pair.
    
    Parameters:
    -----------
    struct_id : int
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
    counting_function : numpy.ndarray
    """
    # Get the structure
    structure = structure_dict[struct_id]["structure"]
    
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

def calculate_partial_counting_functions(structure_dict, struct_id, element_pairs=None, 
                                       r_range=(0, 10), bins=100):
    """
    Calculate counting functions for multiple element pairs.
    
    Parameters:
    -----------
    structure_dict : dict
        Dictionary mapping structure IDs to structure data
    struct_id : int
        ID of the structure to analyze
    element_pairs : list of tuples, optional
        List of element pairs
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
        structure = structure_dict[struct_id]["structure"]
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

def calculate_ensemble_partial_counting_functions(struct_ids=None, counting_functions=None, 
                                                 energies=None, densities=None, use_weights=False,
                                                 temperature=1800, element_pairs=None):
    """
    Calculate ensemble-averaged counting functions with proper density normalization.
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs to use
    counting_functions : dict, optional
        Pre-loaded counting function data
    energies : dict, optional
        Pre-loaded energy data
    densities : dict, optional
        Pre-loaded density data {struct_id: density_data}
    use_weights : bool, optional
        Whether to use Boltzmann weighting (default: False)
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting
    element_pairs : list of tuples, optional
        List of element pairs to consider
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, counting_function) as values
    """
    
    if element_pairs is None:
        element_pairs = [('Si', 'Si'), ('Si', 'O'), ('O', 'O')]
    
    # Auto-load if not provided
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
        # Import here to avoid circular imports
        from src.data_management_v2 import load_densities
        densities = load_densities(struct_ids)
    
    # Create legacy format for weight calculation
    legacy_dict = {}
    for struct_id in counting_functions.keys():
        if struct_id in energies:
            legacy_dict[str(struct_id)] = {"Energy (Ry)": energies[struct_id]}
    
    # Calculate weights based on use_weights parameter
    if use_weights:
        weights = calculate_weights(legacy_dict, temperature)
    else:
        # Equal weights for all structures
        weights = {str(k): 1.0/len(legacy_dict) for k in legacy_dict.keys()}

    # Calculate ensemble average densities for normalization
    from src.data_management_v2 import calculate_ensemble_average_density
    avg_densities = calculate_ensemble_average_density(list(counting_functions.keys()), energies, temperature)

    # Initialize ensemble counting functions
    ensemble_countings = {pair: None for pair in element_pairs}
    r_values = None

    # Calculate ensemble averages with proper density normalization
    for struct_id, struct_cfs in tqdm(counting_functions.items(), desc="Calculating ensemble counting functions"):
        if struct_id not in weights or struct_id not in densities:
            continue
        
        weight = weights[struct_id]
        density_data = densities[struct_id]
        
        # Initialize ensemble counting functions on first iteration
        if r_values is None and element_pairs[0] in struct_cfs:
            r_values = struct_cfs[element_pairs[0]][0]
            for pair in element_pairs:
                if pair in struct_cfs:
                    ensemble_countings[pair] = np.zeros_like(struct_cfs[pair][1])
        
        # Process each element pair
        for pair in element_pairs:
            if pair not in struct_cfs:
                continue
            
            r, counting_func = struct_cfs[pair]
            
            # Calculate density normalization factor for this pair
            elem1, elem2 = pair
            struct_density1 = density_data['partial_densities'].get(elem1, 0)
            struct_density2 = density_data['partial_densities'].get(elem2, 0)
            avg_density1 = avg_densities['partial_densities'].get(elem1, 0)
            avg_density2 = avg_densities['partial_densities'].get(elem2, 0)
            
            # For counting functions, we need to scale by the density ratio
            if elem1 == elem2:
                if avg_density1 > 0:
                    density_factor = struct_density1 / avg_density1
                else:
                    density_factor = 1.0
            else:
                # Use total density for mixed pairs
                struct_total = density_data['total_density'] 
                avg_total = avg_densities['total_density']
                if avg_total > 0:
                    density_factor = struct_total / avg_total
                else:
                    density_factor = 1.0
            
            # Apply density normalization to counting function
            normalized_counting = counting_func * density_factor
            
            # Add weighted contribution to ensemble counting function
            if ensemble_countings[pair] is not None:
                ensemble_countings[pair] += weight * normalized_counting
    
    # Create results dictionary
    results = {}
    for pair in element_pairs:
        if ensemble_countings[pair] is not None:
            results[pair] = (r_values, ensemble_countings[pair])
    
    return results
    
# =====================================================================
# PLOTTING FUNCTIONS
# =====================================================================

def plot_single_counting_function(struct_ids=None, counting_functions=None, energies=None, use_weights=False,
                                 temperature=1800, figsize=(8, 10), label=None, 
                                 save_path=None, show_plot=True):
    """
    Plot counting functions for structures specified by IDs
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs to use
    counting_functions : dict, optional
        Pre-loaded counting function data
    energies : dict, optional
        Pre-loaded energy data
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
    
    # Calculate ensemble partial counting functions
    print("Calculating ensemble partial counting functions...")
    ensemble_countings = calculate_ensemble_partial_counting_functions(
        struct_ids=struct_ids,
        counting_functions=counting_functions,
        energies=energies, 
        use_weights=use_weights,
        temperature=temperature
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

def plot_multiple_counting_functions(*data_items, labels=None, temperature=1800, use_weights=False, figsize=(8, 10), save_path=None):
    """
    Plot counting functions from multiple sets of structure IDs or pre-calculated CFs.
    
    Parameters:
    -----------
    *data_items : variable number of items, each can be:
        - numpy.ndarray or list of structure IDs  
        - dict containing pre-calculated ensemble counting functions {pair: (r_values, cf)}
    labels : list of str, optional
        Labels for each dataset in the legend
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting (for ID-based calculations)
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the combined figure
        
    Returns:
    --------
    list : List of ensemble CF dictionaries for each dataset
    """
    
    if len(data_items) == 0:
        raise ValueError("At least one data item must be provided")
    
    # Store all CF results
    all_cf_results = []
    all_cfs = {}  # For plotting: {pair: [(r, cf, label), ...]}
    
    # Process each data item
    for i, item in enumerate(data_items):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"
        
        if isinstance(item, (list, np.ndarray)):
            # Calculate ensemble CFs from structure IDs
            ensemble_cfs = calculate_ensemble_partial_counting_functions(
                struct_ids=item, temperature=temperature, use_weights=use_weights
            )
            
        elif isinstance(item, dict) and all(isinstance(k, tuple) and len(k) == 2 for k in item.keys()):
            # Pre-calculated ensemble CFs
            ensemble_cfs = item
            
        else:
            print(f"Warning: Unrecognized data type at index {i}, skipping")
            continue
        
        # Store results
        all_cf_results.append(ensemble_cfs)
        
        # Organize for plotting
        for pair, (r, cf) in ensemble_cfs.items():
            if pair not in all_cfs:
                all_cfs[pair] = []
            all_cfs[pair].append((r, cf, label))
    
    # Create plots
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
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return all_cf_results

def plot_single_rdf(struct_ids=None, rdfs=None, energies=None, use_weights=False, temperature=1800, 
                    smoothed=True, figsize=(8, 10), label=None, save_path=None, show_plot=True):
    """
    Plot RDF for structures specified by IDs
    
    Parameters:
    -----------
    struct_ids : list of int, optional
        Structure IDs to use for RDF calculation
    rdfs : dict, optional
        Pre-loaded RDF data
    energies : dict, optional
        Pre-loaded energy data
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting
    smoothed : bool, optional
        If True, use smoothed RDFs. If False, use original RDFs (default: False)
    figsize : tuple, optional
        Figure size (width, height) in inches
    label : str, optional
        Label for the plot legend
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
    show_plot : bool, optional
        Whether to display the plot
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, g_r) values
    """
    
    # Calculate ensemble partial RDFs
    print("Calculating ensemble partial RDFs...")
    ensemble_rdfs = calculate_ensemble_partial_rdfs(
        struct_ids=struct_ids, 
        rdfs=rdfs,
        energies=energies,
        use_weights=use_weights,
        temperature=temperature,
        smoothed=smoothed
    )

    if show_plot:
        # Plot the partial RDFs
        fig = plt.figure(figsize=figsize)
        
        # Add title suffix based on smoothing
        title_suffix = " (KAMEL-LOBE Smoothed)" if smoothed else ""
        
        # Iterate through the partial RDFs
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
            print(f"Figure saved to {save_path}")
        
        if show_plot:
            plt.show()

    return ensemble_rdfs

def plot_multiple_rdfs(*data_items, labels=None, temperature=1800, use_weights=False, smoothed=True, 
                       figsize=(8, 10), save_path=None):
    """
    Plot RDFs from multiple sets of structure IDs or pre-calculated RDFs.
    
    Parameters:
    -----------
    *data_items : variable number of items, each can be:
        - numpy.ndarray or list of structure IDs
        - dict containing pre-calculated ensemble RDFs {pair: (r_values, g_r)}
    labels : list of str, optional
        Labels for each dataset in the legend
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting (for ID-based calculations)
    smoothed : bool, optional
        If True, use smoothed RDFs. If False, use original RDFs (default: False)
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the combined figure
        
    Returns:
    --------
    list : List of ensemble RDF dictionaries for each dataset
    """
    
    if len(data_items) == 0:
        raise ValueError("At least one data item must be provided")
    
    # Store all RDF results
    all_rdf_results = []
    all_rdfs = {}  # For plotting: {pair: [(r, rdf, label), ...]}
    
    # Process each data item
    for i, item in enumerate(data_items):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"
        
        if isinstance(item, (list, np.ndarray)):
            # Calculate ensemble RDFs from structure IDs
            ensemble_rdfs = calculate_ensemble_partial_rdfs(
                struct_ids=item, 
                temperature=temperature,
                use_weights=use_weights,
                smoothed=smoothed
            )
            
        elif isinstance(item, dict) and all(isinstance(k, tuple) and len(k) == 2 for k in item.keys()):
            # Pre-calculated ensemble RDFs
            ensemble_rdfs = item
            
        else:
            print(f"Warning: Unrecognized data type at index {i}, skipping")
            continue
        
        # Store results
        all_rdf_results.append(ensemble_rdfs)
        
        # Organize for plotting
        for pair, (r, rdf) in ensemble_rdfs.items():
            if pair not in all_rdfs:
                all_rdfs[pair] = []
            all_rdfs[pair].append((r, rdf, label))
    
    # Create plots
    if not all_rdfs:
        print("Warning: No valid RDF data to plot")
        return all_rdf_results
    
    # Add title suffix based on smoothing
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
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return all_rdf_results

# =====================================================================
# CONVERSION FUNCTIONS BETWEEN RDF AND COUNTING FUNCTIONS
# =====================================================================

def convert_rdf_to_counting_function(ensemble_rdfs, ensemble_densities=None):
    """
    Convert ensemble RDF dictionary to counting function dictionary with proper density.
    
    Parameters:
    -----------
    ensemble_rdfs : dict
        Dictionary with element pairs as keys and (r_values, g_r) as values
    ensemble_densities : dict, optional
        Ensemble average densities from calculate_ensemble_average_density()
        If None, will use default density
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, counting_function) as values
    """
    ensemble_countings = {}
    
    for pair, (r_values, g_r) in ensemble_rdfs.items():
        # Calculate counting function from RDF
        dr = r_values[1] - r_values[0] if len(r_values) > 1 else 0.05
        
        # Use ensemble average density if provided
        if ensemble_densities is not None:
            elem1, elem2 = pair
            if elem1 == elem2:
                # Same element pair - use single density
                rho = ensemble_densities['partial_densities'].get(elem1, 0.066)
            else:
                # Different element pair - use geometric mean
                rho1 = ensemble_densities['partial_densities'].get(elem1, 0.066)
                rho2 = ensemble_densities['partial_densities'].get(elem2, 0.066)
                rho = np.sqrt(rho1 * rho2)
        else:
            # Use default density for SiO2
            rho = 0.066
        
        # Convert g(r) to pair correlation: 4πr²ρ[g(r)-1] + 4πr²ρ
        pair_correlation = 4 * np.pi * r_values**2 * rho * (g_r - 1)
        
        # Integrate to get counting function
        counting_function = np.cumsum(pair_correlation) * dr
        
        # Add the uniform background contribution
        uniform_background = (4/3) * np.pi * r_values**3 * rho
        counting_function = counting_function + uniform_background
        
        ensemble_countings[pair] = (r_values, counting_function)
    
    return ensemble_countings

def convert_counting_function_to_rdf(ensemble_countings, ensemble_densities=None):
    """
    Convert ensemble counting function dictionary back to RDF dictionary with proper density.
    
    Parameters:
    -----------
    ensemble_countings : dict
        Dictionary with element pairs as keys and (r_values, counting_function) as values
    ensemble_densities : dict, optional
        Ensemble average densities from calculate_ensemble_average_density()
        If None, will estimate from the data
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, g_r) as values
    """
    ensemble_rdfs = {}
    
    for pair, (r_values, counting_function) in ensemble_countings.items():
        # Calculate RDF from counting function
        dr = r_values[1] - r_values[0] if len(r_values) > 1 else 0.05
        
        # Use ensemble average density if provided
        if ensemble_densities is not None:
            elem1, elem2 = pair
            if elem1 == elem2:
                rho = ensemble_densities['partial_densities'].get(elem1, 0.066)
            else:
                rho1 = ensemble_densities['partial_densities'].get(elem1, 0.066)
                rho2 = ensemble_densities['partial_densities'].get(elem2, 0.066)
                rho = np.sqrt(rho1 * rho2)
        else:
            # Estimate density from asymptotic behavior of counting function
            if len(counting_function) > 10:
                r_large = r_values[-5:]  # Use last few points
                N_large = counting_function[-5:]
                rho_estimates = 3 * N_large / (4 * np.pi * r_large**3)
                rho = np.mean(rho_estimates[rho_estimates > 0])
            else:
                rho = 0.066  # Default for SiO2
        
        # Take derivative to get pair correlation
        pair_correlation = np.gradient(counting_function, dr)
        
        # Remove uniform background: dN/dr = 4πr²ρ for uniform distribution
        uniform_background_derivative = 4 * np.pi * r_values**2 * rho
        pair_correlation_excess = pair_correlation - uniform_background_derivative
        
        # Convert back to g(r): g(r) = 1 + excess/(4πr²ρ)
        # Avoid division by zero at r=0
        r_safe = np.where(r_values == 0, np.finfo(float).eps, r_values)
        g_r = 1 + pair_correlation_excess / (4 * np.pi * r_safe**2 * rho)
        
        # Set first few points to 0 to avoid artifacts at r=0
        g_r[:3] = 0.0
        ensemble_rdfs[pair] = (r_values, g_r)
    
    return ensemble_rdfs


