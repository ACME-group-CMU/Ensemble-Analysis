import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
import os
import glob
from tqdm import tqdm
import re
import warnings
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import NeighborList

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
# Ensemble Extension
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

def plot_multiple_rdfs(structure_dicts_array, labels = None, r_range=(0, 10), bins=100, temperature=1800, 
                        figsize=(8, 10), save_path=None):
    """
    Plot RDFs for multiple structure dictionaries with automatic weight calculation.
    
    Parameters:
    -----------
    structure_dicts_array : list
        List of dictionaries, each mapping structure IDs to structure-energy pairs
    r_range : tuple, optional
        (rmin, rmax) range for the RDF in Angstroms
    bins : int, optional
        Number of bins for the histogram
    temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting
    plot_all : bool, optional
        If True, plot all RDFs on the same figure; if False, plot each RDF separately
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, g_r) values
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
    
    # Initialize dictionary to store all RDFs
    all_rdfs = {}
    
    # Process each structure dictionary
    for i, structures_dict in enumerate(tqdm(structure_dicts_array, desc="Processing structure dictionaries")):
        # Get all unique elements in the structures
        all_elements = set()
        for struct_id, data in structures_dict.items():
            structure = data['structure']
            all_elements.update([str(site.specie.symbol) for site in structure])
        elements = ['Si', 'O']  # Specify the elements of interest
        print(f"Elements in structures: {elements}")

        # Set up element pairs for partial RDFs
        element_pairs = []
        for i, elem1 in enumerate(elements):
            for j in range(i, len(elements)):
                elem2 = elements[j]
                element_pairs.append((elem1, elem2))
        
        # Calculate Boltzmann weights
        weights = calculate_weights(structures_dict, temperature)
        
        # Calculate ensemble partial RDFs
        ensemble_rdfs = calculate_ensemble_partial_rdfs(
            structures_dict, weights, r_range, bins, element_pairs
        )
        
        # Store in the all_rdfs dictionary
        for pair, (r, rdf) in ensemble_rdfs.items():
            if pair not in all_rdfs:
                all_rdfs[pair] = []
            all_rdfs[pair].append((r, rdf, i))  # Store r, rdf, and the dictionary index
    
    # Create plots
    # Plot all RDFs for each element pair on the same figure
    n_pairs = len(all_rdfs)
    n_rows = min(3, n_pairs)
    n_cols = (n_pairs + n_rows - 1) // n_rows
    
    plt.figure(figsize=figsize)
    
    for i, (pair, rdf_list) in enumerate(all_rdfs.items()):
        plt.subplot(n_rows, n_cols, i+1)
        
        for r, rdf, dict_idx in rdf_list:
            if labels:
                plt.plot(r, rdf)
            else:
                plt.plot(r, rdf, label=f'Dict {dict_idx}')
        
        plt.title(f'{pair[0]}-{pair[1]} Partial RDF')
        plt.xlabel('Distance (Å)')
        plt.ylabel('g(r)')
        plt.grid(True, alpha=0.3)
        if labels:
            plt.legend(labels)
        else:
            plt.legend()
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
    
    return all_rdfs

def plot_single_rdf(structure_dict, r_range=(0, 10), bins=100, temperature=1800, 
                    figsize=(8, 10)):
    """
    Plot RDF for a single structure dictionary with automatic weight calculation.
    
    Parameters:
    -----------
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
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
        
    Returns:
    --------
    dict : Dictionary with element pairs as keys and (r_values, g_r) values
    """
    # Get all unique elements in the structures
    all_elements = set()
    for struct_id, data in structure_dict.items():
        structure = data['structure']
        all_elements.update([str(site.specie.symbol) for site in structure])
    elements = ['Si', 'O']  # Specify the elements of interest
    print(f"Elements in structures: {elements}")

    # Set up element pairs for partial RDFs
    element_pairs = []
    for i, elem1 in enumerate(elements):
        for j in range(i, len(elements)):
            elem2 = elements[j]
            element_pairs.append((elem1, elem2))


    # Calculate Boltzmann weights at a specified temperature
    temperature = temperature  # Kelvin
    weights = calculate_weights(structure_dict, temperature)
    print(f"Calculated Boltzmann weights at {temperature} K")

    # Parameters for RDF calculation
    r_range = r_range  # Distance range in Angstroms
    bins = bins  # Number of bins

    # Calculate ensemble partial RDFs
    print("Calculating ensemble partial RDFs...")
    ensemble_rdfs = calculate_ensemble_partial_rdfs(
        structure_dict, 
        weights, 
        r_range, 
        bins, 
        element_pairs
    )

    # Plot the partial RDFs
    n_pairs = len(element_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    plt.figure(figsize=figsize)

    # Iterate through the partial RDFs
    for i, ((elem1, elem2), (r, rdf)) in enumerate(ensemble_rdfs.items(), 1):
        plt.subplot(3, 1, i)
        plt.plot(r, rdf)
        plt.title(f'{elem1}-{elem2} Partial RDF')
        plt.xlabel('Distance (Å)')
        plt.ylabel('g(r)')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    return ensemble_rdfs
