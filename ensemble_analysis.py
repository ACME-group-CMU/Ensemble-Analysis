import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
from scipy.spatial import cKDTree
import os
import glob
from tqdm import tqdm
import re
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList
import ase.geometry
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.build import make_supercell
import warnings

# =====================================================================
# POSCAR Handling and Basic Structure Functions
# =====================================================================

def read_poscar_with_energy(filename):
    """
    Read a POSCAR file with energy information in the first line.
    """
    # Read the first line to extract energy
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
    
    # Extract energy using regex
    match = re.search(r'Energy\s*=\s*([-\d.]+)', first_line)
    if match:
        energy = float(match.group(1))
    else:
        energy = None
        warnings.warn(f"Energy not found in first line of {filename}")
    
    # Read the POSCAR file using ASE
    atoms = read(filename, format='vasp')
    
    return atoms, energy

# def create_supercell(atoms, nx, ny, nz):
    """
    Create a supercell by replicating the structure.
    nx, ny, nz are the number of repetitions in each direction.
    """
    # Create the supercell transformation matrix
    P = np.array([[nx, 0, 0], [0, ny, 0], [0, 0, nz]])
    
    # Use ASE's make_supercell function
    supercell = make_supercell(atoms, P)
    
    return supercell

# =====================================================================
# RDF Calculation and Analysis
# =====================================================================

def calculate_rdf(atoms, r_range, bins, element_pairs=None, periodic=True):
    """
    Calculate radial distribution function with proper normalization.
    
    Input :
    atoms : ase.Atoms object
    r_range : tuple of floats
    bins : int
    element_pairs : list of tuples, optional e.g., [('Si', 'O')])
    periodic : bool
        
    Output:
    r_values : numpy.ndarray
    g_r : numpy.ndarray
    """
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

def calculate_partial_rdfs(atoms, r_range, bins, element_pairs=None, periodic=True):
    """
    Calculate partial radial distribution functions for specific element pairs.
    
    Input
    atoms : ase.Atoms object
    r_range : tuple of floats
    bins : int
    element_pairs : list of tuples, optional e.g., [('Si', 'O')])
    periodic : bool
        
    Output:
    dict : Dictionary with element pairs as keys and (r_values, g_r) as values
    """
    if element_pairs is None:
        # Get unique element types and create all possible pairs
        elements = np.unique(atoms.get_chemical_symbols())
        element_pairs = []
        for i, elem1 in enumerate(elements):
            for elem2 in elements[i:]:  # Start from i to avoid duplicates
                element_pairs.append((elem1, elem2))
    
    # Calculate RDF for each element pair
    results = {}
    for pair in element_pairs:
        r_values, g_r = calculate_rdf(atoms, r_range, bins, [pair], periodic)
        results[pair] = (r_values, g_r)
    
    return results

# =====================================================================
# Ensemble Extension
# =====================================================================

def calculate_weights(energies, temperature):
    """
    Calculate Boltzmann weights based on energies.
    
    Input:
    energies : dict : Dictionary mapping filenames to energies
    temperature : float : Temperature in Kelvin
        
    Output:
    dict : Dictionary mapping filenames to weights
    """
    # Convert temperature to Ry (1 Ry = 13.605693009 eV)
    kB_Ry = 6.33362e-6  # Ry/K
    kT = kB_Ry * temperature
    
    # Find the minimum energy
    min_energy = min(energies.values())
    
    # Calculate unnormalized weights
    weights = {}
    for filename, energy in energies.items():
        rel_energy = energy - min_energy
        weights[filename] = np.exp(-rel_energy / (24*kT))
    
    # Normalize weights
    total_weight = sum(weights.values())
    for filename in weights:
        weights[filename] /= total_weight
    
    return weights

# Calculate ensemble RDF
def calculate_ensemble_rdf(file_list, weights, r_range=(0, 10), bins=200):
    ensemble_rdf = None
    r_values = None
    
    for file_path in tqdm(file_list, desc="Calculating ensemble RDF"):
        base_filename = os.path.basename(file_path)
        
        if base_filename not in weights:
            print(f"Warning: No weight found for {base_filename}, skipping")
            continue
        
        weight = weights[base_filename]
        
        try:
            # Read the structure
            atoms, _ = read_poscar_with_energy(file_path)
            
            # Calculate the RDF
            r, rdf = calculate_rdf(atoms, r_range, bins)
            
            # Store r values from first calculation
            if r_values is None:
                r_values = r
                ensemble_rdf = np.zeros_like(rdf)
            
            # Add weighted contribution to ensemble RDF
            ensemble_rdf += weight * rdf
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return r_values, ensemble_rdf

def calculate_ensemble_rdf2(file_list, weights, r_range=(0, 10), bins=200):
    """
    Calculate ensemble-averaged radial distribution function with proper density normalization.
    
    This implementation follows Equation 4 from the paper:
    g^(2)(r) = ∑ₐ (nₐ/⟨n⟩)² Pₐ g^(2)ₐ(r)
    
    Parameters:
    -----------
    file_list : list
        List of POSCAR file paths
    weights : dict
        Dictionary mapping base filenames to Boltzmann weights (Pₐ)
    r_range : tuple, optional
        (rmin, rmax) range for the RDF
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
    
    for file_path in tqdm(file_list, desc="Calculating average density"):
        base_filename = os.path.basename(file_path)
        
        if base_filename not in weights:
            print(f"Warning: No weight found for {base_filename}, skipping")
            continue
        
        weight = weights[base_filename]
        
        try:
            # Read the structure
            atoms, _ = read_poscar_with_energy(file_path)
            
            # Calculate number density (atoms per volume)
            n_atoms = len(atoms)
            volume = atoms.get_volume()
            number_density = n_atoms / volume
            
            # Add to weighted average
            total_density += weight * number_density
            total_weight += weight
            
        except Exception as e:
            print(f"Error processing {file_path} (density calculation): {e}")
    
    # Calculate average number density
    if total_weight > 0:
        avg_number_density = total_density / total_weight
    else:
        print("Warning: Total weight is zero, cannot calculate average density")
        return None, None
    
    # Second pass: calculate ensemble RDF with density normalization
    for file_path in tqdm(file_list, desc="Calculating ensemble RDF"):
        base_filename = os.path.basename(file_path)
        
        if base_filename not in weights:
            continue  # Already warned in first pass
        
        weight = weights[base_filename]
        
        try:
            # Read the structure
            atoms, _ = read_poscar_with_energy(file_path)
            
            # Calculate number density (atoms per volume)
            n_atoms = len(atoms)
            volume = atoms.get_volume()
            number_density = n_atoms / volume
            
            # Calculate the density normalization factor (nₐ/⟨n⟩)²
            density_factor = (number_density / avg_number_density) ** 2
            
            # Calculate the RDF
            r, rdf = calculate_rdf(atoms, r_range, bins)
            
            # Store r values from first calculation
            if r_values is None:
                r_values = r
                ensemble_rdf = np.zeros_like(rdf)
            
            # Add weighted contribution to ensemble RDF with density normalization
            ensemble_rdf += weight * density_factor * rdf
            
        except Exception as e:
            print(f"Error processing {file_path} (RDF calculation): {e}")
    
    return r_values, ensemble_rdf

def calculate_ensemble_partial_rdfs(file_list, weights, r_range, bins, element_pairs):
    """
    Calculate ensemble average RDFs for specific element pairs.
    
    Parameters:
    -----------
    file_list : list
        List of POSCAR files
    weights : dict
        Dictionary mapping base filenames to weights
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
    # Initialize dictionaries to store ensemble RDFs
    ensemble_rdfs = {pair: None for pair in element_pairs}
    r_values = None
    
    for file_path in tqdm(file_list, desc="Calculating ensemble partial RDFs"):
        base_filename = os.path.basename(file_path)
        
        if base_filename not in weights:
            print(f"Warning: No weight found for {base_filename}, skipping")
            continue
        
        weight = weights[base_filename]
        
        try:
            # Read the structure
            atoms, _ = read_poscar_with_energy(file_path)
            
            # Calculate RDFs for each element pair
            partial_rdfs = calculate_partial_rdfs(atoms, r_range, bins, element_pairs)
            
            # Initialize ensemble RDFs on first iteration
            if r_values is None:
                r_values = partial_rdfs[element_pairs[0]][0]
                for pair in element_pairs:
                    ensemble_rdfs[pair] = np.zeros_like(partial_rdfs[pair][1])
            
            # Add weighted contributions to ensemble RDFs
            for pair in element_pairs:
                ensemble_rdfs[pair] += weight * partial_rdfs[pair][1]
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create results dictionary with (r_values, g_r) tuples
    results = {pair: (r_values, ensemble_rdfs[pair]) for pair in element_pairs}
    
    return results

def calculate_ensemble_partial_rdfs2(file_list, weights, r_range, bins, element_pairs):
    """
    Calculate ensemble average RDFs for specific element pairs with density normalization.
    
    This implementation follows Equation 4 from the paper:
    g^(2)(r) = ∑ₐ (nₐ/⟨n⟩)² Pₐ g^(2)ₐ(r)
    
    For each element pair, the proper density normalization is applied.
    
    Parameters:
    -----------
    file_list : list
        List of POSCAR files
    weights : dict
        Dictionary mapping base filenames to weights
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
    
    for file_path in tqdm(file_list, desc="Calculating average density"):
        base_filename = os.path.basename(file_path)
        
        if base_filename not in weights:
            print(f"Warning: No weight found for {base_filename}, skipping")
            continue
        
        weight = weights[base_filename]
        
        try:
            # Read the structure
            atoms, _ = read_poscar_with_energy(file_path)
            
            # Calculate number density (atoms per volume)
            n_atoms = len(atoms)
            volume = atoms.get_volume()
            number_density = n_atoms / volume
            
            # Add to weighted average
            total_density += weight * number_density
            total_weight += weight
            
        except Exception as e:
            print(f"Error processing {file_path} (density calculation): {e}")
    
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
    for file_path in tqdm(file_list, desc="Calculating ensemble partial RDFs"):
        base_filename = os.path.basename(file_path)
        
        if base_filename not in weights:
            continue  # Already warned in first pass
        
        weight = weights[base_filename]
        
        try:
            # Read the structure
            atoms, _ = read_poscar_with_energy(file_path)
            
            # Calculate number density (atoms per volume)
            n_atoms = len(atoms)
            volume = atoms.get_volume()
            number_density = n_atoms / volume
            
            # Calculate the density normalization factor (nₐ/⟨n⟩)²
            density_factor = (number_density / avg_number_density) ** 2
            
            # Calculate RDFs for each element pair
            partial_rdfs = calculate_partial_rdfs(atoms, r_range, bins, element_pairs)
            
            # Initialize ensemble RDFs on first iteration
            if r_values is None:
                r_values = partial_rdfs[element_pairs[0]][0]
                for pair in element_pairs:
                    ensemble_rdfs[pair] = np.zeros_like(partial_rdfs[pair][1])
            
            # Add weighted contributions to ensemble RDFs with density normalization
            for pair in element_pairs:
                ensemble_rdfs[pair] += weight * density_factor * partial_rdfs[pair][1]
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create results dictionary with (r_values, g_r) tuples
    results = {pair: (r_values, ensemble_rdfs[pair]) for pair in element_pairs}
    
    return results

def analyze_rdf_convergence(file_list, r_range, bins, sample_sizes=[50, 100, 200, 400], num_samples=20):
    """
    Analyze how RDF converges with increasing sample size.
    
    Parameters:
    -----------
    file_list : list
        List of POSCAR files
    r_range : tuple
        (rmin, rmax) range for the RDF
    bins : int
        Number of bins for the histogram
    sample_sizes : list
        List of sample sizes to test
    num_samples : int
        Number of random samples to generate for each size
        
    Returns:
    --------
    dict : Dictionary with sample sizes as keys and RDF RMSE values as values
    """
    import random
    
    # Calculate the reference RDF using all structures
    all_atoms = []
    for file_path in tqdm(file_list, desc="Reading all structures"):
        try:
            atoms, _ = read_poscar_with_energy(file_path)
            all_atoms.append(atoms)
        except Exception as e:
            warnings.warn(f"Error processing {file_path}: {e}")
    
    # Equal weights for all structures
    weights = {i: 1.0/len(all_atoms) for i in range(len(all_atoms))}
    
    # Calculate reference ensemble RDF (all structures)
    r_ref, rdf_ref = None, None
    
    for i, atoms in enumerate(all_atoms):
        r, rdf = calculate_rdf(atoms, r_range, bins)
        
        if r_ref is None:
            r_ref = r
            rdf_ref = np.zeros_like(rdf)
        
        rdf_ref += weights[i] * rdf
    
    # Test different sample sizes
    results = {}
    
    for size in tqdm(sample_sizes, desc="Testing sample sizes"):
        rmse_values = []
        
        for _ in range(num_samples):
            # Randomly select structures
            if size <= len(all_atoms):
                sample = random.sample(all_atoms, size)
            else:
                sample = all_atoms  # Use all if requested size is larger
            
            # Equal weights for the sample
            sample_weights = {i: 1.0/len(sample) for i in range(len(sample))}
            
            # Calculate ensemble RDF for this sample
            r_sample, rdf_sample = None, None
            
            for i, atoms in enumerate(sample):
                r, rdf = calculate_rdf(atoms, r_range, bins)
                
                if r_sample is None:
                    r_sample = r
                    rdf_sample = np.zeros_like(rdf)
                
                rdf_sample += sample_weights[i] * rdf
            
            # Calculate RMSE between sample and reference
            rmse = np.sqrt(np.mean((rdf_sample - rdf_ref)**2))
            rmse_values.append(rmse)
        
        # Store the average RMSE for this sample size
        results[size] = np.mean(rmse_values)
    
    return results



    """
    Analyze ring statistics in the network.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        Atomic structure
    max_ring_size : int
        Maximum ring size to consider
    cutoff_factor : float
        Factor to multiply covalent radii for determining bonds
        
    Returns:
    --------
    dict : Dictionary mapping ring sizes to their counts
    """
    try:
        import networkx as nx
    except ImportError:
        warnings.warn("NetworkX is required for ring analysis. Please install with: pip install networkx")
        return {size: 0 for size in range(3, max_ring_size + 1)}
    
    # This implementation focuses on Si-O rings in silicates
    # Get atomic positions and symbols
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Create a graph representation
    G = nx.Graph()
    
    # Add nodes with element type as attribute
    for i, symbol in enumerate(symbols):
        G.add_node(i, element=symbol)
    
    # Get covalent radii for all atoms
    atomic_numbers = atoms.get_atomic_numbers()
    radii = np.array([covalent_radii[z] for z in atomic_numbers])
    
    # Determine connectivity based on distance criteria
    nl = NeighborList(cutoff_factor * radii, self_interaction=False, bothways=False)
    nl.update(atoms)
    
    # Add edges to the graph
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        for j in indices:
            if i < j:  # Add each edge only once
                G.add_edge(i, j)
    
    # For silicates, we're interested in rings where Si and O alternate
    # Find all cycles in the graph
    ring_counts = {size: 0 for size in range(3, max_ring_size + 1)}
    
    # Focus on specific ring types for silicates
    si_indices = [i for i, s in enumerate(symbols) if s == 'Si']
    o_indices = [i for i, s in enumerate(symbols) if s == 'O']
    
    # Use NetworkX's cycle finding algorithms
    # This is a simplified approach - for production, consider specialized algorithms
    for size in range(3, max_ring_size + 1, 1):  # For even-sized rings in silicates (Si-O-Si-O...)
        for si_idx in si_indices:
            # Find cycles of specified length starting from this Si atom
            for path in nx.simple_cycles(G, size):
                if len(path) == size and si_idx in path:
                    # Check if it's a proper alternating Si-O ring
                    is_valid = True
                    for i in range(len(path)):
                        node = path[i]
                        next_node = path[(i + 1) % len(path)]
                        
                        node_elem = G.nodes[node]['element']
                        next_elem = G.nodes[next_node]['element']
                        
                        if (node_elem == 'Si' and next_elem != 'O') or (node_elem == 'O' and next_elem != 'Si'):
                            is_valid = False
                            break
                    
                    if is_valid:
                        ring_counts[size] += 1
    
    # Normalize counts to avoid overcounting
    for size in ring_counts:
        ring_counts[size] //= size
    
    return ring_counts