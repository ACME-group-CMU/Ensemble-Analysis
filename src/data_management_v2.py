import os
import numpy as np
import pickle
import re
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from collections import OrderedDict
from tqdm import tqdm
from src.rings import calculate_ensemble_ring_statistics, calculate_ring_statistics_for_structure

BASE_DATA_DIR = 'data'
POSCAR_DIR = f'{BASE_DATA_DIR}/all_sizes'
ENERGY_DIR = f'{BASE_DATA_DIR}/energies'
RDF_DIR = f'{BASE_DATA_DIR}/rdfs'
CF_DIR = f'{BASE_DATA_DIR}/counting_functions'
STRUCT_DIR = f'{BASE_DATA_DIR}/structures'
SMOOTH_RDF_DIR = f'{BASE_DATA_DIR}/smooth_rdfs'
QN_DIR = f'{BASE_DATA_DIR}/qn_distributions'
BAD_DIR = f'{BASE_DATA_DIR}/bond_angle_distributions'
RING_DIR = f'{BASE_DATA_DIR}/ring_statistics'

# Standard parameters
STANDARD_PARAMS = {
    'r_range': (0, 10),
    'bins': 200,
    'temperature': 1800,
    'element_pairs': [('Si', 'Si'), ('Si', 'O'), ('O', 'O')]
}

def ensure_directories():
    """Create data directories if they don't exist"""
    for directory in [ENERGY_DIR, RDF_DIR, CF_DIR, STRUCT_DIR, QN_DIR, BAD_DIR, RING_DIR]:
        os.makedirs(directory, exist_ok=True)

def vasp_to_pymatgen(struct_id, folder_path=POSCAR_DIR):
    """
    Read a VASP POSCAR file and convert it to a pymatgen Structure object.
    Also extracts energy from the first line if available.

    Parameters:
    -----------
    struct_id : str
        Structure ID (e.g., 'SiO2_36_481' or 'SiO2_24_2')
    folder_path : str
        Path to the folder containing VASP files

    Returns:
    --------
    tuple : (pymatgen_structure, energy)
    """
    filepath = os.path.join(folder_path, f"{struct_id}.vasp")
    
    # Extract energy from the first line
    energy = None
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        energy_match = re.search(r'Energy\s*=\s*(-?\d+\.\d+)', first_line)
        if energy_match:
            energy = float(energy_match.group(1))
    
    # Read the VASP file using ASE
    atoms = read(filepath, format='vasp')
    
    # Convert ASE Atoms to pymatgen Structure
    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(atoms)
    
    return structure, energy


# =====================================================================
# BOLTZMANN WEIGHTING
# =====================================================================

def calculate_weights(struct_ids, energies, structures, temperature):
    """
    Calculate Boltzmann weights for an ensemble of structures.

    Energies are normalized per atom before computing the Boltzmann factor,
    ensuring structures of different sizes are weighted correctly.

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to calculate weights for
    energies : dict
        {struct_id: energy (Ry)}
    structures : dict
        {struct_id: pymatgen Structure} — used to get n_atoms per structure
    temperature : float
        Temperature in Kelvin

    Returns:
    --------
    dict : {struct_id: normalized weight}
    """
    kB_Ry = 6.33362e-6  # Boltzmann constant in Ry/K
    kT = kB_Ry * temperature

    # Compute per-atom energies
    per_atom_energies = {}
    for struct_id in struct_ids:
        if struct_id in energies and struct_id in structures:
            n_atoms = len(structures[struct_id])
            per_atom_energies[struct_id] = energies[struct_id] / n_atoms

    if not per_atom_energies:
        raise ValueError("No valid energy/structure pairs found for weight calculation.")

    min_energy = min(per_atom_energies.values())

    # Unnormalized Boltzmann weights
    weights = {
        struct_id: np.exp(-(e - min_energy) / kT)
        for struct_id, e in per_atom_energies.items()
    }

    # Normalize
    total = sum(weights.values())
    return {struct_id: w / total for struct_id, w in weights.items()}


# =====================================================================
# POPULATION FUNCTIONS (Create and save data)
# =====================================================================

def populate_densities(struct_ids, folder_path=POSCAR_DIR):
    """
    Calculate and save number densities for each structure

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to process (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    folder_path : str
        Path to POSCAR files
    """
    ensure_directories()
    
    # Create densities directory if it doesn't exist
    densities_dir = os.path.join(BASE_DATA_DIR, 'densities')
    os.makedirs(densities_dir, exist_ok=True)
    
    for struct_id in tqdm(struct_ids, desc="Calculating densities"):
        try:
            structure, energy = vasp_to_pymatgen(struct_id, folder_path)
            
            # Calculate number density (atoms per Angstrom^3)
            volume = structure.volume  # in Angstrom^3
            num_atoms = len(structure)
            number_density = num_atoms / volume
            
            # Calculate partial densities for each element
            element_counts = {}
            for site in structure:
                elem = site.specie.symbol
                element_counts[elem] = element_counts.get(elem, 0) + 1
            
            partial_densities = {}
            for elem, count in element_counts.items():
                partial_densities[elem] = count / volume
            
            density_data = {
                'total_density': number_density,
                'partial_densities': partial_densities,
                'volume': volume,
                'num_atoms': num_atoms,
                'element_counts': element_counts
            }
            
            filepath = os.path.join(densities_dir, f'{struct_id}_density.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(density_data, f)
                
        except Exception as e:
            print(f"Error processing density for structure {struct_id}: {e}")

def populate_qn_distributions(struct_ids, folder_path=POSCAR_DIR):
    """
    Calculate and save Qn distributions for each structure
    Qn = number of bridging oxygens per Si atom (n = 0,1,2,3,4)

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to process (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    folder_path : str
        Path to POSCAR files
    """
    ensure_directories()
    
    for struct_id in tqdm(struct_ids, desc="Calculating Qn distributions"):
        try:
            structure, energy = vasp_to_pymatgen(struct_id, folder_path)
            
            # Find Si and O atoms
            si_indices = [i for i, site in enumerate(structure) if site.specie.symbol == 'Si']
            o_indices = [i for i, site in enumerate(structure) if site.specie.symbol == 'O']
            
            # Count bridging oxygens for each Si
            qn_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            
            # Distance threshold for Si-O bonds (typical: 1.8 Angstroms)
            bond_threshold = 1.8
            
            for si_idx in si_indices:
                si_site = structure[si_idx]
                num_bridging_o = 0
                
                # Find O atoms bonded to this Si
                bonded_o = []
                for o_idx in o_indices:
                    dist = structure.get_distance(si_idx, o_idx)
                    if dist < bond_threshold:
                        bonded_o.append(o_idx)
                
                # For each bonded O, check if it's bridging (bonded to 2 Si)
                for o_idx in bonded_o:
                    si_neighbors = 0
                    for other_si in si_indices:
                        if other_si != si_idx:
                            dist = structure.get_distance(o_idx, other_si)
                            if dist < bond_threshold:
                                si_neighbors += 1
                    
                    # If O is bonded to another Si, it's bridging
                    if si_neighbors >= 1:
                        num_bridging_o += 1
                
                # Count this Si's Qn value
                if num_bridging_o <= 4:
                    qn_counts[num_bridging_o] += 1
            
            # Convert to fractions
            total_si = len(si_indices)
            qn_fractions = {n: count/total_si for n, count in qn_counts.items()}
            
            qn_data = {
                'qn_counts': qn_counts,
                'qn_fractions': qn_fractions,
                'total_si': total_si,
                'params': {
                    'bond_threshold': bond_threshold,
                    'source': f'{struct_id}.vasp'
                }
            }
            
            filepath = os.path.join(QN_DIR, f'{struct_id}_qn.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(qn_data, f)
                
        except Exception as e:
            print(f"Error processing Qn for structure {struct_id}: {e}")

def populate_bond_angle_distributions(struct_ids, folder_path=POSCAR_DIR, 
                                     angle_range=(60, 180), bins=120):
    """
    Calculate and save bond angle distributions for each structure
    Focuses on Si-O-Si angles (most important for glass structure)
    """
    ensure_directories()
    
    for struct_id in tqdm(struct_ids, desc="Calculating bond angle distributions"):
        try:
            structure, energy = vasp_to_pymatgen(struct_id, folder_path)
            
            # Find Si and O atoms
            si_indices = [i for i, site in enumerate(structure) if site.specie.symbol == 'Si']
            o_indices = [i for i, site in enumerate(structure) if site.specie.symbol == 'O']
            
            # Distance threshold for bonds
            bond_threshold = 1.8
            
            # Collect all Si-O-Si angles
            angles = []
            
            for o_idx in o_indices:
                # Find Si atoms bonded to this O
                bonded_si = []
                for si_idx in si_indices:
                    dist = structure.get_distance(o_idx, si_idx)
                    if dist < bond_threshold:
                        bonded_si.append(si_idx)
                
                # If O is bridging (bonded to 2 Si), calculate angle
                if len(bonded_si) == 2:
                    si1_idx, si2_idx = bonded_si
                    
                    # Get all three distances with PBC
                    d_si1_o = structure.get_distance(si1_idx, o_idx)
                    d_si2_o = structure.get_distance(si2_idx, o_idx)
                    d_si1_si2 = structure.get_distance(si1_idx, si2_idx)
                    
                    # Calculate angle using law of cosines
                    # For angle at O: cos(angle) = (d1^2 + d2^2 - d_si_si^2) / (2*d1*d2)
                    cos_angle = (d_si1_o**2 + d_si2_o**2 - d_si1_si2**2) / (2 * d_si1_o * d_si2_o)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.degrees(np.arccos(cos_angle))
                    
                    angles.append(angle)
            
            # Create histogram
            hist, bin_edges = np.histogram(angles, bins=bins, range=angle_range)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Normalize to probability distribution
            hist_normalized = hist / (np.sum(hist) * (bin_edges[1] - bin_edges[0]))
            
            bad_data = {
                'angles': np.array(angles),
                'histogram': hist_normalized,
                'bin_centers': bin_centers,
                'bin_edges': bin_edges,
                'num_angles': len(angles),
                'mean_angle': np.mean(angles) if angles else 0,
                'std_angle': np.std(angles) if angles else 0,
                'params': {
                    'bond_threshold': bond_threshold,
                    'angle_range': angle_range,
                    'bins': bins,
                    'source': f'{struct_id}.vasp'
                }
            }
            
            filepath = os.path.join(BAD_DIR, f'{struct_id}_bad.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(bad_data, f)
                
        except Exception as e:
            print(f"Error processing BAD for structure {struct_id}: {e}")
                                  
def populate_energies(struct_ids, folder_path=POSCAR_DIR):
    """
    Extract energies from POSCAR files and save to individual files

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to process (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    folder_path : str
        Path to POSCAR files
    """
    ensure_directories()
    
    for struct_id in tqdm(struct_ids, desc="Populating energies"):
        try:
            structure, energy = vasp_to_pymatgen(struct_id, folder_path)
            
            energy_data = {
                'energy': energy,
                'volume': structure.volume,
                'params': {'source': f'{struct_id}.vasp', 'units': 'Ry'}
            }
            
            filepath = os.path.join(ENERGY_DIR, f'{struct_id}_energy.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(energy_data, f)
                
        except Exception as e:
            print(f"Error processing structure {struct_id}: {e}")

def populate_structures(struct_ids, folder_path=POSCAR_DIR):
    """
    Convert and save pymatgen structures to individual files

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to process (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    folder_path : str
        Path to POSCAR files
    """
    ensure_directories()
    
    for struct_id in tqdm(struct_ids, desc="Populating structures"):
        try:
            structure, energy = vasp_to_pymatgen(struct_id, folder_path)
            
            struct_data = {
                'structure': structure,
                'params': {'source': f'{struct_id}.vasp', 'converted_from': 'POSCAR'}
            }
            
            filepath = os.path.join(STRUCT_DIR, f'{struct_id}_struct.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(struct_data, f)
                
        except Exception as e:
            print(f"Error processing structure {struct_id}: {e}")

def populate_rdfs(struct_ids, folder_path=POSCAR_DIR, r_range=(0, 10), bins=1000,
                  element_pairs=None, temperature=1800):
    """
    Calculate and save RDFs for each structure

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to process (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    folder_path : str
        Path to POSCAR files
    r_range : tuple
        (r_min, r_max) for RDF calculation
    bins : int
        Number of bins
    element_pairs : list of tuples
        Element pairs to calculate. If None, uses standard pairs
    temperature : float
        Temperature for ensemble calculations
    """
    ensure_directories()
    
    if element_pairs is None:
        element_pairs = STANDARD_PARAMS['element_pairs']
    
    # Import here to avoid circular imports
    from src.rdf_v2 import calculate_rdf, calculate_partial_rdfs
    
    params = {
        'r_range': r_range,
        'bins': bins,
        'temperature': temperature,
        'element_pairs': element_pairs
    }
    
    for struct_id in tqdm(struct_ids, desc="Populating RDFs"):
        try:
            structure, energy = vasp_to_pymatgen(struct_id, folder_path)
            
            # Calculate total RDF
            r_values, g_r = calculate_rdf(structure, r_range, bins)
            total_rdf_data = {'params': params, 'data': (r_values, g_r)}
            
            filepath = os.path.join(RDF_DIR, f'{struct_id}_total_rdf.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(total_rdf_data, f)
            
            # Calculate partial RDFs
            partial_rdfs = calculate_partial_rdfs(structure, r_range, bins, element_pairs)
            
            for pair, (r_vals, g_vals) in partial_rdfs.items():
                pair_name = f"{pair[0]}_{pair[1]}"
                partial_data = {'params': params, 'data': (r_vals, g_vals)}
                
                filepath = os.path.join(RDF_DIR, f'{struct_id}_{pair_name}_rdf.pkl')
                with open(filepath, 'wb') as f:
                    pickle.dump(partial_data, f)
                    
        except Exception as e:
            print(f"Error processing RDF for structure {struct_id}: {e}")

def populate_counting_functions(struct_ids, folder_path=POSCAR_DIR, r_range=(0, 10),
                               bins=200, element_pairs=None, temperature=1800):
    """
    Calculate and save counting functions for each structure

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to process (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    folder_path : str
        Path to POSCAR files
    r_range : tuple
        (r_min, r_max) for counting function calculation
    bins : int
        Number of bins
    element_pairs : list of tuples
        Element pairs to calculate. If None, uses standard pairs
    temperature : float
        Temperature for ensemble calculations
    """
    ensure_directories()
    
    if element_pairs is None:
        element_pairs = STANDARD_PARAMS['element_pairs']
    
    # Import here to avoid circular imports
    from src.rdf_v2 import simple_counting_function, calculate_partial_counting_functions
    
    params = {
        'r_range': r_range,
        'bins': bins,
        'temperature': temperature,
        'element_pairs': element_pairs
    }
    
    for struct_id in tqdm(struct_ids, desc="Populating counting functions"):
        try:
            structure, energy = vasp_to_pymatgen(struct_id, folder_path)
            
            # Calculate partial counting functions
            partial_cfs = calculate_partial_counting_functions(
                {struct_id: {'structure': structure}}, struct_id, element_pairs, r_range, bins
            )
            
            for pair, (r_vals, cf_vals) in partial_cfs.items():
                pair_name = f"{pair[0]}_{pair[1]}"
                cf_data = {'params': params, 'data': (r_vals, cf_vals)}
                
                filepath = os.path.join(CF_DIR, f'{struct_id}_{pair_name}_cf.pkl')
                with open(filepath, 'wb') as f:
                    pickle.dump(cf_data, f)
                    
        except Exception as e:
            print(f"Error processing counting function for structure {struct_id}: {e}")

def kamel_lobe_smooth(r_values, g_r, w=0.03):
    """
    Apply KAMEL-LOBE smoothing to RDF data using the official M5-Lab implementation
    
    Direct implementation of the algorithm from:
    S. Arman Ghaffarizadeh & Gerald J. Wang
    "Getting over the hump with KAMEL-LOBE: Kernel-averaging method to eliminate 
    length-of-bin effects in radial distribution functions"
    Journal of Chemical Physics (2023)
    
    Parameters:
    -----------
    r_values : array-like
        Radial distance values (equispaced)
    g_r : array-like
        RDF values
    w : float
        Non-dimensional Gaussian kernel width (default: 0.015 as recommended)
        
    Returns:
    --------
    tuple : (r_tilde, gr_tilde) - smoothed r and RDF values
    """
    from scipy.stats import norm
    import scipy.sparse as sp
    
    r = np.array(r_values)
    RDF = np.array(g_r)
    
    Nbins = RDF.shape[0]  # number of bins
    delr = r[1] - r[0]    # bin width
    m_KL = int(np.ceil(2*w/delr))  # number of bins to average over
    
    if m_KL <= 1:
        print('w <= delr/2, no averaging is performed')
        return r, RDF
    
    # ===== Compute T1 matrix =====
    T1 = np.zeros((Nbins, Nbins))
    for col in range(1, Nbins):  # First column is all zeros
        value = (col * delr)**2
        T1[col, col] = value
        T1[col+1:, col] = 2 * value
    
    # ===== Compute T2 matrix =====
    k_KL = 2 * m_KL - 1
    fractions = np.zeros((1, k_KL))
    A1_block = sp.identity(m_KL, format='csr')
    A2_block = sp.lil_matrix((m_KL, Nbins - m_KL))
    
    # Calculate fractions using normal CDF
    fractions[0, m_KL-1:] = (norm.cdf(((np.arange(0, m_KL) + 0.5) * delr), 0, w) - 
                            norm.cdf(((np.arange(0, m_KL) - 0.5) * delr), 0, w))
    fractions[0, :m_KL-1] = np.flip(fractions[0, m_KL:2*m_KL-1])
    fractions[0, :] *= 1/np.sum(fractions)
    
    # Build T2 matrix using sparse operations
    B_block = sp.diags(np.tile(fractions, (Nbins-2*m_KL, 1)).T, 
                       np.arange(0, 2*(m_KL-1)+1), 
                       shape=(Nbins-2*m_KL, Nbins))
    T2 = sp.vstack((sp.hstack((A1_block, A2_block)), 
                    B_block, 
                    sp.hstack((A2_block, A1_block))))
    
    # ===== Compute T3 matrix =====
    T3 = np.zeros((Nbins, Nbins))
    constant = 1/(delr**2)
    for row in range(1, Nbins):
        T3[row, row] = constant/(row)**2
        factor = 2*constant/(row)**2
        sign = 1 - 2 * (row & 1)  # alternating sign pattern
        for col in range(row):
            T3[row, col] = sign * factor
            sign *= -1
    
    # Apply the three transformations: g_tilde = T3 @ T2 @ T1 @ RDF
    intermediate1 = T1 @ RDF
    intermediate2 = T2 @ intermediate1
    gr_convert = T3 @ intermediate2
    
    # Extract final result (remove last m_KL bins)
    gr_tilde = gr_convert[:-m_KL]
    r_tilde = r[:-m_KL]

    gr_tilde = np.maximum(gr_tilde, 0.0)
    
    return r_tilde, gr_tilde

def smooth_all_rdfs(w=0.03, overwrite=False):
    """
    Apply KAMEL-LOBE smoothing to all existing RDF files
    
    Parameters:
    -----------
    w : float
        Non-dimensional Gaussian kernel width (default: 0.015)
    overwrite : bool
        Whether to overwrite existing smoothed files (default: False)
    """
    # Ensure smooth_rdfs directory exists
    os.makedirs(SMOOTH_RDF_DIR, exist_ok=True)
    
    # Get all RDF files
    rdf_files = [f for f in os.listdir(RDF_DIR) if f.endswith('.pkl')]
    
    print(f"Found {len(rdf_files)} RDF files to smooth")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for filename in tqdm(rdf_files, desc="Smoothing RDFs"):
        try:
            # Check if output file already exists
            output_path = os.path.join(SMOOTH_RDF_DIR, filename)
            if os.path.exists(output_path) and not overwrite:
                skipped += 1
                continue
            
            # Load original RDF data
            input_path = os.path.join(RDF_DIR, filename)
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract r_values and g_r
            r_values, g_r = data['data']
            
            # Apply KAMEL-LOBE smoothing
            r_smooth, g_smooth = kamel_lobe_smooth(r_values, g_r, w=w)
            
            # Create new data structure with smoothed values
            smoothed_data = {
                'params': data['params'].copy(),
                'data': (r_smooth, g_smooth),
                'smoothing': {
                    'method': 'KAMEL-LOBE',
                    'kernel_width': w,
                    'original_file': filename
                }
            }
            
            # Add smoothing info to params
            smoothed_data['params']['smoothed'] = True
            smoothed_data['params']['smoothing_method'] = 'KAMEL-LOBE'
            smoothed_data['params']['kernel_width'] = w
            
            # Save smoothed data
            with open(output_path, 'wb') as f:
                pickle.dump(smoothed_data, f)
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            errors += 1
    
    print(f"\nSmoothing complete!")
    print(f"Processed: {processed}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Errors: {errors}")
    print(f"Smoothed RDFs saved to: {SMOOTH_RDF_DIR}")

def populate_ring_statistics(struct_ids, folder_path=POSCAR_DIR, cutoffs=None, max_ring_size=20):
    """
    Calculate and save ring statistics for structures.

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to process (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    folder_path : str
        Path to POSCAR files
    cutoffs : dict, optional
        Custom bond cutoffs
    max_ring_size : int
        Maximum ring size to search for
    """
    ensure_directories()
    
    from tqdm import tqdm
    
    for struct_id in tqdm(struct_ids, desc="Calculating ring statistics"):
        # Load structure
        structure, _ = vasp_to_pymatgen(struct_id, folder_path)
        
        # Calculate ring statistics
        ring_stats = calculate_ring_statistics_for_structure(
            structure, 
            cutoffs=cutoffs,
            max_ring_size=max_ring_size
        )
        
        # Save to file
        output_file = os.path.join(RING_DIR, f'{struct_id}_rings.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(ring_stats, f)

# =====================================================================
# LOADING FUNCTIONS (Read data from disk)
# =====================================================================

def load_energies(struct_ids):
    """
    Load energy data for specified structures

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to load (e.g., ['SiO2_36_481', 'SiO2_24_2'])

    Returns:
    --------
    dict : {struct_id: energy_value}
    """
    energies = {}
    
    for struct_id in struct_ids:
        filepath = os.path.join(ENERGY_DIR, f'{struct_id}_energy.pkl')
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                energies[struct_id] = data['energy']
        except FileNotFoundError:
            print(f"Warning: Energy file not found for structure {struct_id}")
        except Exception as e:
            print(f"Error loading energy for structure {struct_id}: {e}")
    
    return energies

def load_structures(struct_ids):
    """
    Load pymatgen structures for specified structures

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to load (e.g., ['SiO2_36_481', 'SiO2_24_2'])

    Returns:
    --------
    dict : {struct_id: pymatgen_structure}
    """
    structures = {}
    
    for struct_id in struct_ids:
        filepath = os.path.join(STRUCT_DIR, f'{struct_id}_struct.pkl')
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                structures[struct_id] = data['structure']
        except FileNotFoundError:
            print(f"Warning: Structure file not found for structure {struct_id}")
        except Exception as e:
            print(f"Error loading structure {struct_id}: {e}")
    
    return structures

def load_rdfs(struct_ids, pairs='all', smoothed=False):
    """
    Load RDF data for specified structures and pairs

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to load (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    pairs : str or list
        'all' for all available pairs, 'total' for total RDF,
        or list like ['Si_Si', 'Si_O'] for specific pairs
    smoothed : bool
        If True, load from SMOOTH_RDF_DIR. If False, load from RDF_DIR (default: False)

    Returns:
    --------
    dict : {struct_id: {pair: (r_values, g_r)} or {struct_id: (r_values, g_r)} for total}
    """
    # Choose directory based on smoothed parameter
    data_dir = SMOOTH_RDF_DIR if smoothed else RDF_DIR

    rdfs = {}
    
    # Determine which files to load
    if pairs == 'total':
        file_patterns = ['total_rdf']
    elif pairs == 'all':
        file_patterns = ['total_rdf', 'Si_Si_rdf', 'Si_O_rdf', 'O_O_rdf']
    else:
        file_patterns = [f"{pair}_rdf" for pair in pairs]
    
    for struct_id in struct_ids:
        struct_rdfs = {}
        
        for pattern in file_patterns:
            filepath = os.path.join(data_dir, f'{struct_id}_{pattern}.pkl')
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    
                    if pattern == 'total_rdf':
                        if pairs == 'total':
                            rdfs[struct_id] = data['data']
                        else:
                            struct_rdfs['total'] = data['data']
                    else:
                        # Convert back to tuple format for pair key
                        pair_str = pattern.replace('_rdf', '')
                        if '_' in pair_str:
                            elem1, elem2 = pair_str.split('_')
                            struct_rdfs[(elem1, elem2)] = data['data']
                        
            except FileNotFoundError:
                data_type = "Smoothed" if smoothed else "Original"
                print(f"Warning: {data_type} RDF file not found: {struct_id}_{pattern}.pkl")
            except Exception as e:
                print(f"Error loading RDF {struct_id}_{pattern}: {e}")
        
        if pairs != 'total' and struct_rdfs:
            rdfs[struct_id] = struct_rdfs
    
    return rdfs

def load_counting_functions(struct_ids, pairs='all'):
    """
    Load counting function data for specified structures and pairs

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to load (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    pairs : str or list
        'all' for all available pairs or list like ['Si_Si', 'Si_O'] for specific pairs

    Returns:
    --------
    dict : {struct_id: {pair: (r_values, counting_func)}}
    """
    cfs = {}
    
    # Determine which files to load
    if pairs == 'all':
        file_patterns = ['Si_Si_cf', 'Si_O_cf', 'O_O_cf']
    else:
        file_patterns = [f"{pair}_cf" for pair in pairs]
    
    for struct_id in struct_ids:
        struct_cfs = {}
        
        for pattern in file_patterns:
            filepath = os.path.join(CF_DIR, f'{struct_id}_{pattern}.pkl')
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    
                    # Convert back to tuple format for pair key
                    pair_str = pattern.replace('_cf', '')
                    if '_' in pair_str:
                        elem1, elem2 = pair_str.split('_')
                        struct_cfs[(elem1, elem2)] = data['data']
                        
            except FileNotFoundError:
                print(f"Warning: CF file not found: {struct_id}_{pattern}.pkl")
            except Exception as e:
                print(f"Error loading counting function {struct_id}_{pattern}: {e}")
        
        if struct_cfs:
            cfs[struct_id] = struct_cfs
    
    return cfs

def load_densities(struct_ids):
    """
    Load density data for specified structure IDs

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to load (e.g., ['SiO2_36_481', 'SiO2_24_2'])

    Returns:
    --------
    dict : Dictionary mapping struct_id to density data
    """
    densities_dir = os.path.join(BASE_DATA_DIR, 'densities')
    densities = {}
    
    for struct_id in struct_ids:
        filepath = os.path.join(densities_dir, f'{struct_id}_density.pkl')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                densities[struct_id] = pickle.load(f)
        else:
            print(f"Warning: Density file not found for structure {struct_id}")
    
    return densities

def load_qn_distributions(struct_ids):
    """
    Load Qn distribution data for specified structures

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to load (e.g., ['SiO2_36_481', 'SiO2_24_2'])

    Returns:
    --------
    dict : {struct_id: qn_data}
    """
    qn_data = {}
    
    for struct_id in struct_ids:
        filepath = os.path.join(QN_DIR, f'{struct_id}_qn.pkl')
        try:
            with open(filepath, 'rb') as f:
                qn_data[struct_id] = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Qn file not found for structure {struct_id}")
        except Exception as e:
            print(f"Error loading Qn for structure {struct_id}: {e}")
    
    return qn_data

def load_bond_angle_distributions(struct_ids):
    """
    Load bond angle distribution data for specified structures

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to load (e.g., ['SiO2_36_481', 'SiO2_24_2'])

    Returns:
    --------
    dict : {struct_id: bad_data}
    """
    bad_data = {}
    
    for struct_id in struct_ids:
        filepath = os.path.join(BAD_DIR, f'{struct_id}_bad.pkl')
        try:
            with open(filepath, 'rb') as f:
                bad_data[struct_id] = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: BAD file not found for structure {struct_id}")
        except Exception as e:
            print(f"Error loading BAD for structure {struct_id}: {e}")
    
    return bad_data

def load_ring_statistics(struct_ids):
    """
    Load ring statistics from saved files.

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to load (e.g., ['SiO2_36_481', 'SiO2_24_2'])

    Returns:
    --------
    dict : Dictionary mapping struct_id -> ring_statistics
    """
    ring_stats = {}
    
    for struct_id in struct_ids:
        ring_file = os.path.join(RING_DIR, f'{struct_id}_rings.pkl')
        
        if os.path.exists(ring_file):
            with open(ring_file, 'rb') as f:
                ring_stats[struct_id] = pickle.load(f)
        else:
            print(f"Warning: Ring statistics not found for structure {struct_id}")
    
    return ring_stats


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def calculate_ensemble_average_density(struct_ids, energies=None, temperature=1800):
    """
    Calculate Boltzmann-weighted ensemble average density

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    energies : dict, optional
        Energy data {struct_id: energy}
    temperature : float
        Temperature in Kelvin

    Returns:
    --------
    dict : Dictionary with average densities
    """
    densities = load_densities(struct_ids)
    if energies is None:
        energies = load_energies(struct_ids)
    structures = load_structures(struct_ids)

    weights = calculate_weights(struct_ids, energies, structures, temperature)
    
    avg_total_density = 0
    avg_partial_densities = {}

    for struct_id in struct_ids:
        if struct_id in densities and struct_id in weights:
            weight = weights[struct_id]
            density_data = densities[struct_id]
            
            avg_total_density += weight * density_data['total_density']
            
            for elem, partial_density in density_data['partial_densities'].items():
                if elem not in avg_partial_densities:
                    avg_partial_densities[elem] = 0
                avg_partial_densities[elem] += weight * partial_density
    
    return {
        'total_density': avg_total_density,
        'partial_densities': avg_partial_densities
    }

def populate_all_data(struct_ids, folder_path=POSCAR_DIR, **kwargs):
    """
    Convenience function to populate all data types for given structure IDs

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to process (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    folder_path : str
        Path to POSCAR files
    **kwargs : dict
        Additional parameters for calculations
    """
    print("Populating all data types...")
    populate_energies(struct_ids, folder_path)
    populate_structures(struct_ids, folder_path)
    populate_densities(struct_ids, folder_path)
    populate_rdfs(struct_ids, folder_path, **kwargs)
    populate_counting_functions(struct_ids, folder_path, **kwargs)
    populate_qn_distributions(struct_ids, folder_path)
    populate_bond_angle_distributions(struct_ids, folder_path, **kwargs)
    print("All data populated successfully!")

def check_data_availability(struct_ids, data_types=None):
    """
    Check which data files exist for given structure IDs

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to check (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    data_types : list of str
        Data types to check ['energies', 'structures', 'rdfs', 'cfs']
        If None, checks all types

    Returns:
    --------
    dict : Summary of available data
    """
    if data_types is None:
        data_types = ['energies', 'structures', 'rdfs', 'cfs']
    
    availability = {}
    
    for struct_id in struct_ids:
        availability[struct_id] = {}
        
        if 'energies' in data_types:
            energy_file = os.path.join(ENERGY_DIR, f'{struct_id}_energy.pkl')
            availability[struct_id]['energy'] = os.path.exists(energy_file)
            
        if 'structures' in data_types:
            struct_file = os.path.join(STRUCT_DIR, f'{struct_id}_struct.pkl')
            availability[struct_id]['structure'] = os.path.exists(struct_file)
            
        if 'rdfs' in data_types:
            rdf_files = [f'{struct_id}_total_rdf.pkl', f'{struct_id}_Si_Si_rdf.pkl', 
                        f'{struct_id}_Si_O_rdf.pkl', f'{struct_id}_O_O_rdf.pkl']
            rdf_exists = [os.path.exists(os.path.join(RDF_DIR, f)) for f in rdf_files]
            availability[struct_id]['rdfs'] = sum(rdf_exists)
            
        if 'cfs' in data_types:
            cf_files = [f'{struct_id}_Si_Si_cf.pkl', f'{struct_id}_Si_O_cf.pkl', 
                       f'{struct_id}_O_O_cf.pkl']
            cf_exists = [os.path.exists(os.path.join(CF_DIR, f)) for f in cf_files]
            availability[struct_id]['cfs'] = sum(cf_exists)
    
    return availability

def calculate_ensemble_qn(struct_ids, use_weights=False, temperature=1800):
    """
    Calculate ensemble-averaged Qn distribution.

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    use_weights : bool
        If True, apply Boltzmann weighting (default: False)
    temperature : float
        Temperature in Kelvin for Boltzmann weighting

    Returns:
    --------
    dict : {'qn_fractions': {0: frac, 1: frac, ...}}
    """
    qn_data = load_qn_distributions(struct_ids)

    if use_weights:
        energies = load_energies(struct_ids)
        structures = load_structures(struct_ids)
        weights = calculate_weights(struct_ids, energies, structures, temperature)
    else:
        weights = {sid: 1.0 / len(struct_ids) for sid in struct_ids}

    ensemble_qn = {n: 0.0 for n in range(5)}

    for struct_id in struct_ids:
        if struct_id in qn_data and struct_id in weights:
            qn_fracs = qn_data[struct_id]['qn_fractions']
            w = weights[struct_id]
            for n in range(5):
                ensemble_qn[n] += w * qn_fracs.get(n, 0)

    total = sum(ensemble_qn.values())
    if total > 0:
        ensemble_qn = {n: frac / total for n, frac in ensemble_qn.items()}

    return {'qn_fractions': ensemble_qn}

def calculate_ensemble_bad(struct_ids, bins=60, use_weights=False, temperature=1800):
    """
    Calculate ensemble-averaged bond angle distribution.

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    bins : int
        Number of bins for histogram (default: 60)
    use_weights : bool
        If True, apply Boltzmann weighting (default: False)
    temperature : float
        Temperature in Kelvin for Boltzmann weighting

    Returns:
    --------
    dict : {'bin_centers': array, 'histogram': array, 'mean': float, 'std': float}
    """
    bad_data = load_bond_angle_distributions(struct_ids)

    if use_weights:
        energies = load_energies(struct_ids)
        structures = load_structures(struct_ids)
        weights = calculate_weights(struct_ids, energies, structures, temperature)
    else:
        weights = {sid: 1.0 / len(struct_ids) for sid in struct_ids}

    # Weighted histogram: accumulate weighted angle counts
    angle_range = (60, 180)
    bin_edges = np.linspace(angle_range[0], angle_range[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weighted_hist = np.zeros(bins)
    weighted_mean = 0.0
    weighted_sq = 0.0
    total_weight = 0.0

    for struct_id in struct_ids:
        if struct_id in bad_data and struct_id in weights:
            angles = np.array(bad_data[struct_id]['angles'])
            w = weights[struct_id]
            hist, _ = np.histogram(angles, bins=bin_edges)
            weighted_hist += w * hist
            weighted_mean += w * np.mean(angles)
            weighted_sq += w * np.mean(angles ** 2)
            total_weight += w

    # Normalize histogram to probability density
    bin_width = bin_edges[1] - bin_edges[0]
    hist_sum = np.sum(weighted_hist)
    if hist_sum > 0:
        weighted_hist = weighted_hist / (hist_sum * bin_width)

    weighted_std = np.sqrt(max(weighted_sq - weighted_mean ** 2, 0))

    return {
        'bin_centers': bin_centers,
        'histogram': weighted_hist,
        'mean': weighted_mean,
        'std': weighted_std
    }

def calculate_ensemble_rings(struct_ids, max_ring_size=20, use_weights=False, temperature=1800):
    """
    Calculate ensemble-averaged ring statistics (RN and RC).

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs (e.g., ['SiO2_36_481', 'SiO2_24_2'])
    max_ring_size : int
        Maximum ring size to include (default: 20)
    use_weights : bool
        If True, apply Boltzmann weighting (default: False)
    temperature : float
        Temperature in Kelvin for Boltzmann weighting

    Returns:
    --------
    dict : {'RN': {size: avg}, 'RC': {size: avg}}
    """
    ring_data = load_ring_statistics(struct_ids)

    if use_weights:
        energies = load_energies(struct_ids)
        structures = load_structures(struct_ids)
        weights = calculate_weights(struct_ids, energies, structures, temperature)
    else:
        weights = {sid: 1.0 / len(struct_ids) for sid in struct_ids}

    ring_sizes = range(2, max_ring_size + 1)
    avg_rn = {s: 0.0 for s in ring_sizes}
    avg_rc = {s: 0.0 for s in ring_sizes}

    for struct_id in struct_ids:
        if struct_id in ring_data and struct_id in weights:
            w = weights[struct_id]
            stats = ring_data[struct_id]
            for s in ring_sizes:
                avg_rn[s] += w * stats['RN'].get(s, 0.0)
                avg_rc[s] += w * stats['RC'].get(s, 0.0)

    return {'RN': avg_rn, 'RC': avg_rc}

def check_ring_data_availability(struct_ids):
    """
    Check which structures have ring statistics computed.

    Parameters:
    -----------
    struct_ids : list of str
        Structure IDs to check (e.g., ['SiO2_36_481', 'SiO2_24_2'])

    Returns:
    --------
    dict : Dictionary mapping struct_id -> bool (whether rings exist)
    """
    availability = {}
    
    for struct_id in struct_ids:
        ring_file = os.path.join(RING_DIR, f'{struct_id}_rings.pkl')
        availability[struct_id] = os.path.exists(ring_file)
    
    return availability