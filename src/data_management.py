import os
import numpy as np
import re
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from collections import OrderedDict
import pickle

def vasp_to_pymatgen(struct_id, folder_path = '/Users/raphaelzstone/Documents/CMU/Research/Ensemble-Analysis/data/3k_poscar'):
    """
    Read a VASP POSCAR file and convert it to a pymatgen Structure object.
    Also extracts energy from the first line if available.
    
    Parameters:
    -----------
    filepath : str
        Path to the VASP POSCAR file
    
    Returns:
    --------
    tuple
        (pymatgen_structure, energy)
    """
    # Extract energy from the first line

    os.chdir(folder_path)
    filepath = f"{struct_id}.vasp"
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

def build_structure_energy_dict(struct_ids, folder_path = '/Users/raphaelzstone/Documents/CMU/Research/Ensemble-Analysis/data/3k_poscar'):
    """
    Build a dictionary of structure-energy pairs by reading VASP files with numeric names.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing VASP files named as numbers (e.g., "0.vasp", "1.vasp", etc.)
    
    Returns:
    --------
    OrderedDict
        Dictionary mapping file numbers (as strings) to dictionaries containing structure and energy
    """
    # Change to the specified directory
    os.chdir(folder_path)
    
    # Create an ordered dictionary to preserve the order of files
    structures_dict = OrderedDict()
    
    # Track processing statistics
    processed_count = 0
    error_count = 0
    
    # Process files from 0 to 999
    for i in struct_ids:
        filename = f"{i}.vasp"

        # Check if file exists
        if os.path.exists(filename):
            try:
                # Use the previous function to get structure and energy
                structure, energy = vasp_to_pymatgen(i, folder_path)
                
                # Store in dictionary
                structures_dict[str(i)] = {
                    "structure": structure,
                    "Energy (Ry)": energy
                }
                
                processed_count += 1
                
                # # Print progress every 100 files
                # if processed_count % 100 == 0:
                #     print(f"Processed {processed_count} files...")
                    
            except Exception as e:
                error_count += 1
                print(f"Error processing {filename}: {e}")
    
    # Print summary
    print(f"Completed! Successfully processed {processed_count} files with {error_count} errors.")
    print(f"Total structures in dictionary: {len(structures_dict)}")
    
    return structures_dict

def array_to_npy(array, filename):
    """
    Save an array to a NumPy .npy file.
    
    Parameters:
    -----------
    array : array-like
        The array to save (can be a list, numpy array, etc.)
    filename : str
        Filename to save to (will append .npy if not present)
    
    Returns:
    --------
    str
        The full path to the saved file
    """
    # Convert to numpy array if not already
    np_array = np.array(array)
    
    # Add extension if not present
    if not filename.endswith('.npy'):
        filename = filename + '.npy'
    
    # Save the array
    np.save(filename, np_array)
    
    return os.path.abspath(filename)

def npy_to_array(filename):
    """
    Load an array from a NumPy .npy file.
    
    Parameters:
    -----------
    filename : str
        Path to the .npy file
    
    Returns:
    --------
    numpy.ndarray
        The loaded array
    """
    # Add extension if not present
    if not filename.endswith('.npy'):
        filename = filename + '.npy'
        
    # Load from .npy file
    return np.load(filename)

def build_energy_dict(struct_ids, folder_path = '/Users/raphaelzstone/Documents/CMU/Research/Ensemble-Analysis/data/3k_poscar'):
    """
    Build a dictionary of structure-energy pairs by reading VASP files with numeric names.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing VASP files named as numbers (e.g., "0.vasp", "1.vasp", etc.)
    
    Returns:
    --------
    OrderedDict
        Dictionary mapping file numbers (as strings) to dictionaries containing structure and energy
    """
    # Change to the specified directory
    os.chdir(folder_path)
    
    # Create an ordered dictionary to preserve the order of files
    structures_dict = OrderedDict()
    
    # Track processing statistics
    processed_count = 0
    error_count = 0
    
    # Process files from 0 to 999
    for i in struct_ids:
        filename = f"{i}.vasp"

        # Check if file exists
        if os.path.exists(filename):
            try:
                # Use the previous function to get structure and energy
                structure, energy = vasp_to_pymatgen(i, folder_path)
                
                # Store in dictionary
                structures_dict[str(i)] = energy
                
                processed_count += 1
                
                # Print progress every 100 files
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")
                    
            except Exception as e:
                error_count += 1
                print(f"Error processing {filename}: {e}")
    
    # Print summary
    print(f"Completed! Successfully processed {processed_count} files with {error_count} errors.")
    print(f"Total structures in dictionary: {len(structures_dict)}")
    
    return structures_dict

def build_structure_dict(struct_ids, folder_path = '/Users/raphaelzstone/Documents/CMU/Research/Ensemble-Analysis/data/3k_poscar'):
    """
    Build a dictionary of structure-energy pairs by reading VASP files with numeric names.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing VASP files named as numbers (e.g., "0.vasp", "1.vasp", etc.)
    
    Returns:
    --------
    OrderedDict
        Dictionary mapping file numbers (as strings) to dictionaries containing structure and energy
    """
    # Change to the specified directory
    os.chdir(folder_path)
    
    # Create an ordered dictionary to preserve the order of files
    structures_dict = OrderedDict()
    
    # Track processing statistics
    processed_count = 0
    error_count = 0
    
    # Process files from 0 to 999
    for i in struct_ids:
        filename = f"{i}.vasp"

        # Check if file exists
        if os.path.exists(filename):
            try:
                # Use the previous function to get structure and energy
                structure, energy = vasp_to_pymatgen(i, folder_path)
                
                # Store in dictionary
                structures_dict[str(i)] = structure
                
                processed_count += 1
                
                # Print progress every 100 files
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")
                    
            except Exception as e:
                error_count += 1
                print(f"Error processing {filename}: {e}")
    
    # Print summary
    print(f"Completed! Successfully processed {processed_count} files with {error_count} errors.")
    print(f"Total structures in dictionary: {len(structures_dict)}")
    
    return structures_dict

def save_rdf_data(ensemble_rdfs, filename):
    """
    Save RDF data to a file for later use.
    
    Parameters:
    -----------
    ensemble_rdfs : dict
        Dictionary with element pairs as keys and (r_values, g_r) as values
    filename : str
        Path to save the RDF data
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(ensemble_rdfs, f)
        print(f"RDF data saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving RDF data: {e}")
        return False

def load_rdf_data(filename):
    """
    Load RDF data from a file.
    
    Parameters:
    -----------
    filename : str
        Path to the RDF data file
        
    Returns:
    --------
    dict
        Dictionary with element pairs as keys and (r_values, g_r) as values,
        or None if loading fails
    """
    try:
        with open(filename, 'rb') as f:
            ensemble_rdfs = pickle.load(f)
        print(f"RDF data loaded from {filename}")
        return ensemble_rdfs
    except Exception as e:
        print(f"Error loading RDF data: {e}")
        return None