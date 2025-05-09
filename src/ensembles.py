import random
import numpy as np

def random_structure_sample(structure_dict, sample_size=100):
    """
    Randomly sample a specified number of structure IDs from a structure dictionary.
    
    Parameters:
    -----------
    structure_dict : dict or OrderedDict
        Dictionary containing structure IDs as keys
    sample_size : int, optional
        Number of structure IDs to sample (default: 100)
    save_to_file : str, optional
        Filename to save the sample IDs (will use .npy format)
        
    Returns:
    --------
    numpy.ndarray
        Array of sampled structure IDs
    """
    # Get all available structure IDs
    available_ids = list(structure_dict.keys())
    
    # Convert string IDs to integers if needed
    try:
        available_ids = [int(id_str) for id_str in available_ids]
    except:
        # If conversion fails, keep as is
        pass
    
    # Check if we have enough structures
    if len(available_ids) < sample_size:
        print(f"Warning: Requested {sample_size} samples but only {len(available_ids)} available")
        sample_size = len(available_ids)
    
    # Randomly sample structure IDs
    sampled_ids = random.sample(available_ids, sample_size)
    
    # Convert to numpy array
    sampled_ids_array = np.array(sampled_ids)
    
    # Sort the array for convenience
    sampled_ids_array.sort()
    
    return sampled_ids_array

