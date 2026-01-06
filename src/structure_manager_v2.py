from collections import OrderedDict
import numpy as np
from src.data_management_v2 import load_energies, load_structures, load_rdfs, load_counting_functions

class StructureManager:
    """
    Simplified structure manager for v2 data format
    Handles loading and basic operations on structure ensembles
    """
    
    def __init__(self, struct_ids=None):
        """
        Initialize structure manager
        
        Parameters:
        -----------
        struct_ids : list of int, optional
            Structure IDs to manage. Can be set later.
        """
        self.struct_ids = struct_ids if struct_ids is not None else []
        self._energies = None
        self._structures = None
        self._rdfs = None
        self._counting_functions = None
    
    def set_structure_ids(self, struct_ids):
        """Set the structure IDs to manage"""
        self.struct_ids = list(struct_ids)
        # Clear cached data when IDs change
        self._energies = None
        self._structures = None
        self._rdfs = None
        self._counting_functions = None
    
    def get_structure_ids(self):
        """Get list of structure IDs"""
        return self.struct_ids.copy()
    
    # =====================================================================
    # DATA LOADING METHODS
    # =====================================================================
    
    def load_energies(self, force_reload=False):
        """
        Load energies for managed structures
        
        Parameters:
        -----------
        force_reload : bool
            If True, reload from disk even if cached
            
        Returns:
        --------
        dict : {struct_id: energy}
        """
        if self._energies is None or force_reload:
            self._energies = load_energies(self.struct_ids)
        return self._energies
    
    def load_structures(self, force_reload=False):
        """
        Load pymatgen structures for managed structures
        
        Parameters:
        -----------
        force_reload : bool
            If True, reload from disk even if cached
            
        Returns:
        --------
        dict : {struct_id: pymatgen_structure}
        """
        if self._structures is None or force_reload:
            self._structures = load_structures(self.struct_ids)
        return self._structures
    
    def load_rdfs(self, pairs='all', force_reload=False):
        """
        Load RDF data for managed structures
        
        Parameters:
        -----------
        pairs : str or list
            'all', 'total', or list of specific pairs
        force_reload : bool
            If True, reload from disk even if cached
            
        Returns:
        --------
        dict : RDF data
        """
        if self._rdfs is None or force_reload:
            self._rdfs = load_rdfs(self.struct_ids, pairs)
        return self._rdfs
    
    def load_counting_functions(self, pairs='all', force_reload=False):
        """
        Load counting function data for managed structures
        
        Parameters:
        -----------
        pairs : str or list
            'all' or list of specific pairs
        force_reload : bool
            If True, reload from disk even if cached
            
        Returns:
        --------
        dict : Counting function data
        """
        if self._counting_functions is None or force_reload:
            self._counting_functions = load_counting_functions(self.struct_ids, pairs)
        return self._counting_functions
    
    # =====================================================================
    # CONVENIENCE METHODS
    # =====================================================================
    
    def get_energy(self, struct_id):
        """Get energy for a single structure"""
        energies = self.load_energies()
        return energies.get(struct_id)
    
    def get_structure(self, struct_id):
        """Get pymatgen structure for a single structure"""
        structures = self.load_structures()
        return structures.get(struct_id)
    
    def get_structure_energy_pair(self, struct_id):
        """Get structure-energy pair for a single structure"""
        structure = self.get_structure(struct_id)
        energy = self.get_energy(struct_id)
        
        if structure is not None and energy is not None:
            return {
                "structure": structure,
                "Energy (Ry)": energy
            }
        return None
    
    def get_structures_sorted_by_energy(self):
        """
        Get structures sorted by energy (lowest first)
        
        Returns:
        --------
        list : [(struct_id, {structure, energy}), ...]
        """
        energies = self.load_energies()
        structures = self.load_structures()
        
        # Create pairs and sort by energy
        pairs = []
        for struct_id in self.struct_ids:
            if struct_id in energies and struct_id in structures:
                pair_data = {
                    "structure": structures[struct_id],
                    "Energy (Ry)": energies[struct_id]
                }
                pairs.append((struct_id, pair_data))
        
        # Sort by energy
        pairs.sort(key=lambda x: x[1]["Energy (Ry)"])
        
        return pairs
    
    # =====================================================================
    # STATISTICAL METHODS
    # =====================================================================
    
    def get_energy_statistics(self):
        """
        Get basic statistics about energies
        
        Returns:
        --------
        dict : Statistics summary
        """
        energies = self.load_energies()
        energy_values = list(energies.values())
        
        if not energy_values:
            return {}
        
        energy_array = np.array(energy_values)
        
        return {
            'count': len(energy_values),
            'min': np.min(energy_array),
            'max': np.max(energy_array),
            'mean': np.mean(energy_array),
            'std': np.std(energy_array),
            'median': np.median(energy_array)
        }
    
    def filter_by_energy_range(self, min_energy=None, max_energy=None):
        """
        Filter structures by energy range
        
        Parameters:
        -----------
        min_energy : float, optional
            Minimum energy threshold
        max_energy : float, optional
            Maximum energy threshold
            
        Returns:
        --------
        list : Filtered structure IDs
        """
        energies = self.load_energies()
        filtered_ids = []
        
        for struct_id, energy in energies.items():
            if min_energy is not None and energy < min_energy:
                continue
            if max_energy is not None and energy > max_energy:
                continue
            filtered_ids.append(struct_id)
        
        return filtered_ids
    
    def get_lowest_energy_structures(self, n=10):
        """
        Get the n lowest energy structures
        
        Parameters:
        -----------
        n : int
            Number of structures to return
            
        Returns:
        --------
        list : Structure IDs of lowest energy structures
        """
        sorted_pairs = self.get_structures_sorted_by_energy()
        return [struct_id for struct_id, _ in sorted_pairs[:n]]
    
    # =====================================================================
    # ENSEMBLE COMPATIBILITY METHODS
    # =====================================================================
    
    def to_legacy_format(self):
        """
        Convert to legacy structure_dict format for compatibility
        
        Returns:
        --------
        dict : Legacy format {struct_id_str: {structure, Energy (Ry)}}
        """
        energies = self.load_energies()
        structures = self.load_structures()
        
        legacy_dict = OrderedDict()
        
        for struct_id in self.struct_ids:
            if struct_id in energies and struct_id in structures:
                legacy_dict[str(struct_id)] = {
                    "structure": structures[struct_id],
                    "Energy (Ry)": energies[struct_id]
                }
        
        return legacy_dict
    
    def calculate_weights(self, temperature):
        """
        Calculate Boltzmann weights for ensemble averaging
        
        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin
            
        Returns:
        --------
        dict : {struct_id: weight}
        """
        from src.rdf_v2 import calculate_weights
        
        # Create temporary legacy format for weight calculation
        legacy_dict = self.to_legacy_format()
        
        # Calculate weights (this returns string keys)
        string_weights = calculate_weights(legacy_dict, temperature)
        
        # Convert back to integer keys
        weights = {int(k): v for k, v in string_weights.items()}
        
        return weights
    
    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    
    def summary(self):
        """Print a summary of the managed structures"""
        print(f"StructureManager Summary:")
        print(f"  Total structures: {len(self.struct_ids)}")

        if self.struct_ids:
            # Only show ID range if all IDs are integers
            try:
                if all(isinstance(x, int) or (isinstance(x, str) and x.isdigit()) for x in self.struct_ids):
                    numeric_ids = [int(x) if isinstance(x, str) else x for x in self.struct_ids]
                    print(f"  ID range: {min(numeric_ids)} - {max(numeric_ids)}")
                else:
                    # Show sample IDs for string-based naming
                    sample_ids = list(self.struct_ids)[:5]
                    print(f"  Sample IDs: {', '.join(str(x) for x in sample_ids)}{' ...' if len(self.struct_ids) > 5 else ''}")
            except:
                # Fallback: just show first few IDs
                sample_ids = list(self.struct_ids)[:5]
                print(f"  Sample IDs: {', '.join(str(x) for x in sample_ids)}{' ...' if len(self.struct_ids) > 5 else ''}")
            
            # Check data availability
            energies = self.load_energies()
            structures = self.load_structures()
            
            print(f"  Energies loaded: {len(energies)}")
            print(f"  Structures loaded: {len(structures)}")
            
            if energies:
                stats = self.get_energy_statistics()
                print(f"  Energy range: {stats['min']:.3f} - {stats['max']:.3f} Ry")
                print(f"  Energy mean: {stats['mean']:.3f} ± {stats['std']:.3f} Ry")
    
    def __len__(self):
        """Return number of managed structures"""
        return len(self.struct_ids)
    
    def __repr__(self):
        return f"StructureManager({len(self.struct_ids)} structures)"