from pymatgen.core.structure import Structure
from collections import OrderedDict
import json

class StructureManager:
    """Class to manage structures and their associated energies"""
    
    def __init__(self):
        self.data = OrderedDict()  # Ordered dict mapping IDs to structure-energy pairs
    
    def add_structure(self, structure_id, structure, energy):
        """Add a single structure with energy"""
        self.data[structure_id] = {
            "structure": structure,
            "Energy (Ry)": energy
        }
    
    def add_structures_from_dict(self, structures_dict):
        """Add multiple structures from a dictionary of structure-energy pairs"""
        for structure_id, entry in structures_dict.items():
            if "structure" in entry and "Energy (Ry)" in entry:
                # Convert structure dict back to pymatgen Structure object if needed
                structure = entry["structure"]
                if not isinstance(structure, Structure):
                    structure = Structure.from_dict(structure)
                
                self.data[structure_id] = {
                    "structure": structure,
                    "Energy (Ry)": entry["Energy (Ry)"]
                }
    
    def get_structure(self, structure_id):
        """Get a structure by ID"""
        if structure_id in self.data:
            return self.data[structure_id]["structure"]
        return None
    
    def get_energy(self, structure_id):
        """Get energy for a structure by ID"""
        if structure_id in self.data:
            return self.data[structure_id]["Energy (Ry)"]
        return None
    
    def get_structure_energy_pair(self, structure_id):
        """Get structure-energy pair by ID"""
        return self.data.get(structure_id)
    
    def get_structure_ids(self):
        """Get list of all structure IDs"""
        return list(self.data.keys())
    
    def get_structures_sorted_by_energy(self):
        """Get structures sorted by energy (lowest first)"""
        sorted_ids = sorted(self.data.keys(), 
                           key=lambda x: self.data[x]["Energy (Ry)"])
        return [(id, self.data[id]) for id in sorted_ids]
