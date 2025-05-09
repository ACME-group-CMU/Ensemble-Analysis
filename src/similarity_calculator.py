import numpy as np
import math
from pymatgen.core.structure import Molecule, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pymatgen.analysis.molecule_matcher as MM
from pymatgen.io.ase import AseAtomsAdaptor
import copy
import random

class BaseSimilarityCalculator:
    """Base class for structure similarity calculations"""
    
    def compare_structures(self, struct1, struct2):
        """Calculate similarity between two structures"""
        raise NotImplementedError("Subclasses must implement this method")


class LIBFPSimilarityCalculator(BaseSimilarityCalculator):
    """Calculate similarity using LIBFP fingerprints"""
    
    def __init__(self, max_radius=5):
        self.max_radius = max_radius
        
        try:
            import libfp
            self.libfp = libfp
        except ImportError:
            raise ImportError("libfp package is required for LIBFP similarity calculations")
    
    def compare_structures(self, struct1, struct2):
        """Calculate similarity using LIBFP fingerprints"""
        # Convert to libfp format
        cell1 = self.convert_pymatgen_to_libfp_cell(struct1)
        cell2 = self.convert_pymatgen_to_libfp_cell(struct2)
        
        # Get element types
        types1 = np.array(cell1[2])
        types2 = np.array(cell2[2])
        
        if set(types1) != set(types2):
            return 1000.0  # High value for different element types
        
        # Choose the structure with more atoms for type representation
        types_representative = types1
        if len(types2) > len(types1):
            types_representative = types2
        
        # Calculate fingerprints and distance
        fp1 = self.libfp.get_lfp(cell1, cutoff=self.max_radius, log=True, orbital='s')
        fp2 = self.libfp.get_lfp(cell2, cutoff=self.max_radius, log=True, orbital='s')
        
        dist = self.libfp.get_fp_dist(fp1, fp2, types_representative)
        return dist
    
    def convert_pymatgen_to_libfp_cell(self, struct_pm):
        """Convert pymatgen structure to libfp cell format"""
        # Convert to ASE format first
        adaptor = AseAtomsAdaptor()
        struct_ase = adaptor.get_atoms(struct_pm)
        
        # Extract cell data
        cell_array = struct_ase.get_cell().array
        rxyz = struct_ase.get_positions()
        types_temp = struct_ase.get_atomic_numbers()
        
        # Convert types to libfp format
        types_group_index = []
        types_z = []
        
        i = 0
        for new_type in types_temp:
            if new_type not in types_z:
                i += 1
                types_z.append(new_type)
            types_group_index.append(i)
        
        cell = (cell_array, rxyz, np.array(types_group_index), np.array(types_z))
        return cell


class MoleculeComparer:
    """Helper class to compare molecules using RMSD"""
    
    def get_rmsd(self, mol1, mol2):
        """Calculate RMSD between two molecules using Hungarian matching"""
        matcher = MM.HungarianOrderMatcher(mol1)
        
        try:
            result = matcher.fit(mol2)
            rmsd = result[1]
            return rmsd
        except Exception:
            return 10000.0  # High value if matching fails


class LEVRSimilarityCalculator(BaseSimilarityCalculator):
    """Calculate similarity using Local Environment Variable Radius approach"""
    
    def __init__(self, required_neighbors=6, fit_tol=0.001):
        self.required_neighbors = required_neighbors
        self.fit_tol = fit_tol
        self.site_compare_engine = MoleculeComparer()
    
    def compare_structures(self, struct1, struct2, iterations=10, random_seed=None, debug=False):
        """Calculate LEVR similarity between two structures"""
        # Find distinct sites in each structure
        struct1_distinct, struct1_equi = self.find_distinct_sites(struct1)
        struct2_distinct, struct2_equi = self.find_distinct_sites(struct2)
        
        # Initialize matching data
        struct1_matched_atoms = [[] for _ in range(len(struct1_distinct))]
        struct2_matched_atoms = [[] for _ in range(len(struct2_distinct))]
        
        # Default value for sites that don't match
        bad_match_val = 10000
        
        # Compare each distinct site in structure 1 with each in structure 2
        for s1_idx, struct1_site_idx in enumerate(struct1_distinct):
            for s2_idx, struct2_site_idx in enumerate(struct2_distinct):
                # Define molecules for comparison
                s1_mol_atoms = self.define_atoms_in_molecule(struct1, struct1_site_idx)
                s2_mol_atoms = self.define_atoms_in_molecule(struct2, struct2_site_idx)
                
                struct1_mol = Molecule.from_sites(s1_mol_atoms)
                struct2_mol = Molecule.from_sites(s2_mol_atoms)
                
                # Calculate RMSD between molecules
                try:
                    new_rmsd = self.site_compare_engine.get_rmsd(struct1_mol, struct2_mol)
                except Exception as e:
                    if debug:
                        print(e)
                    new_rmsd = bad_match_val
                
                # Record matching data for both structures
                s2_sites_names = struct2_equi[s2_idx]
                s1_new_tuples = [(site_name, new_rmsd) for site_name in s2_sites_names]
                struct1_matched_atoms[s1_idx].extend(s1_new_tuples)
                
                s1_sites_names = struct1_equi[s1_idx]
                s2_new_tuples = [(site_name, new_rmsd) for site_name in s1_sites_names]
                struct2_matched_atoms[s2_idx].extend(s2_new_tuples)
        
        # Package results
        result_dict = {
            "s1": {
                "list_rmsd": struct1_matched_atoms,
                "equivalent_sites": struct1_equi
            },
            "s2": {
                "list_rmsd": struct2_matched_atoms,
                "equivalent_sites": struct2_equi
            }
        }
        
        # Find minimum matching value (optimized matching)
        final_rmsd = self.find_minimum_matching_value(result_dict, random_seed=random_seed, 
                                                     iterations=iterations, debug=debug)
        
        return final_rmsd
    
    def find_distinct_sites(self, structure):
        """Find distinct sites in a structure based on local environments"""
        # Use symmetry to find initial equivalent sites
        analyzer = SpacegroupAnalyzer(structure)
        sym_struct = analyzer.get_symmetrized_structure()
        global_equi_sets = sym_struct.equivalent_indices
        global_distinct_sites = [equi_set[0] for equi_set in global_equi_sets]
        
        # Generate molecules for each distinct site
        global_distinct_molecules = []
        for site_idx in global_distinct_sites:
            allowed_sites = self.define_atoms_in_molecule(structure, site_idx)
            new_molecule = Molecule.from_sites(allowed_sites)
            global_distinct_molecules.append(new_molecule)
        
        # Find local equivalence based on molecule similarity
        all_indices_matched = []
        local_equi_sets = []
        
        for idx1, site_idx1 in enumerate(global_distinct_sites):
            if site_idx1 in all_indices_matched:
                continue
            
            all_indices_matched.append(site_idx1)
            local_equi_set = [site_idx1]
            primary_mol = global_distinct_molecules[idx1]
            
            for idx2, site_idx2 in enumerate(global_distinct_sites):
                if site_idx2 in all_indices_matched:
                    continue
                
                secondary_mol = global_distinct_molecules[idx2]
                
                try:
                    new_rmsd = self.site_compare_engine.get_rmsd(primary_mol, secondary_mol)
                    if new_rmsd < self.fit_tol:
                        local_equi_set.append(site_idx2)
                        all_indices_matched.append(site_idx2)
                except:
                    continue
            
            local_equi_sets.append(local_equi_set)
        
        # Expand to include all equivalent sites
        full_equivalent_sets = []
        for local_set in local_equi_sets:
            new_equi_set = []
            for idx in local_set:
                global_equivalents = [equi_set for equi_set in global_equi_sets if idx in equi_set]
                new_equi_set.extend(global_equivalents[0])
            full_equivalent_sets.append(list(set(new_equi_set)))
        
        local_distinct_sites = [equi_set[0] for equi_set in full_equivalent_sets]
        
        return local_distinct_sites, full_equivalent_sets
    
    def define_atoms_in_molecule(self, structure, origin_site_idx):
        """Define atoms in a molecule centered at a given site"""
        origin_site = structure.sites[origin_site_idx]
        
        # Find neighbors
        effective_radius = 4
        my_neighbor_sites = []
        while len(my_neighbor_sites) < self.required_neighbors:
            my_neighbor_sites = structure.get_neighbors(origin_site, effective_radius)
            effective_radius += 1
        
        # Limit to required number
        if len(my_neighbor_sites) > self.required_neighbors:
            my_neighbor_sites = my_neighbor_sites[:self.required_neighbors]
        
        # Add the origin site
        allowed_sites = my_neighbor_sites.copy()
        allowed_sites.append(origin_site)
        
        return allowed_sites
    
    def find_lcm(self, num1, num2):
        """Find the least common multiple of two numbers"""
        return abs(num1 * num2) // math.gcd(num1, num2)
    
    def convert_to_int(self, list_of_lists):
        """Convert all elements in nested lists to integers"""
        return [[int(element) for element in sublist] for sublist in list_of_lists]
    
    def get_distinct_site_fraction(self, equivalent_sites_indices):
        """Get the fraction of sites that belong to each distinct type"""
        distinct_site_sum = [len(eq_list) for eq_list in equivalent_sites_indices]
        all_atoms = sum(distinct_site_sum)
        distinct_fraction = [(ds_num / all_atoms) for ds_num in distinct_site_sum]
        return distinct_fraction
    
    def expand_rmsds_by_multiplier(self, s1_lists_of_rmsds, s2_multiplier, s2_max_val, debug=False):
        """Expand RMSD lists to account for structure size differences"""
        included_s2_sites = [new_tuple[0] for sublist in s1_lists_of_rmsds for new_tuple in sublist]
        included_s2_sites = list(set(included_s2_sites))
        included_s2_sites.sort()
        
        new_equivalents = [list(np.arange(s2_multiplier)*(1+s2_max_val)+original_site_num) 
                          for original_site_num in included_s2_sites]
        new_equivalents = self.convert_to_int(new_equivalents)
        
        s1_lists_of_rmsds_w_dupes = copy.deepcopy(s1_lists_of_rmsds)
        
        for list_idx, rmsd_list in enumerate(s1_lists_of_rmsds):
            list_of_duplicate_sites = []
            for rmsd_tuple in rmsd_list:
                old_site_num, rmsd = rmsd_tuple
                matched_site_nums = [new_equivalent_list for new_equivalent_list in new_equivalents 
                                     if old_site_num in new_equivalent_list][0]
                
                for matched_site_num in matched_site_nums:
                    if matched_site_num != old_site_num:
                        new_tuple = (matched_site_num, rmsd)
                        list_of_duplicate_sites.append(new_tuple)
            
            s1_lists_of_rmsds_w_dupes[list_idx].extend(list_of_duplicate_sites)
        
        return s1_lists_of_rmsds_w_dupes
    
    def perturb_list(self, original_list, perturbation_strength=0.3):
        """Randomly perturb a list by moving items around"""
        if not original_list or len(original_list) <= 1:
            return original_list.copy()
        
        perturbed = original_list.copy()
        list_length = len(perturbed)
        
        num_swaps = max(1, int(list_length * perturbation_strength))
        
        for _ in range(num_swaps):
            pos1 = random.randint(0, list_length - 1)
            pos2 = random.randint(0, list_length - 1)
            
            while pos1 == pos2 and list_length > 1:
                pos2 = random.randint(0, list_length - 1)
            
            perturbed[pos1], perturbed[pos2] = perturbed[pos2], perturbed[pos1]
        
        return perturbed
    
    def select_matches(self, list_of_rmsds, selections, random_seed=None, iterations=10, 
                       debug=False, max_match_per_selection=1000000, metropolis_selection=True):
        """Select optimal matches between sites"""
        # Helper function to remove tuples by index
        def remove_tuples_by_index(list_of_lists, indeces_to_remove):
            return [[tup for tup in sublist if tup[0] not in indeces_to_remove] for sublist in list_of_lists]
        
        selections_copy = copy.deepcopy(selections)
        num_atoms = len(selections)
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Sort each sublist by RMSD
        for sublist_idx, sublist in enumerate(list_of_rmsds):
            list_of_rmsds[sublist_idx] = sorted(sublist, key=lambda tup: tup[1])
        
        # Count allowed selections for each sublist
        allowed_selections_by_sublist = {}
        current_selections_by_sublist = {}
        
        sublist_indeces = list(set(selections))
        for sublist_idx in sublist_indeces:
            allowed_selections_by_sublist[sublist_idx] = selections.count(sublist_idx)
            current_selections_by_sublist[sublist_idx] = 0
        
        min_similarity = 10000000
        min_selection_list = False
        min_selection_list_full = False
        
        for i in range(iterations):
            if metropolis_selection and min_selection_list:
                perturbation_strength = (1-(i/iterations))
                selections_copy = self.perturb_list(min_selection_list_full, perturbation_strength)
            
            # Reset selection counters
            for sublist_idx in sublist_indeces:
                current_selections_by_sublist[sublist_idx] = 0
            
            sum_rmsd = 0
            final_selections = []
            selection_idx = -1
            list_of_rmsds_copy = copy.deepcopy(list_of_rmsds)
            match_count = 0
            
            while match_count < num_atoms:
                selection_idx += 1
                if selection_idx >= len(selections_copy):
                    break
                    
                selection = selections_copy[selection_idx]
                available_list = list_of_rmsds_copy[selection]
                
                if not available_list:
                    continue
                
                first_match_rmsd = False
                available_list_idx = 0
                newly_matched_atom_sites = []
                
                while (len(newly_matched_atom_sites) < max_match_per_selection and
                       current_selections_by_sublist[selection] < allowed_selections_by_sublist[selection]):
                    
                    if available_list_idx >= len(available_list):
                        break
                    
                    selected_atom_site, selected_rmsd = available_list[available_list_idx]
                    
                    if first_match_rmsd is False:
                        first_match_rmsd = selected_rmsd
                        newly_matched_atom_sites.append(selected_atom_site)
                        final_selections.append(selection)
                        current_selections_by_sublist[selection] += 1
                        match_count += 1
                        sum_rmsd += selected_rmsd
                        available_list_idx += 1
                        continue
                    
                    if selected_rmsd == first_match_rmsd:
                        newly_matched_atom_sites.append(selected_atom_site)
                        current_selections_by_sublist[selection] += 1
                        match_count += 1
                        sum_rmsd += selected_rmsd
                        available_list_idx += 1
                    else:
                        break
                
                list_of_rmsds_copy = remove_tuples_by_index(list_of_rmsds_copy, newly_matched_atom_sites)
            
            # Calculate average similarity
            if match_count > 0:
                new_similarity = sum_rmsd / match_count
                
                if new_similarity < min_similarity:
                    min_similarity = new_similarity
                    min_selection_list = copy.deepcopy(final_selections)
                    min_selection_list_full = copy.deepcopy(selections_copy)
        
        return min_similarity
    
    def find_minimum_matching_value(self, result_dict, random_seed=None, iterations=10, debug=False):
        """Find the minimum matching value between structures"""
        s1_list_rmsd = result_dict['s1']['list_rmsd']
        s1_equivalent_sites = result_dict['s1']['equivalent_sites']
        s2_equivalent_sites = result_dict['s2']['equivalent_sites']
        
        # Count total sites
        s1_count = sum(len(sites) for sites in s1_equivalent_sites)
        s2_count = sum(len(sites) for sites in s2_equivalent_sites)
        
        # Find least common multiple for fair comparison
        lcm = self.find_lcm(s1_count, s2_count)
        s1_multiplier = lcm / s1_count
        s2_multiplier = lcm / s2_count
        
        # Expand RMSD lists for fair comparison
        s2_all_sites = [site for equi_list in s2_equivalent_sites for site in equi_list]
        s2_max_val = max(s2_all_sites)
        expanded_rmsds = self.expand_rmsds_by_multiplier(s1_list_rmsd, s2_multiplier, s2_max_val, debug)
        
        # Create selection choices
        selection_choices = []
        for idx, equi_set in enumerate(s1_equivalent_sites):
            selections = [idx] * int(s1_multiplier) * len(equi_set)
            selection_choices.extend(selections)
        
        # Find optimal matching
        min_rmsd = self.select_matches(expanded_rmsds, selection_choices, 
                                      random_seed=random_seed, iterations=iterations, debug=debug)
        
        return min_rmsd