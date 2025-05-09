import numpy as np
import time
import pickle
import multiprocessing as mp
import networkx as nx
from tqdm import tqdm

class SimilarityAnalyzer:
    """Class to analyze similarities between structures"""
    
    def __init__(self, structure_manager, calculator_type='LEVR', **calculator_kwargs):
        """Initialize with a structure manager and calculator type"""
        self.structure_manager = structure_manager
        self.calculator_type = calculator_type
        
        # Initialize the appropriate calculator
        if calculator_type == 'LEVR':
            from src.similarity_calculator import LEVRSimilarityCalculator
            self.calculator = LEVRSimilarityCalculator(**calculator_kwargs)
        elif calculator_type == 'LIBFP':
            from src.similarity_calculator import LIBFPSimilarityCalculator
            self.calculator = LIBFPSimilarityCalculator(**calculator_kwargs)
        else:
            raise ValueError(f"Unknown calculator type: {calculator_type}")
        
        self.similarity_matrix = {}  # (id1, id2) -> similarity
        self.similarity_graph = nx.Graph()
    
    def calculate_all_similarities(self, parallel=True, n_jobs=None, batch_size=100):
        """Calculate similarities between all structures"""
        start_time = time.time()
        structure_ids = self.structure_manager.get_structure_ids()
        n_structures = len(structure_ids)
        
        # Calculate total number of comparisons
        total_comparisons = n_structures * (n_structures - 1) // 2
        
        print(f"Calculating similarities for {n_structures} structures ({total_comparisons} comparisons)")
        
        if parallel:
            self._calculate_similarities_parallel(structure_ids, n_jobs, batch_size)
        else:
            self._calculate_similarities_sequential(structure_ids)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Calculation completed in {duration:.2f} seconds")
        print(f"Average time per comparison: {(duration / max(1, len(self.similarity_matrix))):.4f} seconds")
        
        # Build similarity graph
        self._build_similarity_graph()
        
        return self.similarity_matrix
    
    def _calculate_similarities_sequential(self, structure_ids):
        """Calculate similarities sequentially"""
        # Create all pairs to calculate
        pairs = []
        for i, id1 in enumerate(structure_ids):
            for j in range(i+1, len(structure_ids)):
                id2 = structure_ids[j]
                if not self._is_similarity_calculated(id1, id2):
                    pairs.append((id1, id2))
        
        # Calculate similarities with progress bar
        for id1, id2 in tqdm(pairs, desc="Calculating similarities"):
            similarity = self._calculate_single_similarity(id1, id2)
            self._update_similarity_matrix(id1, id2, similarity)
    
    def _calculate_similarities_parallel(self, structure_ids, n_jobs=None, batch_size=100):
        """Calculate similarities in parallel"""
        if n_jobs is None:
            n_jobs = max(1, mp.cpu_count() - 1)
        
        print(f"Using {n_jobs} parallel jobs")
        
        # Generate all pairs to calculate
        all_pairs = []
        for i, id1 in enumerate(structure_ids):
            for j in range(i+1, len(structure_ids)):
                id2 = structure_ids[j]
                if not self._is_similarity_calculated(id1, id2):
                    all_pairs.append((id1, id2))
        
        # Calculate in batches with progress bar
        with tqdm(total=len(all_pairs), desc="Calculating similarities") as pbar:
            for batch_start in range(0, len(all_pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(all_pairs))
                batch = all_pairs[batch_start:batch_end]
                
                # Process batch in parallel
                with mp.Pool(processes=n_jobs) as pool:
                    results = pool.map(self._similarity_worker, 
                                       [(id1, id2, self.calculator_type) for id1, id2 in batch])
                
                # Update similarity matrix with results
                for (id1, id2), similarity in results:
                    self._update_similarity_matrix(id1, id2, similarity)
                
                pbar.update(len(batch))
    
    def _similarity_worker(self, args):
        """Worker function for parallel calculation"""
        id1, id2, calc_type = args
        
        # We need to recreate the calculator in each worker
        if calc_type == 'LEVR':
            from similarity_calculator import LEVRSimilarityCalculator
            calculator = LEVRSimilarityCalculator(required_neighbors=self.calculator.required_neighbors)
        else:
            from similarity_calculator import LIBFPSimilarityCalculator
            calculator = LIBFPSimilarityCalculator(max_radius=self.calculator.max_radius)
        
        struct1 = self.structure_manager.get_structure(id1)
        struct2 = self.structure_manager.get_structure(id2)
        
        try:
            similarity = calculator.compare_structures(struct1, struct2)
        except Exception as e:
            print(f"Error calculating similarity between {id1} and {id2}: {e}")
            similarity = 1000.0
        
        return ((id1, id2), similarity)
    
    def _calculate_single_similarity(self, id1, id2):
        """Calculate similarity between two structures"""
        struct1 = self.structure_manager.get_structure(id1)
        struct2 = self.structure_manager.get_structure(id2)
        
        try:
            similarity = self.calculator.compare_structures(struct1, struct2)
        except Exception as e:
            print(f"Error calculating similarity between {id1} and {id2}: {e}")
            similarity = 1000.0
        
        return similarity
    
    def _is_similarity_calculated(self, id1, id2):
        """Check if similarity has already been calculated"""
        key = (min(id1, id2), max(id1, id2))
        return key in self.similarity_matrix
    
    def _update_similarity_matrix(self, id1, id2, similarity):
        """Update similarity matrix with a new result"""
        key = (min(id1, id2), max(id1, id2))
        self.similarity_matrix[key] = similarity
    
    def _build_similarity_graph(self):
        """Build a NetworkX graph from similarity matrix"""
        self.similarity_graph.clear()
        
        # Add nodes
        structure_ids = self.structure_manager.get_structure_ids()
        self.similarity_graph.add_nodes_from(structure_ids)
        
        # Add edges
        for (id1, id2), similarity in self.similarity_matrix.items():
            # Lower similarity = higher weight (more similar)
            inv_similarity = 1.0 / (similarity + 1e-6)  # Avoid division by zero
            self.similarity_graph.add_edge(id1, id2, weight=inv_similarity, 
                                          similarity=similarity)
    
    def get_similarity(self, id1, id2):
        """Get similarity between two structures"""
        key = (min(id1, id2), max(id1, id2))
        return self.similarity_matrix.get(key)
    
    def find_similar_structures(self, structure_id, threshold=5.0):
        """Find structures similar to a given structure"""
        similar_structures = []
        
        for (id1, id2), similarity in self.similarity_matrix.items():
            if id1 == structure_id and similarity <= threshold:
                similar_structures.append((id2, similarity))
            elif id2 == structure_id and similarity <= threshold:
                similar_structures.append((id1, similarity))
        
        # Sort by similarity (ascending)
        similar_structures.sort(key=lambda x: x[1])
        
        return similar_structures
    
    def find_structure_communities(self, resolution=1.0):
        """Find communities of similar structures"""
        # Ensure graph is built
        if len(self.similarity_graph.edges) == 0:
            self._build_similarity_graph()
        
        # Find communities using Louvain algorithm
        try:
            communities = nx.community.louvain_communities(
                self.similarity_graph, weight='weight', resolution=resolution
            )
            return list(communities)
        except:
            print("Warning: Could not find communities")
            return []
    
    def save_similarity_matrix(self, filename):
        """Save similarity matrix to file"""
        data = {
            'similarity_matrix': self.similarity_matrix,
            'calculator_type': self.calculator_type
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Similarity matrix saved to {filename}")
    
    def load_similarity_matrix(self, filename):
        """Load similarity matrix from file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.similarity_matrix = data.get('similarity_matrix', {})
            
            # Rebuild the graph
            self._build_similarity_graph()
            
            print(f"Loaded {len(self.similarity_matrix)} similarity values")
            return True
        except Exception as e:
            print(f"Error loading similarity matrix: {e}")
            return False
    
    def export_graph(self, filename):
        """Export similarity graph to GraphML format"""
        nx.write_graphml(self.similarity_graph, filename)
        print(f"Graph exported to {filename}")