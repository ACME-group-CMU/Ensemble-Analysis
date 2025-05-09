import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import argparse
import os
from tqdm import tqdm

def load_similarity_matrix(file_path):
    """Load numpy similarity matrix from file."""
    print(f"Loading similarity matrix from {file_path}...")
    sim_matrix = np.load(file_path)
    print(f"Loaded matrix with shape: {sim_matrix.shape}")
    return sim_matrix

def plot_similarity_distribution(sim_matrix, output_dir):
    """Plot the distribution of similarity values."""
    plt.figure(figsize=(10, 6))
    
    # Convert matrix to 1D array, excluding diagonal elements (self-similarity)
    sim_values = sim_matrix.flatten()
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool).flatten()
    sim_values = sim_values[mask]
    
    sns.histplot(sim_values, bins=50, kde=True)
    plt.title('Distribution of Similarity Values')
    plt.xlabel('Similarity Value')
    plt.ylabel('Frequency')
    plt.axvline(np.median(sim_values), color='r', linestyle='--', label=f'Median: {np.median(sim_values):.3f}')
    plt.axvline(np.mean(sim_values), color='g', linestyle='--', label=f'Mean: {np.mean(sim_values):.3f}')
    plt.legend()
    
    output_file = os.path.join(output_dir, 'similarity_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity distribution to {output_file}")
    
    return {
        'min': np.min(sim_values),
        'max': np.max(sim_values),
        'mean': np.mean(sim_values),
        'median': np.median(sim_values),
        'std': np.std(sim_values)
    }

def plot_similarity_heatmap(sim_matrix, output_dir, max_size=100):
    """Plot a heatmap of the similarity matrix (potentially downsampled)."""
    if sim_matrix.shape[0] > max_size:
        # If matrix is too large, sample a subset
        indices = np.linspace(0, sim_matrix.shape[0]-1, max_size, dtype=int)
        sample_matrix = sim_matrix[np.ix_(indices, indices)]
        print(f"Matrix too large, sampled {max_size}x{max_size} subset for heatmap")
    else:
        sample_matrix = sim_matrix
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sample_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('Similarity Matrix Heatmap')
    
    output_file = os.path.join(output_dir, 'similarity_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity heatmap to {output_file}")

def plot_reduced_dimensions(sim_matrix, output_dir, method='tsne', perplexity=30, random_state=42):
    """Create a reduced dimensionality plot using t-SNE or MDS."""
    # Convert similarity matrix to distance matrix if needed
    # For libfp, smaller values indicate more similarity, so we can use directly
    n_samples = sim_matrix.shape[0]
    
    # If the matrix is too large, use PCA first to reduce to 50 dimensions
    if n_samples > 5000:
        print("Large matrix detected, applying PCA first...")
        pca = PCA(n_components=50, random_state=random_state)
        sim_matrix_reduced = pca.fit_transform(sim_matrix)
    else:
        sim_matrix_reduced = sim_matrix
    
    print(f"Applying {method.upper()} dimensionality reduction...")
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(perplexity, n_samples-1), 
                       random_state=random_state, metric='precomputed' if method == 'mds' else 'euclidean')
    elif method.lower() == 'mds':
        reducer = MDS(n_components=2, random_state=random_state, 
                      dissimilarity='precomputed', n_jobs=-1)
    
    embedding = reducer.fit_transform(sim_matrix_reduced)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=5)
    plt.title(f'{method.upper()} Visualization of Similarity Space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    output_file = os.path.join(output_dir, f'{method.lower()}_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {method.upper()} visualization to {output_file}")

def find_most_similar_pairs(sim_matrix, top_n=20):
    """Find the top N most similar structure pairs."""
    n = sim_matrix.shape[0]
    
    # Create a mask to exclude diagonal elements (self-similarity)
    mask = ~np.eye(n, dtype=bool)
    masked_sim = sim_matrix.copy()
    
    # For libfp, smaller values indicate more similarity
    # Set diagonal to infinity so they're not selected
    np.fill_diagonal(masked_sim, np.inf)
    
    # Find the indices of the smallest n values
    most_similar_pairs = []
    
    for _ in range(top_n):
        # Find the minimum value and its indices
        min_idx = np.unravel_index(np.argmin(masked_sim), masked_sim.shape)
        sim_value = sim_matrix[min_idx]
        
        # Add to our list
        most_similar_pairs.append((min_idx[0], min_idx[1], sim_value))
        
        # Mark this pair as processed by setting to infinity
        masked_sim[min_idx[0], min_idx[1]] = np.inf
        masked_sim[min_idx[1], min_idx[0]] = np.inf
    
    return most_similar_pairs

def plot_combined_similarity_distribution(low_sim_matrix, high_sim_matrix, output_dir):
    """Plot the distribution of similarity values for both matrices."""
    plt.figure(figsize=(12, 7))
    
    # Process low energy matrix
    low_sim_values = low_sim_matrix.flatten()
    low_mask = ~np.eye(low_sim_matrix.shape[0], dtype=bool).flatten()
    low_sim_values = low_sim_values[low_mask]
    
    # Process high energy matrix
    high_sim_values = high_sim_matrix.flatten()
    high_mask = ~np.eye(high_sim_matrix.shape[0], dtype=bool).flatten()
    high_sim_values = high_sim_values[high_mask]
    
    # Plot both distributions
    sns.histplot(low_sim_values, bins=50, kde=True, color='blue', alpha=0.6, label='Low Energy')
    sns.histplot(high_sim_values, bins=50, kde=True, color='red', alpha=0.6, label='High Energy')
    
    # Add vertical lines for means and medians
    plt.axvline(np.median(low_sim_values), color='blue', linestyle='--', 
                label=f'Low Energy Median: {np.median(low_sim_values):.3f}')
    plt.axvline(np.mean(low_sim_values), color='blue', linestyle='-', 
                label=f'Low Energy Mean: {np.mean(low_sim_values):.3f}')
    
    plt.axvline(np.median(high_sim_values), color='red', linestyle='--', 
                label=f'High Energy Median: {np.median(high_sim_values):.3f}')
    plt.axvline(np.mean(high_sim_values), color='red', linestyle='-', 
                label=f'High Energy Mean: {np.mean(high_sim_values):.3f}')
    
    plt.title('Distribution of Similarity Values by Energy Group')
    plt.xlabel('Similarity Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    output_file = os.path.join(output_dir, 'combined_similarity_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined similarity distribution to {output_file}")

def plot_combined_reduced_dimensions(low_sim_matrix, high_sim_matrix, output_dir, method='tsne', perplexity=30, random_state=42):
    """Create a combined reduced dimensionality plot using t-SNE or MDS."""
    
    # Function to prepare and transform a similarity matrix
    def transform_matrix(sim_matrix, reducer=None):
        n_samples = sim_matrix.shape[0]
        
        # If the matrix is too large, use PCA first to reduce dimensions
        if n_samples > 5000:
            print(f"Large matrix detected ({n_samples} samples), applying PCA first...")
            pca = PCA(n_components=50, random_state=random_state)
            sim_matrix_reduced = pca.fit_transform(sim_matrix)
        else:
            sim_matrix_reduced = sim_matrix
        
        # If reducer is provided, use it, otherwise create a new one
        if reducer is None:
            if method.lower() == 'tsne':
                reducer = TSNE(n_components=2, perplexity=min(perplexity, n_samples-1), 
                               random_state=random_state, metric='precomputed' if method == 'mds' else 'euclidean')
            elif method.lower() == 'mds':
                reducer = MDS(n_components=2, random_state=random_state, 
                              dissimilarity='precomputed', n_jobs=-1)
            embedding = reducer.fit_transform(sim_matrix_reduced)
            return embedding, reducer
        else:
            # Use the same reducer for consistent transformation
            embedding = reducer.fit_transform(sim_matrix_reduced)
            return embedding
    
    print(f"Applying {method.upper()} dimensionality reduction for both matrices...")
    
    # Option 1: Transform matrices independently (creates separate clusters)
    low_embedding, _ = transform_matrix(low_sim_matrix)
    high_embedding, _ = transform_matrix(high_sim_matrix)
    
    # Create figure for separate transformations
    plt.figure(figsize=(12, 10))
    plt.scatter(low_embedding[:, 0], low_embedding[:, 1], alpha=0.7, s=10, color='blue', label='Low Energy')
    plt.scatter(high_embedding[:, 0], high_embedding[:, 1], alpha=0.7, s=10, color='red', label='High Energy')
    plt.title(f'{method.upper()} Visualization - Independent Transformations')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    
    output_file = os.path.join(output_dir, f'combined_{method.lower()}_independent.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved independently transformed {method.upper()} visualization to {output_file}")
    
    # Option 2: Combine matrices for joint transformation (better for comparison)
    # First, ensure matrices have same dimensions by padding if necessary
    low_n = low_sim_matrix.shape[0]
    high_n = high_sim_matrix.shape[0]
    
    if low_n != high_n:
        print(f"Matrices have different dimensions: low={low_n}, high={high_n}")
        print("Using joint transformation on combined data...")
        
        # Create combined matrix by stacking all data points
        combined_data = np.vstack([
            low_sim_matrix if low_n <= high_n else low_sim_matrix[:high_n, :high_n],
            high_sim_matrix if high_n <= low_n else high_sim_matrix[:low_n, :low_n]
        ])
        
        # Apply dimensionality reduction to combined data
        combined_embedding, _ = transform_matrix(combined_data)
        
        # Split result back into low and high energy groups
        max_n = min(low_n, high_n)
        low_embedding_joint = combined_embedding[:max_n]
        high_embedding_joint = combined_embedding[max_n:max_n*2]
    else:
        # Matrices are the same size, can be combined directly
        print("Creating joint transformation of both matrices...")
        combined_embedding, _ = transform_matrix(np.vstack([low_sim_matrix, high_sim_matrix]))
        low_embedding_joint = combined_embedding[:low_n]
        high_embedding_joint = combined_embedding[low_n:]
    
    # Create figure for joint transformation
    plt.figure(figsize=(12, 10))
    plt.scatter(low_embedding_joint[:, 0], low_embedding_joint[:, 1], alpha=0.7, s=10, color='blue', label='Low Energy')
    plt.scatter(high_embedding_joint[:, 0], high_embedding_joint[:, 1], alpha=0.7, s=10, color='red', label='High Energy')
    plt.title(f'{method.upper()} Visualization - Joint Transformation')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    
    output_file = os.path.join(output_dir, f'combined_{method.lower()}_joint.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved jointly transformed {method.upper()} visualization to {output_file}")


def main():
    return

if __name__ == "__main__":
    main()