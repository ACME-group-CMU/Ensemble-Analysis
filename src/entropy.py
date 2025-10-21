#!/usr/bin/env python3
"""
VASP QUESTS Entropy Analysis Functions
======================================

Functions to compute information entropy on VASP structures using QUESTS.
Designed to integrate with existing ensemble analysis pipeline.

Usage:
    from quests_analysis import compute_entropy_analysis, entropy_sampling

Author: Assistant
Based on: Schwalbe-Koda et al. 2024 - Model-free quantification of completeness...
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Existing imports from your pipeline
from src.data_management_v2 import load_structures, load_energies

# QUESTS imports (actual API)
try:
    from quests import entropy, descriptor, matrix
    from quests.tools import plotting
except ImportError:
    print("QUESTS not found. Install with: pip install git+https://github.com/dskoda/quests.git")
    raise


def compute_entropy_analysis(struct_ids, k_neighbors=32, cutoff=5.0, bandwidth=0.015, 
                           save_descriptors=False):
    """
    Compute QUESTS information entropy analysis for a set of structures.
    
    Parameters:
    -----------
    struct_ids : list of int
        Structure IDs to analyze (should exist in your data pipeline)
    k_neighbors : int
        Number of nearest neighbors for descriptor (default: 32)
    cutoff : float  
        Cutoff radius in Angstroms (default: 5.0)
    bandwidth : float
        Bandwidth for KDE in A^-1 (default: 0.015) 
    save_descriptors : bool
        Whether to save computed descriptors
        
    Returns:
    --------
    dict : Results containing entropy values, descriptors, etc.
    """
    print(f"Computing QUESTS entropy analysis for {len(struct_ids)} structures")
    print(f"Parameters: k={k_neighbors}, cutoff={cutoff}Å, bandwidth={bandwidth}Å⁻¹")
    
    # Load structures using existing pipeline
    print("Loading structures...")
    structures_dict = load_structures(struct_ids)
    valid_ids = [sid for sid in struct_ids if sid in structures_dict]
    
    if not valid_ids:
        raise ValueError("No valid structures found!")
    
    print(f"Loaded {len(valid_ids)} valid structures")
    
    # Convert pymatgen structures to format QUESTS expects
    print("Converting structures for QUESTS...")
    from pymatgen.io.ase import AseAtomsAdaptor
    
    atoms_list = []
    adaptor = AseAtomsAdaptor()
    
    for sid in valid_ids:
        structure = structures_dict[sid]
        # Convert pymatgen Structure to ASE Atoms (QUESTS expects ASE format)
        atoms = adaptor.get_atoms(structure)
        atoms_list.append(atoms)
    
    # Compute descriptors using QUESTS
    print("Computing atomic environment descriptors...")
    descriptors = descriptor.get_descriptors(
        atoms_list, 
        k=k_neighbors, 
        cutoff=cutoff,
        concat=True
    )
    print(f"Computed descriptors shape: {descriptors.shape}")
    
    # Compute total entropy
    print("Computing information entropy...")
    H_total = entropy.entropy(descriptors, h=bandwidth)
    
    # Compute differential entropy for each structure
    print("Computing differential entropies...")
    delta_H_values = []
    
    # Get number of environments per structure to split descriptors
    env_counts = [len(atoms) for atoms in atoms_list]
    start_idx = 0
    
    for i, (struct_id, count) in enumerate(tqdm(zip(valid_ids, env_counts), desc="Computing δH")):
        # Get descriptors for this structure's environments
        end_idx = start_idx + count
        struct_descriptors = descriptors[start_idx:end_idx]
        
        # Get reference descriptors (all others)
        if len(descriptors) > count:
            other_indices = list(range(len(descriptors)))
            struct_indices = list(range(start_idx, end_idx))
            ref_indices = [idx for idx in other_indices if idx not in struct_indices]
            reference_descriptors = descriptors[ref_indices]
            
            # Compute differential entropy for this structure
            delta_H = entropy.delta_entropy(
                struct_descriptors, 
                reference_descriptors, 
                h=bandwidth
            )
            delta_H_values.append(np.mean(delta_H))  # Average over environments
        else:
            delta_H_values.append(0.0)
        
        start_idx = end_idx
    
    # Load energies for correlation analysis
    energies_dict = load_energies(valid_ids)
    energies = [energies_dict.get(sid, np.nan) for sid in valid_ids]
    
    # Compile results
    results = {
        'struct_ids': valid_ids,
        'total_entropy': H_total,
        'differential_entropies': np.array(delta_H_values),
        'energies': np.array(energies),
        'descriptors': descriptors if save_descriptors else None,
        'parameters': {
            'k_neighbors': k_neighbors,
            'cutoff': cutoff,
            'bandwidth': bandwidth,
            'n_structures': len(valid_ids),
            'descriptor_dim': descriptors.shape[1] if descriptors is not None else None
        }
    }
    
    # Create summary DataFrame
    results['summary_df'] = pd.DataFrame({
        'struct_id': valid_ids,
        'delta_H': delta_H_values,
        'energy': energies,
        'novelty_rank': np.argsort(np.argsort(-np.array(delta_H_values))) + 1
    })
    
    print(f"Analysis complete!")
    print(f"Total entropy: {H_total:.3f} nats")
    print(f"Mean δH: {np.mean(delta_H_values):.3f} ± {np.std(delta_H_values):.3f} nats")
    
    return results


def entropy_sampling(struct_ids, sample_size=100, method='high_entropy', **entropy_kwargs):
    """
    Sample structures based on entropy criteria using QUESTS.
    
    Parameters:
    -----------
    struct_ids : list of int
        Structure IDs to sample from
    sample_size : int
        Number of structures to sample
    method : str
        Sampling method: 'high_entropy', 'low_entropy', 'diverse_entropy'
    **entropy_kwargs : dict
        Additional arguments for compute_entropy_analysis()
        
    Returns:
    --------
    numpy.ndarray : Sampled structure IDs
    """
    print(f"Entropy-based sampling: {method}, n={sample_size}")
    
    # Compute entropy analysis
    results = compute_entropy_analysis(struct_ids, **entropy_kwargs)
    delta_H = results['differential_entropies']
    valid_ids = results['struct_ids']
    
    if sample_size > len(valid_ids):
        print(f"Warning: sample_size {sample_size} > available structures {len(valid_ids)}")
        sample_size = len(valid_ids)
    
    if method == 'high_entropy':
        # Sample structures with highest differential entropy (most novel)
        indices = np.argsort(delta_H)[-sample_size:]
        
    elif method == 'low_entropy':
        # Sample structures with lowest differential entropy (most representative)
        indices = np.argsort(delta_H)[:sample_size]
        
    elif method == 'diverse_entropy':
        # Sample across entropy range for diversity
        indices = np.linspace(0, len(delta_H)-1, sample_size, dtype=int)
        indices = np.argsort(delta_H)[indices]
        
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    sampled_ids = np.array([valid_ids[i] for i in indices])
    
    print(f"Sampled {len(sampled_ids)} structures")
    print(f"δH range: {delta_H[indices].min():.3f} to {delta_H[indices].max():.3f} nats")
    
    return np.sort(sampled_ids)


def identify_outliers(struct_ids, threshold_sigma=2.0, **entropy_kwargs):
    """
    Identify structural outliers using differential entropy.
    
    Parameters:
    -----------
    struct_ids : list of int
        Structure IDs to analyze
    threshold_sigma : float
        Number of standard deviations above mean to consider outlier
    **entropy_kwargs : dict
        Additional arguments for compute_entropy_analysis()
        
    Returns:
    --------
    dict : Outlier analysis results
    """
    print(f"Identifying outliers (threshold: {threshold_sigma}σ)")
    
    # Compute entropy analysis
    results = compute_entropy_analysis(struct_ids, **entropy_kwargs)
    delta_H = results['differential_entropies']
    valid_ids = results['struct_ids']
    
    # Define outlier threshold
    mean_dH = np.mean(delta_H)
    std_dH = np.std(delta_H)
    threshold = mean_dH + threshold_sigma * std_dH
    
    # Identify outliers
    outlier_mask = delta_H > threshold
    outlier_ids = np.array(valid_ids)[outlier_mask]
    outlier_entropies = delta_H[outlier_mask]
    
    print(f"Found {len(outlier_ids)} outliers")
    print(f"Threshold: {threshold:.3f} nats")
    
    return {
        'outlier_ids': outlier_ids,
        'outlier_entropies': outlier_entropies,
        'threshold': threshold,
        'mean_entropy': mean_dH,
        'std_entropy': std_dH,
        'all_results': results
    }


def entropy_learning_curve(struct_ids, sample_sizes=None, n_trials=3, **entropy_kwargs):
    """
    Compute learning curve showing how entropy scales with dataset size.
    
    Parameters:
    -----------
    struct_ids : list of int
        Structure IDs to analyze
    sample_sizes : list of int
        Sample sizes to test (default: logarithmic spacing)
    n_trials : int
        Number of trials per sample size
    **entropy_kwargs : dict
        Additional arguments for compute_entropy_analysis()
        
    Returns:
    --------
    dict : Learning curve results
    """
    if sample_sizes is None:
        max_size = min(len(struct_ids), 500)  # Cap at 500 for speed
        sample_sizes = np.logspace(1, np.log10(max_size), 8, dtype=int)
        sample_sizes = np.unique(np.clip(sample_sizes, 10, max_size))
    
    print(f"Computing entropy learning curve for {len(sample_sizes)} sample sizes")
    
    results = {
        'sample_sizes': [],
        'mean_entropies': [],
        'std_entropies': [],
        'all_entropies': []
    }
    
    for size in tqdm(sample_sizes, desc="Sample sizes"):
        size_entropies = []
        
        for trial in range(n_trials):
            # Random sample
            sample_ids = np.random.choice(struct_ids, size=size, replace=False)
            
            # Compute entropy
            trial_results = compute_entropy_analysis(sample_ids.tolist(), **entropy_kwargs)
            size_entropies.append(trial_results['total_entropy'])
        
        results['sample_sizes'].append(size)
        results['mean_entropies'].append(np.mean(size_entropies))
        results['std_entropies'].append(np.std(size_entropies))
        results['all_entropies'].append(size_entropies)
    
    return results


def plot_entropy_analysis(results, figsize=(12, 8)):
    """
    Plot entropy analysis results.
    
    Parameters:
    -----------
    results : dict
        Results from compute_entropy_analysis()
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Differential entropy distribution
    axes[0,0].hist(results['differential_entropies'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel('Differential Entropy δH (nats)')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Distribution of Differential Entropies')
    axes[0,0].axvline(np.mean(results['differential_entropies']), color='red', 
                     linestyle='--', label='Mean')
    axes[0,0].legend()
    
    # Plot 2: Energy vs entropy correlation
    valid_energies = ~np.isnan(results['energies'])
    if np.any(valid_energies):
        axes[0,1].scatter(results['energies'][valid_energies], 
                         results['differential_entropies'][valid_energies], alpha=0.6)
        axes[0,1].set_xlabel('Energy')
        axes[0,1].set_ylabel('Differential Entropy δH (nats)')
        axes[0,1].set_title('Energy vs Differential Entropy')
        
        # Correlation coefficient
        corr = np.corrcoef(results['energies'][valid_energies], 
                          results['differential_entropies'][valid_energies])[0,1]
        axes[0,1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0,1].transAxes,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Structure novelty ranking
    novelty_ranks = np.argsort(np.argsort(-results['differential_entropies'])) + 1
    axes[1,0].scatter(novelty_ranks, results['differential_entropies'], alpha=0.6)
    axes[1,0].set_xlabel('Novelty Rank')
    axes[1,0].set_ylabel('Differential Entropy δH (nats)')
    axes[1,0].set_title('Structure Novelty Ranking')
    
    # Plot 4: Summary statistics
    axes[1,1].axis('off')
    stats_text = f"""
    Analysis Summary
    ================
    Total Structures: {len(results['struct_ids'])}
    Total Entropy: {results['total_entropy']:.3f} nats
    
    Differential Entropy:
      Mean: {np.mean(results['differential_entropies']):.3f} nats
      Std:  {np.std(results['differential_entropies']):.3f} nats
      Min:  {np.min(results['differential_entropies']):.3f} nats
      Max:  {np.max(results['differential_entropies']):.3f} nats
    
    Parameters:
      k_neighbors: {results['parameters']['k_neighbors']}
      cutoff: {results['parameters']['cutoff']} Å
      bandwidth: {results['parameters']['bandwidth']} Å⁻¹
    """
    axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes,
                  fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    return fig


def compare_entropy_sampling_methods(struct_ids, sample_size=100, **entropy_kwargs):
    """
    Compare different entropy-based sampling methods.
    
    Parameters:
    -----------
    struct_ids : list of int
        Structure IDs to sample from
    sample_size : int
        Number of structures to sample with each method
    **entropy_kwargs : dict
        Additional arguments for entropy analysis
        
    Returns:
    --------
    dict : Comparison results
    """
    print(f"Comparing entropy sampling methods (n={sample_size})")
    
    methods = ['high_entropy', 'low_entropy', 'diverse_entropy']
    results = {}
    
    for method in methods:
        print(f"\nSampling with method: {method}")
        sampled_ids = entropy_sampling(struct_ids, sample_size, method, **entropy_kwargs)
        results[method] = sampled_ids
    
    return results


def plot_entropy_hexgrid(results, figsize=(8, 6)):
    """Plot energy vs differential entropy using seaborn hexgrid."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Get valid data points
    valid_mask = ~np.isnan(results['energies'])
    x = results['energies'][valid_mask]
    y = results['differential_entropies'][valid_mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create hexbin plot with marginal histograms
    g = sns.jointplot(x=x, y=y, kind='hex', marginal_kws=dict(bins=30))
    g.set_axis_labels('Energy', 'Differential Entropy δH (nats)')
    
    return g


def plot_entropy_hexgrid(results, figsize=(8, 6)):
    """Plot energy vs differential entropy using seaborn hexgrid."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Get valid data points
    valid_mask = ~np.isnan(results['energies'])
    x = results['energies'][valid_mask]
    y = results['differential_entropies'][valid_mask]
    
    # Create hexbin plot with marginal histograms - this creates its own figure
    g = sns.jointplot(x=x, y=y, kind='hex', marginal_kws=dict(bins=30), height=figsize[1])
    g.set_axis_labels('Energy', 'Differential Entropy δH (nats)')
    
    return g.fig  # Return the figure from the JointGrid


def density_based_sampling(results, sample_size=100, n_bins=20):
    """
    Sample structures proportionally from 2D energy-entropy density.
    
    Parameters:
    -----------
    results : dict
        Results from compute_entropy_analysis()
    sample_size : int
        Number of structures to sample
    n_bins : int
        Number of bins in each dimension for density estimation
        
    Returns:
    --------
    numpy.ndarray : Sampled structure IDs
    """
    valid_mask = ~np.isnan(results['energies'])
    energies = results['energies'][valid_mask]
    entropies = results['differential_entropies'][valid_mask]
    struct_ids = np.array(results['struct_ids'])[valid_mask]
    
    # Create 2D histogram for density estimation
    H, energy_edges, entropy_edges = np.histogram2d(
        energies, entropies, bins=n_bins
    )
    
    # Find which bin each structure belongs to
    energy_bins = np.digitize(energies, energy_edges) - 1
    entropy_bins = np.digitize(entropies, entropy_edges) - 1
    
    # Clip to valid bin indices
    energy_bins = np.clip(energy_bins, 0, n_bins-1)
    entropy_bins = np.clip(entropy_bins, 0, n_bins-1)
    
    # Calculate sampling probability for each structure
    # Probability proportional to bin density
    densities = H[energy_bins, entropy_bins]
    densities = densities / np.sum(densities)  # Normalize
    
    # Sample structures based on density
    if sample_size >= len(struct_ids):
        return struct_ids
    
    sampled_indices = np.random.choice(
        len(struct_ids), size=sample_size, 
        replace=False, p=densities
    )
    
    sampled_ids = struct_ids[sampled_indices]
    
    print(f"Density-based sampling: selected {len(sampled_ids)} structures")
    print(f"Energy range: {energies[sampled_indices].min():.3f} to {energies[sampled_indices].max():.3f}")
    print(f"Entropy range: {entropies[sampled_indices].min():.3f} to {entropies[sampled_indices].max():.3f}")
    
    return np.sort(sampled_ids)


def identify_energy_streaks(results, method='clustering', n_clusters=3, energy_gaps=None):
    """
    Identify energy streaks/clusters in the energy-entropy space.
    
    Parameters:
    -----------
    results : dict
        Results from compute_entropy_analysis()
    method : str
        'clustering' (DBSCAN) or 'energy_ranges' (manual ranges)
    n_clusters : int
        Expected number of clusters for clustering method
    energy_gaps : list of float
        Energy boundaries for manual streaks (e.g., [-1042.7, -1042.3])
        
    Returns:
    --------
    dict : Dictionary with streak assignments and metadata
    """
    valid_mask = ~np.isnan(results['energies'])
    energies = results['energies'][valid_mask]
    entropies = results['differential_entropies'][valid_mask]
    struct_ids = np.array(results['struct_ids'])[valid_mask]
    
    if method == 'clustering':
        # Use DBSCAN clustering on energy-entropy space
        data = np.column_stack([energies, entropies])
        
        # Standardize for clustering
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.3, min_samples=10).fit(data_scaled)
        streak_labels = clustering.labels_
        
        print(f"DBSCAN found {len(np.unique(streak_labels[streak_labels >= 0]))} clusters")
        print(f"Noise points: {np.sum(streak_labels == -1)}")
        
    elif method == 'energy_ranges':
        # Manual energy range assignment
        if energy_gaps is None:
            # Auto-detect gaps by looking at energy distribution
            energy_hist, edges = np.histogram(energies, bins=50)
            
            # Find local minima as potential gaps
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(-energy_hist, height=-np.max(energy_hist)*0.1)
            
            if len(peaks) >= 2:
                energy_gaps = [edges[peaks[0]], edges[peaks[1]]]
                print(f"Auto-detected energy gaps at: {energy_gaps}")
            else:
                # Default gaps based on your example
                energy_gaps = [-1042.7, -1042.3]
                print(f"Using default energy gaps: {energy_gaps}")
        
        # Assign streaks based on energy ranges
        streak_labels = np.zeros(len(energies), dtype=int)
        
        for i, gap in enumerate(energy_gaps):
            streak_labels[energies >= gap] = i + 1
        
        print(f"Energy ranges method: {len(np.unique(streak_labels))} streaks")
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Organize results
    streak_data = {
        'streak_labels': streak_labels,
        'struct_ids': struct_ids,
        'energies': energies,
        'entropies': entropies,
        'method': method
    }
    
    # Summary by streak
    unique_labels = np.unique(streak_labels[streak_labels >= 0])
    for label in unique_labels:
        mask = streak_labels == label
        count = np.sum(mask)
        energy_range = (energies[mask].min(), energies[mask].max())
        entropy_range = (entropies[mask].min(), entropies[mask].max())
        
        print(f"Streak {label}: {count} structures")
        print(f"  Energy range: {energy_range[0]:.3f} to {energy_range[1]:.3f}")
        print(f"  Entropy range: {entropy_range[0]:.3f} to {entropy_range[1]:.3f}")
    
    return streak_data


def sample_from_streak(streak_data, streak_id, sample_size=50, entropy_method='high_entropy'):
    """
    Sample structures from a specific energy streak.
    
    Parameters:
    -----------
    streak_data : dict
        Results from identify_energy_streaks()
    streak_id : int
        Which streak to sample from
    sample_size : int
        Number of structures to sample
    entropy_method : str
        'high_entropy', 'low_entropy', 'diverse_entropy', or 'random'
        
    Returns:
    --------
    numpy.ndarray : Sampled structure IDs from the streak
    """
    # Get structures in this streak
    mask = streak_data['streak_labels'] == streak_id
    
    if not np.any(mask):
        print(f"No structures found in streak {streak_id}")
        return np.array([])
    
    streak_struct_ids = streak_data['struct_ids'][mask]
    streak_entropies = streak_data['entropies'][mask]
    
    if sample_size >= len(streak_struct_ids):
        print(f"Streak {streak_id}: returning all {len(streak_struct_ids)} structures")
        return np.sort(streak_struct_ids)
    
    # Sample based on entropy method
    if entropy_method == 'random':
        indices = np.random.choice(len(streak_struct_ids), size=sample_size, replace=False)
        
    elif entropy_method == 'high_entropy':
        # Sample highest entropy structures
        indices = np.argsort(streak_entropies)[-sample_size:]
        
    elif entropy_method == 'low_entropy':
        # Sample lowest entropy structures  
        indices = np.argsort(streak_entropies)[:sample_size]
        
    elif entropy_method == 'diverse_entropy':
        # Sample across entropy range
        indices = np.linspace(0, len(streak_entropies)-1, sample_size, dtype=int)
        indices = np.argsort(streak_entropies)[indices]
        
    else:
        raise ValueError(f"Unknown entropy method: {entropy_method}")
    
    sampled_ids = streak_struct_ids[indices]
    
    print(f"Streak {streak_id} sampling ({entropy_method}): {len(sampled_ids)} structures")
    print(f"  Entropy range: {streak_entropies[indices].min():.3f} to {streak_entropies[indices].max():.3f}")
    
    return np.sort(sampled_ids)


def plot_streak_analysis(streak_data, figsize=(12, 8)):
    """
    Plot energy-entropy space with streak assignments.
    
    Parameters:
    -----------
    streak_data : dict
        Results from identify_energy_streaks()
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    energies = streak_data['energies']
    entropies = streak_data['entropies']
    labels = streak_data['streak_labels']
    
    # Plot 1: Colored by streak
    unique_labels = np.unique(labels[labels >= 0])
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        axes[0,0].scatter(energies[mask], entropies[mask], 
                         c=[colors[i]], label=f'Streak {label}', alpha=0.7)
    
    # Plot noise points if any
    if np.any(labels == -1):
        noise_mask = labels == -1
        axes[0,0].scatter(energies[noise_mask], entropies[noise_mask], 
                         c='gray', label='Noise', alpha=0.5, s=10)
    
    axes[0,0].set_xlabel('Energy')
    axes[0,0].set_ylabel('Differential Entropy δH (nats)')
    axes[0,0].set_title('Energy Streaks')
    axes[0,0].legend()
    
    # Plot 2: Energy distribution with streak boundaries
    axes[0,1].hist(energies, bins=50, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Energy')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Energy Distribution')
    
    # Add vertical lines for streak boundaries if using energy ranges
    if streak_data['method'] == 'energy_ranges':
        energy_vals = np.sort(energies)
        for label in unique_labels[:-1]:  # Don't add line after last streak
            boundary_mask = labels == label
            if np.any(boundary_mask):
                max_energy = energies[boundary_mask].max()
                axes[0,1].axvline(max_energy, color='red', linestyle='--', alpha=0.7)
    
    # Plot 3: Streak size distribution
    streak_sizes = [np.sum(labels == label) for label in unique_labels]
    axes[1,0].bar(unique_labels, streak_sizes, color=colors)
    axes[1,0].set_xlabel('Streak ID')
    axes[1,0].set_ylabel('Number of Structures')
    axes[1,0].set_title('Streak Sizes')
    
    # Plot 4: Summary statistics
    axes[1,1].axis('off')
    stats_text = f"""
    Streak Analysis Summary
    ======================
    Method: {streak_data['method']}
    Total Structures: {len(energies)}
    Number of Streaks: {len(unique_labels)}
    
    Streak Statistics:
    """
    
    for label in unique_labels:
        mask = labels == label
        count = np.sum(mask)
        energy_range = (energies[mask].min(), energies[mask].max())
        entropy_range = (entropies[mask].min(), entropies[mask].max())
        
        stats_text += f"""
    Streak {label}: {count} structures
      Energy: {energy_range[0]:.3f} to {energy_range[1]:.3f}
      Entropy: {entropy_range[0]:.3f} to {entropy_range[1]:.3f}"""
    
    axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes,
                  fontfamily='monospace', fontsize=9, verticalalignment='top')
    
    plt.tight_layout()
    return fig

