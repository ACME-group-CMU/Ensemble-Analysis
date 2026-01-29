"""
Plotting functions for structural analysis metrics.
Provides consistent interfaces for plotting Qn distributions, bond angle distributions,
and ring statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.data_management_v2 import (
    load_qn_distributions,
    load_bond_angle_distributions,
    load_ring_statistics
)


# =====================================================================
# Qn DISTRIBUTION PLOTTING
# =====================================================================

def plot_qn_distributions(*struct_id_lists, labels=None, figsize=(8, 5)):
    """
    Plot Qn distributions from multiple sets of structure IDs.

    Parameters:
    -----------
    *struct_id_lists : variable number of lists
        Each list contains structure IDs to analyze
    labels : list of str, optional
        Labels for each dataset in the legend
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    list : List of Qn fraction dictionaries for each dataset

    Example:
    --------
    >>> plot_qn_distributions(struct_ids_24, struct_ids_36, labels=['24-atom', '36-atom'])
    """

    if len(struct_id_lists) == 0:
        raise ValueError("At least one list of structure IDs must be provided")

    all_qn_fractions = []

    # Process each dataset
    for i, struct_ids in enumerate(struct_id_lists):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"

        # Load Qn data
        qn_data = load_qn_distributions(struct_ids)

        if len(qn_data) == 0:
            raise FileNotFoundError(f"No Qn distribution data found for dataset {i+1} ({label}). Run populate_qn_distributions() first.")

        if len(qn_data) < len(struct_ids):
            print(f"Warning: Only found {len(qn_data)}/{len(struct_ids)} Qn files for {label}")

        # Calculate average Qn fractions
        qn_counts = {n: [] for n in range(5)}

        for struct_id, data in qn_data.items():
            qn_fractions = data['qn_fractions']
            for n in range(5):
                qn_counts[n].append(qn_fractions.get(n, 0.0))

        # Average across structures
        qn_fractions = {n: np.mean(qn_counts[n]) for n in range(5)}
        all_qn_fractions.append((label, qn_fractions))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(5)
    width = 0.8 / len(struct_id_lists)

    for i, (label, qn_fracs) in enumerate(all_qn_fractions):
        offset = width * (i - len(struct_id_lists)/2 + 0.5)
        values = [qn_fracs[n] for n in range(5)]
        ax.bar(x + offset, values, width, label=label, alpha=0.8)

    ax.set_xlabel('Qn Species')
    ax.set_ylabel('Fraction')
    ax.set_title('Qn Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{n}' for n in range(5)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    return [qn_fracs for _, qn_fracs in all_qn_fractions]


# =====================================================================
# BOND ANGLE DISTRIBUTION PLOTTING
# =====================================================================

def plot_bond_angle_distributions(*struct_id_lists, labels=None, bins=60, figsize=(10, 6)):
    """
    Plot bond angle distributions from multiple sets of structure IDs.

    Parameters:
    -----------
    *struct_id_lists : variable number of lists
        Each list contains structure IDs to analyze
    labels : list of str, optional
        Labels for each dataset in the legend
    bins : int, optional
        Number of bins for histogram (default: 60)
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    list : List of BAD dictionaries for each dataset

    Example:
    --------
    >>> plot_bond_angle_distributions(struct_ids_24, struct_ids_36, labels=['24-atom', '36-atom'])
    """

    if len(struct_id_lists) == 0:
        raise ValueError("At least one list of structure IDs must be provided")

    all_bad_results = []

    # Process each dataset
    for i, struct_ids in enumerate(struct_id_lists):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"

        # Load BAD data
        bad_data = load_bond_angle_distributions(struct_ids)

        if len(bad_data) == 0:
            raise FileNotFoundError(f"No BAD data found for dataset {i+1} ({label}). Run populate_bond_angle_distributions() first.")

        if len(bad_data) < len(struct_ids):
            print(f"Warning: Only found {len(bad_data)}/{len(struct_ids)} BAD files for {label}")

        # Collect all angles
        all_angles = []
        for struct_id, data in bad_data.items():
            all_angles.extend(data['angles'])

        # Create histogram
        hist, bin_edges = np.histogram(all_angles, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate statistics
        mean_angle = np.mean(all_angles)
        std_angle = np.std(all_angles)

        all_bad_results.append({
            'label': label,
            'bin_centers': bin_centers,
            'histogram': hist,
            'mean': mean_angle,
            'std': std_angle,
            'all_angles': all_angles
        })

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    for result in all_bad_results:
        ax.plot(result['bin_centers'], result['histogram'],
                label=f"{result['label']} (μ={result['mean']:.1f}°, σ={result['std']:.1f}°)",
                linewidth=2)

    ax.set_xlabel('Si-O-Si Angle (degrees)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Bond Angle Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return all_bad_results


# =====================================================================
# RING STATISTICS PLOTTING
# =====================================================================

def plot_ring_distributions(*struct_id_lists, labels=None, max_ring_size=20, figsize=(10, 6)):
    """
    Plot ring size distributions from multiple sets of structure IDs.
    Uses RN (rings per node) metric.

    Parameters:
    -----------
    *struct_id_lists : variable number of lists
        Each list contains structure IDs to analyze
    labels : list of str, optional
        Labels for each dataset in the legend
    max_ring_size : int, optional
        Maximum ring size to plot (default: 20)
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    list : List of average RN dictionaries for each dataset

    Example:
    --------
    >>> plot_ring_distributions(struct_ids_24, struct_ids_36, labels=['24-atom', '36-atom'])
    """

    if len(struct_id_lists) == 0:
        raise ValueError("At least one list of structure IDs must be provided")

    all_ring_results = []

    # Process each dataset
    for i, struct_ids in enumerate(struct_id_lists):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"

        # Load ring statistics
        ring_stats = load_ring_statistics(struct_ids)

        if len(ring_stats) == 0:
            raise FileNotFoundError(f"No ring statistics found for dataset {i+1} ({label}). Run populate_ring_statistics() first.")

        if len(ring_stats) < len(struct_ids):
            print(f"Warning: Only found {len(ring_stats)}/{len(struct_ids)} ring files for {label}")

        # Calculate average RN for each ring size
        ring_sizes = range(2, max_ring_size + 1)
        avg_rn = {}

        for size in ring_sizes:
            rn_values = [stats['RN'].get(size, 0) for stats in ring_stats.values()]
            avg_rn[size] = np.mean(rn_values)

        all_ring_results.append({
            'label': label,
            'avg_rn': avg_rn,
            'n_structures': len(ring_stats)
        })

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    x = list(ring_sizes)

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, result in enumerate(all_ring_results):
        marker = markers[i % len(markers)]
        y = [result['avg_rn'][s] for s in x]
        ax.plot(x, y, marker=marker, label=f"{result['label']} (n={result['n_structures']})",
                linewidth=2, markersize=6)

    ax.set_xlabel('Ring Size')
    ax.set_ylabel('Average RN (rings/node)')
    ax.set_title('Ring Size Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return all_ring_results


# =====================================================================
# COMBINED PLOTTING
# =====================================================================

def plot_all_metrics(*struct_id_lists, labels=None, figsize=(14, 10)):
    """
    Create a 2x2 grid showing all structural metrics: RDF, Qn, BAD, and Rings.
    Note: RDF plotting requires importing rdf_v2.plot_multiple_rdfs separately.

    Parameters:
    -----------
    *struct_id_lists : variable number of lists
        Each list contains structure IDs to analyze
    labels : list of str, optional
        Labels for each dataset in the legend
    figsize : tuple, optional
        Figure size (width, height) in inches

    Example:
    --------
    >>> plot_all_metrics(struct_ids_24, struct_ids_36, labels=['24-atom', '36-atom'])
    """

    if len(struct_id_lists) == 0:
        raise ValueError("At least one list of structure IDs must be provided")

    # Create individual plots
    print("Plotting Qn distributions...")
    qn_results = plot_qn_distributions(*struct_id_lists, labels=labels)

    print("Plotting bond angle distributions...")
    bad_results = plot_bond_angle_distributions(*struct_id_lists, labels=labels)

    print("Plotting ring distributions...")
    ring_results = plot_ring_distributions(*struct_id_lists, labels=labels)

    print("Note: To plot RDFs, use rdf_v2.plot_multiple_rdfs() separately")

    return {
        'qn': qn_results,
        'bad': bad_results,
        'rings': ring_results
    }
