"""
Plotting functions for structural analysis metrics.
Provides consistent interfaces for plotting Qn distributions, bond angle distributions,
and ring statistics. All functions support optional Boltzmann weighting.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.data_management_v2 import (
    load_qn_distributions,
    load_bond_angle_distributions,
    load_ring_statistics,
    load_energies,
    load_structures,
    calculate_weights
)


# =====================================================================
# Qn DISTRIBUTION PLOTTING
# =====================================================================

def plot_qn_distributions(*struct_id_lists, labels=None, use_weights=False,
                          temperature=1800, figsize=(8, 5)):
    """
    Plot Qn distributions from multiple sets of structure IDs.

    Parameters:
    -----------
    *struct_id_lists : variable number of lists
        Each list contains structure IDs to analyze
    labels : list of str, optional
        Labels for each dataset in the legend
    use_weights : bool
        If True, apply Boltzmann weighting (default: False)
    temperature : float
        Temperature in Kelvin for Boltzmann weighting
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    list : List of Qn fraction dictionaries for each dataset
    """
    if len(struct_id_lists) == 0:
        raise ValueError("At least one list of structure IDs must be provided")

    all_qn_fractions = []

    for i, struct_ids in enumerate(struct_id_lists):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"

        qn_data = load_qn_distributions(struct_ids)

        if len(qn_data) == 0:
            raise FileNotFoundError(f"No Qn distribution data found for dataset {i+1} ({label}). Run populate_qn_distributions() first.")
        if len(qn_data) < len(struct_ids):
            print(f"Warning: Only found {len(qn_data)}/{len(struct_ids)} Qn files for {label}")

        # Compute weights
        if use_weights:
            energies = load_energies(struct_ids)
            structures = load_structures(struct_ids)
            weights = calculate_weights(struct_ids, energies, structures, temperature)
        else:
            weights = {sid: 1.0 / len(struct_ids) for sid in struct_ids}

        qn_fractions = {n: 0.0 for n in range(5)}
        for struct_id, data in qn_data.items():
            if struct_id not in weights:
                continue
            w = weights[struct_id]
            for n in range(5):
                qn_fractions[n] += w * data['qn_fractions'].get(n, 0.0)

        # Renormalize (in case of missing structures)
        total = sum(qn_fractions.values())
        if total > 0:
            qn_fractions = {n: v / total for n, v in qn_fractions.items()}

        all_qn_fractions.append((label, qn_fractions))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(5)
    width = 0.8 / len(struct_id_lists)

    for i, (label, qn_fracs) in enumerate(all_qn_fractions):
        offset = width * (i - len(struct_id_lists) / 2 + 0.5)
        values = [qn_fracs[n] for n in range(5)]
        ax.bar(x + offset, values, width, label=label, alpha=0.8)

    weight_str = f" (Boltzmann T={temperature}K)" if use_weights else ""
    ax.set_xlabel('Qn Species')
    ax.set_ylabel('Fraction')
    ax.set_title(f'Qn Distribution Comparison{weight_str}')
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

def plot_bond_angle_distributions(*struct_id_lists, labels=None, bins=60,
                                   use_weights=False, temperature=1800, figsize=(10, 6)):
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
    use_weights : bool
        If True, apply Boltzmann weighting (default: False)
    temperature : float
        Temperature in Kelvin for Boltzmann weighting
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    list : List of BAD result dictionaries for each dataset
    """
    if len(struct_id_lists) == 0:
        raise ValueError("At least one list of structure IDs must be provided")

    all_bad_results = []
    angle_range = (60, 180)
    bin_edges = np.linspace(angle_range[0], angle_range[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    for i, struct_ids in enumerate(struct_id_lists):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"

        bad_data = load_bond_angle_distributions(struct_ids)

        if len(bad_data) == 0:
            raise FileNotFoundError(f"No BAD data found for dataset {i+1} ({label}). Run populate_bond_angle_distributions() first.")
        if len(bad_data) < len(struct_ids):
            print(f"Warning: Only found {len(bad_data)}/{len(struct_ids)} BAD files for {label}")

        if use_weights:
            energies = load_energies(struct_ids)
            structures = load_structures(struct_ids)
            weights = calculate_weights(struct_ids, energies, structures, temperature)
        else:
            weights = {sid: 1.0 / len(struct_ids) for sid in struct_ids}

        weighted_hist = np.zeros(bins)
        weighted_mean = 0.0
        weighted_sq = 0.0

        for struct_id, data in bad_data.items():
            if struct_id not in weights:
                continue
            w = weights[struct_id]
            angles = np.array(data['angles'])
            hist, _ = np.histogram(angles, bins=bin_edges)
            weighted_hist += w * hist
            weighted_mean += w * np.mean(angles)
            weighted_sq += w * np.mean(angles ** 2)

        # Normalize to probability density
        hist_sum = np.sum(weighted_hist)
        if hist_sum > 0:
            weighted_hist = weighted_hist / (hist_sum * bin_width)

        weighted_std = np.sqrt(max(weighted_sq - weighted_mean ** 2, 0))

        all_bad_results.append({
            'label': label,
            'bin_centers': bin_centers,
            'histogram': weighted_hist,
            'mean': weighted_mean,
            'std': weighted_std,
        })

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    for result in all_bad_results:
        ax.plot(result['bin_centers'], result['histogram'],
                label=f"{result['label']} (μ={result['mean']:.1f}°, σ={result['std']:.1f}°)",
                linewidth=2)

    weight_str = f" (Boltzmann T={temperature}K)" if use_weights else ""
    ax.set_xlabel('Si-O-Si Angle (degrees)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Bond Angle Distribution Comparison{weight_str}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return all_bad_results


# =====================================================================
# RING STATISTICS PLOTTING
# =====================================================================

def plot_ring_distributions(*struct_id_lists, labels=None, max_ring_size=20,
                             use_weights=False, temperature=1800, figsize=(10, 6)):
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
    use_weights : bool
        If True, apply Boltzmann weighting (default: False)
    temperature : float
        Temperature in Kelvin for Boltzmann weighting
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    list : List of average RN dictionaries for each dataset
    """
    if len(struct_id_lists) == 0:
        raise ValueError("At least one list of structure IDs must be provided")

    all_ring_results = []
    ring_sizes = range(2, max_ring_size + 1)

    for i, struct_ids in enumerate(struct_id_lists):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"

        ring_stats = load_ring_statistics(struct_ids)

        if len(ring_stats) == 0:
            raise FileNotFoundError(f"No ring statistics found for dataset {i+1} ({label}). Run populate_ring_statistics() first.")
        if len(ring_stats) < len(struct_ids):
            print(f"Warning: Only found {len(ring_stats)}/{len(struct_ids)} ring files for {label}")

        if use_weights:
            energies = load_energies(struct_ids)
            structures = load_structures(struct_ids)
            weights = calculate_weights(struct_ids, energies, structures, temperature)
        else:
            weights = {sid: 1.0 / len(struct_ids) for sid in struct_ids}

        avg_rn = {s: 0.0 for s in ring_sizes}

        for struct_id, stats in ring_stats.items():
            if struct_id not in weights:
                continue
            w = weights[struct_id]
            for s in ring_sizes:
                avg_rn[s] += w * stats['RN'].get(s, 0.0)

        all_ring_results.append({
            'label': label,
            'avg_rn': avg_rn,
            'n_structures': len(ring_stats)
        })

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x = list(ring_sizes)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, result in enumerate(all_ring_results):
        marker = markers[i % len(markers)]
        y = [result['avg_rn'][s] for s in x]
        ax.plot(x, y, marker=marker,
                label=f"{result['label']} (n={result['n_structures']})",
                linewidth=2, markersize=6)

    weight_str = f" (Boltzmann T={temperature}K)" if use_weights else ""
    ax.set_xlabel('Ring Size')
    ax.set_ylabel('Average RN (rings/node)')
    ax.set_title(f'Ring Size Distribution Comparison{weight_str}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return all_ring_results


# =====================================================================
# COMBINED PLOTTING
# =====================================================================

def plot_all_metrics(*struct_id_lists, labels=None, use_weights=False,
                     temperature=1800, figsize=(14, 10)):
    """
    Create plots for all structural metrics: Qn, BAD, and Rings.
    Note: RDF plotting requires importing rdf_v2.plot_multiple_rdfs separately.

    Parameters:
    -----------
    *struct_id_lists : variable number of lists
        Each list contains structure IDs to analyze
    labels : list of str, optional
        Labels for each dataset in the legend
    use_weights : bool
        If True, apply Boltzmann weighting for all metrics (default: False)
    temperature : float
        Temperature in Kelvin for Boltzmann weighting
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    if len(struct_id_lists) == 0:
        raise ValueError("At least one list of structure IDs must be provided")

    print("Plotting Qn distributions...")
    qn_results = plot_qn_distributions(*struct_id_lists, labels=labels,
                                        use_weights=use_weights, temperature=temperature)

    print("Plotting bond angle distributions...")
    bad_results = plot_bond_angle_distributions(*struct_id_lists, labels=labels,
                                                 use_weights=use_weights, temperature=temperature)

    print("Plotting ring distributions...")
    ring_results = plot_ring_distributions(*struct_id_lists, labels=labels,
                                            use_weights=use_weights, temperature=temperature)

    print("Note: To plot RDFs, use rdf_v2.plot_multiple_rdfs() separately")

    return {
        'qn': qn_results,
        'bad': bad_results,
        'rings': ring_results
    }