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

def plot_sq(*data_items, labels=None, temperature=1800, use_weights=False,
            weighting='xray', figsize=(8, 5), save_path=None,
            plot_reduced=True, show_partials=False):
    """
    Plot ensemble-averaged S(q) or q(S(q)-1) from multiple datasets.

    Parameters
    ----------
    *data_items : list of struct_ids  OR  pre-calculated ensemble sq dicts
        Pre-calculated dicts should have keys 'q', 'S_total', and optionally 'partials'.
    labels : list of str
    temperature : float
    use_weights : bool
    weighting : str  'xray' | 'neutron' | 'unweighted'  (for ID-based calculation)
    figsize : tuple
    save_path : str or None
    plot_reduced : bool
        If True (default), plot q(S(q)-1) — more informative at large q (Wolf et al. style)
        If False, plot raw S(q)
    show_partials : bool
        If True, also plot partial S_αβ(q) in subplots below total

    Returns
    -------
    list of ensemble sq dicts
    """
    from src.rdf_v2 import calculate_ensemble_sq
    from src.data_management_v2 import load_sq

    if not data_items:
        raise ValueError("At least one data item required")

    all_sq_results = []

    for i, item in enumerate(data_items):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"

        if isinstance(item, (list, np.ndarray)):
            sq_raw = load_sq(list(item), pairs='all')
            # Rebuild into format expected by calculate_ensemble_sq
            # convert flat load to full sq_data structure
            sq_data_full = {}
            for sid, d in sq_raw.items():
                if isinstance(d, dict) and 'total' in d:
                    entry = d['total'].copy()
                    entry['partials'] = {
                        tuple(k.split('_')): v['S_total']
                        for k, v in d.items() if k != 'total'
                    }
                    sq_data_full[sid] = entry
                else:
                    sq_data_full[sid] = d

            ensemble = calculate_ensemble_sq(
                list(item), sq_data=sq_data_full,
                use_weights=use_weights, temperature=temperature
            )
        elif isinstance(item, dict) and 'q' in item:
            ensemble = item
        else:
            print(f"Warning: unrecognised type at index {i}, skipping")
            continue

        ensemble['_label'] = label
        all_sq_results.append(ensemble)

    if not all_sq_results:
        return all_sq_results

    n_pairs = len(all_sq_results[0].get('partials', {}))
    n_subplots = 1 + (n_pairs if show_partials else 0)

    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
    if n_subplots == 1:
        axes = [axes]

    y_label = 'q(S(q) − 1)  [1/Å]' if plot_reduced else 'S(q)'
    weight_label = {'xray': 'X-ray', 'neutron': 'Neutron',
                    'unweighted': 'Unweighted'}.get(weighting, weighting)

    # Total S(q)
    ax = axes[0]
    for res in all_sq_results:
        q = res['q']
        S = res['S_total']
        y = q * (S - 1) if plot_reduced else S
        ax.plot(q, y, label=res['_label'], linewidth=1.5)

    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('q  [1/Å]')
    ax.set_ylabel(y_label)
    ax.set_title(f'Total Structure Factor — {weight_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Partial S(q)
    if show_partials and n_pairs > 0:
        pairs = list(all_sq_results[0]['partials'].keys())
        for j, pair in enumerate(pairs):
            ax = axes[j + 1]
            for res in all_sq_results:
                q = res['q']
                S_ab = res['partials'].get(pair, np.ones_like(q))
                y = q * (S_ab - 1) if plot_reduced else S_ab
                ax.plot(q, y, label=res['_label'], linewidth=1.5)
            ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
            ax.set_xlabel('q  [1/Å]')
            ax.set_ylabel(y_label)
            ax.set_title(f'{pair[0]}-{pair[1]} Partial S(q)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    plt.show()
    return all_sq_results


# =====================================================================
# FT g(r) / D(r) PLOTTING
# =====================================================================

def plot_gr_from_sq(*data_items, labels=None, temperature=1800, use_weights=False,
                    plot_Dr=True, show_partials=False,
                    figsize=(8, 5), save_path=None):
    """
    Plot ensemble-averaged D(r) or g(r) derived from S(q) via Fourier transform.

    Parameters
    ----------
    *data_items : list of struct_ids  OR  pre-calculated ensemble gr_from_sq dicts
    labels : list of str
    temperature : float
    use_weights : bool
    plot_Dr : bool
        If True (default), plot reduced PDF D(r)
        If False, plot g(r)
    show_partials : bool
    figsize : tuple
    save_path : str or None

    Returns
    -------
    list of ensemble gr_from_sq dicts
    """
    from src.rdf_v2 import calculate_ensemble_gr_from_sq
    from src.data_management_v2 import load_gr_from_sq

    if not data_items:
        raise ValueError("At least one data item required")

    all_results = []

    for i, item in enumerate(data_items):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"

        if isinstance(item, (list, np.ndarray)):
            gr_sq_raw = load_gr_from_sq(list(item), pairs='all')
            # Reformat for ensemble averaging
            gr_sq_data = {}
            for sid, d in gr_sq_raw.items():
                if isinstance(d, dict) and 'total' in d:
                    entry = d['total'].copy()
                    entry['D_partials'] = {
                        tuple(k.split('_')): v['D_r']
                        for k, v in d.items() if k != 'total'
                    }
                    entry['g_partials'] = {
                        tuple(k.split('_')): v['g_r']
                        for k, v in d.items() if k != 'total' and v.get('g_r') is not None
                    }
                    gr_sq_data[sid] = entry
                else:
                    gr_sq_data[sid] = d

            ensemble = calculate_ensemble_gr_from_sq(
                list(item), gr_sq_data=gr_sq_data,
                use_weights=use_weights, temperature=temperature
            )
        elif isinstance(item, dict) and 'r' in item:
            ensemble = item
        else:
            print(f"Warning: unrecognised type at index {i}, skipping")
            continue

        ensemble['_label'] = label
        all_results.append(ensemble)

    if not all_results:
        return all_results

    n_pairs = len(all_results[0].get('D_partials', {}))
    n_subplots = 1 + (n_pairs if show_partials else 0)
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
    if n_subplots == 1:
        axes = [axes]

    y_key = 'D_r' if plot_Dr else 'g_r'
    y_label = 'D(r)  [1/Å²]' if plot_Dr else 'g(r)'
    lorch_note = ' (Lorch)' if all_results[0].get('use_lorch', True) else ''
    weight_label = {'xray': 'X-ray', 'neutron': 'Neutron',
                    'unweighted': 'Unweighted'}.get(
        all_results[0].get('weighting', ''), '')

    ax = axes[0]
    for res in all_results:
        r = res['r']
        y = res.get(y_key)
        if y is None:
            print(f"Warning: {y_key} not available for {res['_label']}, skipping")
            continue
        ax.plot(r, y, label=res['_label'], linewidth=1.5)

    ax.set_xlabel('r  [Å]')
    ax.set_ylabel(y_label)
    ax.set_title(f'Total {y_label} from S(q){lorch_note} — {weight_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show_partials:
        pairs = list(all_results[0]['D_partials'].keys())
        partial_key = 'D_partials' if plot_Dr else 'g_partials'
        for j, pair in enumerate(pairs):
            ax = axes[j + 1]
            for res in all_results:
                r = res['r']
                partials = res.get(partial_key, {})
                y = partials.get(pair)
                if y is None:
                    continue
                ax.plot(r, y, label=res['_label'], linewidth=1.5)
            ax.set_xlabel('r  [Å]')
            ax.set_ylabel(y_label)
            ax.set_title(f'{pair[0]}-{pair[1]} {y_label} from S(q){lorch_note}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    plt.show()
    return all_results


# =====================================================================
# WOLF-STYLE SIDE-BY-SIDE PLOT: q(S(q)-1) | D(r)
# =====================================================================

def plot_sq_vs_gr(*data_items, labels=None, temperature=1800, use_weights=False,
                  weighting='xray', figsize=(12, 5), save_path=None):
    """
    Side-by-side plot of q(S(q)-1) and D(r), mirroring Wolf et al. Fig. 2.

    Left panel:  q(S(q) − 1) vs q
    Right panel: D(r) vs r

    Parameters
    ----------
    *data_items : list of struct_ids  OR  tuple of (sq_dict, gr_sq_dict)
    labels : list of str
    temperature : float
    use_weights : bool
    weighting : str  'xray' | 'neutron' | 'unweighted'
    figsize : tuple
    save_path : str or None
    """
    from src.rdf_v2 import calculate_ensemble_sq, calculate_ensemble_gr_from_sq
    from src.data_management_v2 import load_sq, load_gr_from_sq

    if not data_items:
        raise ValueError("At least one data item required")

    weight_label = {'xray': 'X-ray', 'neutron': 'Neutron',
                    'unweighted': 'Unweighted'}.get(weighting, weighting)

    fig, (ax_sq, ax_gr) = plt.subplots(1, 2, figsize=figsize)

    for i, item in enumerate(data_items):
        label = labels[i] if labels and i < len(labels) else f"Dataset {i+1}"

        if isinstance(item, (list, np.ndarray)):
            struct_ids = list(item)
            sq_raw = load_sq(struct_ids, pairs='total')
            gr_raw = load_gr_from_sq(struct_ids, pairs='total')

            # Quick ensemble average
            valid_sq = [sid for sid in struct_ids if sid in sq_raw]
            valid_gr = [sid for sid in struct_ids if sid in gr_raw]
            n_sq = len(valid_sq)
            n_gr = len(valid_gr)

            if n_sq == 0 or n_gr == 0:
                print(f"No S(q) or g(r) data found for {label}, skipping")
                continue

            q_arr = sq_raw[valid_sq[0]]['q']
            S_avg = np.mean([sq_raw[sid]['S_total'] for sid in valid_sq], axis=0)
            r_arr = gr_raw[valid_gr[0]]['r']
            D_avg = np.mean([gr_raw[sid]['D_r'] for sid in valid_gr], axis=0)

        elif isinstance(item, tuple) and len(item) == 2:
            sq_ens, gr_ens = item
            q_arr = sq_ens['q']
            S_avg = sq_ens['S_total']
            r_arr = gr_ens['r']
            D_avg = gr_ens['D_r']
        else:
            print(f"Warning: unrecognised type at index {i}, skipping")
            continue

        ax_sq.plot(q_arr, q_arr * (S_avg - 1), label=label, linewidth=1.5)
        ax_gr.plot(r_arr, D_avg, label=label, linewidth=1.5)

    ax_sq.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax_sq.set_xlabel('q  [1/Å]')
    ax_sq.set_ylabel('q(S(q) − 1)  [1/Å]')
    ax_sq.set_title(f'q(S(q) − 1) — {weight_label}')
    ax_sq.legend()
    ax_sq.grid(True, alpha=0.3)

    ax_gr.set_xlabel('r  [Å]')
    ax_gr.set_ylabel('D(r)  [1/Å²]')
    ax_gr.set_title(f'D(r) from FT — {weight_label}')
    ax_gr.legend()
    ax_gr.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    plt.show()


# =====================================================================
# OVERLAY: DIRECT-SPACE g(r) vs FT g(r) — Wolf et al. Fig. 2b style
# =====================================================================

def plot_direct_vs_ft(struct_ids, labels=('Direct space', 'FT from S(q)'),
                      temperature=1800, use_weights=False,
                      smoothed=True, element_pairs=None,
                      figsize=(8, 10), save_path=None):
    """
    Overlay direct-space g(r) and FT-derived g(r) for the same ensemble.
    Reproduces the orange vs red comparison in Wolf et al. Fig. 2b.

    Parameters
    ----------
    struct_ids : list of str
    labels : tuple of str  (direct_label, ft_label)
    temperature : float
    use_weights : bool
    smoothed : bool  use KAMEL-LOBE smoothed direct-space RDFs
    element_pairs : list of tuples or None
    figsize : tuple
    save_path : str or None
    """
    from src.rdf_v2 import (calculate_ensemble_partial_rdfs,
                             calculate_ensemble_gr_from_sq)
    from src.data_management_v2 import load_gr_from_sq
    from src.data_management_v2 import STANDARD_PARAMS

    if element_pairs is None:
        element_pairs = STANDARD_PARAMS['element_pairs']

    # Direct space
    direct_rdfs = calculate_ensemble_partial_rdfs(
        struct_ids=struct_ids, use_weights=use_weights,
        temperature=temperature, smoothed=smoothed,
        element_pairs=element_pairs
    )

    # FT route
    gr_sq_raw = load_gr_from_sq(struct_ids, pairs='all')
    gr_sq_data = {}
    for sid, d in gr_sq_raw.items():
        if isinstance(d, dict) and 'total' in d:
            entry = d['total'].copy()
            entry['D_partials'] = {
                tuple(k.split('_')): v['D_r']
                for k, v in d.items() if k != 'total'
            }
            entry['g_partials'] = {
                tuple(k.split('_')): v['g_r']
                for k, v in d.items() if k != 'total' and v.get('g_r') is not None
            }
            gr_sq_data[sid] = entry
        else:
            gr_sq_data[sid] = d

    ft_rdfs = calculate_ensemble_gr_from_sq(
        struct_ids, gr_sq_data=gr_sq_data,
        use_weights=use_weights, temperature=temperature
    )

    n = len(element_pairs)
    fig, axes = plt.subplots(n, 1, figsize=figsize)
    if n == 1:
        axes = [axes]

    direct_label, ft_label = labels
    lorch_note = ' (Lorch)' if ft_rdfs.get('use_lorch', True) else ''

    for i, pair in enumerate(element_pairs):
        ax = axes[i]

        if pair in direct_rdfs:
            r_d, g_d = direct_rdfs[pair]
            ax.plot(r_d, g_d, color='tab:orange', label=direct_label,
                    linewidth=1.5, alpha=0.9)

        g_partials = ft_rdfs.get('g_partials', {})
        if pair in g_partials and g_partials[pair] is not None:
            r_ft = ft_rdfs['r']
            g_ft = g_partials[pair]
            ax.plot(r_ft, g_ft, color='tab:red', label=f'{ft_label}{lorch_note}',
                    linewidth=1.5)

        ax.set_xlabel('r  [Å]')
        ax.set_ylabel('g(r)')
        ax.set_title(f'{pair[0]}-{pair[1]}  g(r):  Direct vs FT')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    plt.show()