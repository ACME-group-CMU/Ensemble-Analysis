"""
Birch-Murnaghan Equation of State Analysis
Robust implementation for limited volume range data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Physical constants
RY_TO_EV = 13.6056980659     # Rydberg to eV
EV_PER_A3_TO_GPA = 160.2176  # eV/Å³ to GPa


def parse_bulk_modulus_file(filename):
    """Parse bulk_modulus_summary.txt file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                data.append({
                    'volume': float(parts[2]),
                    'energy': float(parts[3]),
                    'status': parts[4]
                })
            except ValueError:
                continue
    
    df = pd.DataFrame(data)
    return df[df['status'] == 'SUCCESS']


def parabolic_fit_k0(volumes, energies_ev):
    """
    Get bulk modulus from parabolic fit
    This is the most stable method for limited data
    """
    # Fit E = a*V^2 + b*V + c
    poly_coeffs = np.polyfit(volumes, energies_ev, 2)
    a, b, c = poly_coeffs
    
    # Find minimum
    V0 = -b / (2 * a)
    
    # K = V * d²E/dV² at V0
    # For parabola: d²E/dV² = 2a
    K0_ev_per_a3 = V0 * 2 * a
    K0_gpa = K0_ev_per_a3 * EV_PER_A3_TO_GPA
    
    # Calculate R² for parabolic fit
    E_fit = np.polyval(poly_coeffs, volumes)
    ss_res = np.sum((energies_ev - E_fit)**2)
    ss_tot = np.sum((energies_ev - energies_ev.mean())**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return K0_gpa, V0, r_squared, poly_coeffs


def birch_murnaghan_energy_3rd(V, E0, V0, K0, K0_prime):
    """3rd order Birch-Murnaghan E(V) equation"""
    eta = (V0/V)**(2/3)
    E = E0 + (9*V0*K0/16) * ((eta - 1)**3 * K0_prime + (eta - 1)**2 * (6 - 4*eta))
    return E


def fit_eos_robust(volumes, energies_ry):
    """
    Robust EOS fitting with multiple methods
    
    Returns the best result from:
    1. Parabolic fit (most stable)
    2. 3rd order BM with K0'=4
    3. Full 3rd order BM
    """
    # Sort by volume
    sort_idx = np.argsort(volumes)
    V = volumes[sort_idx]
    E_ry = energies_ry[sort_idx]
    E_ev = E_ry * RY_TO_EV
    
    # Method 1: Parabolic fit (always works)
    K0_para, V0_para, r2_para, poly_coeffs = parabolic_fit_k0(V, E_ev)
    
    results = {
        'method': 'parabolic',
        'V0': V0_para,
        'K0': K0_para,
        'K0_prime': 4.0,  # Not applicable but included for consistency
        'r_squared': r2_para,
        'volumes': V,
        'energies_ev': E_ev,
        'poly_coeffs': poly_coeffs
    }
    
    # Only try BM fitting if parabolic K0 is reasonable
    if 10 < K0_para < 300 and r2_para > 0.9:
        
        # Method 2: 3rd order BM with K0'=4 (more stable)
        try:
            E0_guess = E_ev.min()
            popt, _ = curve_fit(
                lambda V, E0, V0, K0: birch_murnaghan_energy_3rd(V, E0, V0, K0, 4.0),
                V, E_ev,
                p0=[E0_guess, V0_para, K0_para],
                bounds=([E0_guess-1, V.min()*0.9, 10], 
                       [E0_guess+1, V.max()*1.1, 300]),
                maxfev=5000
            )
            
            E0_bm, V0_bm, K0_bm = popt
            E_fit_bm = birch_murnaghan_energy_3rd(V, E0_bm, V0_bm, K0_bm, 4.0)
            
            ss_res = np.sum((E_ev - E_fit_bm)**2)
            ss_tot = np.sum((E_ev - E_ev.mean())**2)
            r2_bm = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Use BM if it's better
            if r2_bm > r2_para and r2_bm > 0.95:
                results = {
                    'method': 'BM-3rd-fixed',
                    'V0': V0_bm,
                    'K0': K0_bm,
                    'K0_prime': 4.0,
                    'r_squared': r2_bm,
                    'volumes': V,
                    'energies_ev': E_ev,
                    'E0': E0_bm,
                    'energies_fit': E_fit_bm
                }
        except:
            pass  # Keep parabolic result
    
    return results


def analyze_bulk_modulus(filename, plot=True, save_dir=None):
    """
    Complete bulk modulus analysis
    
    Parameters:
    -----------
    filename : str
        Path to bulk_modulus_summary.txt
    plot : bool
        Generate plot
    save_dir : str
        Directory to save results (optional)
        
    Returns:
    --------
    dict with results
    """
    # Parse data
    df = parse_bulk_modulus_file(filename)
    
    if len(df) < 4:
        raise ValueError(f"Need at least 4 data points, got {len(df)}")
    
    volumes = df['volume'].values
    energies = df['energy'].values
    
    # Check data quality
    V_range = (volumes.max() - volumes.min()) / volumes.mean() * 100
    E_range_mev = (energies.max() - energies.min()) * RY_TO_EV * 1000
    
    print(f"\nAnalyzing: {filename}")
    print(f"Data points: {len(df)}")
    print(f"Volume range: {V_range:.1f}%")
    print(f"Energy range: {E_range_mev:.1f} meV")
    
    # Fit EOS
    results = fit_eos_robust(volumes, energies)
    
    # Print results
    print(f"\nResults ({results['method']}):")
    print(f"  V₀ = {results['V0']:.2f} Å³")
    print(f"  K₀ = {results['K0']:.1f} GPa")
    print(f"  R² = {results['r_squared']:.4f}")
    
    # Add data quality info
    results['volume_range_percent'] = V_range
    results['energy_range_mev'] = E_range_mev
    results['filename'] = filename
    
    # Generate plot
    if plot:
        plot_eos_fit(results, filename, save_dir)
    
    # Save results
    if save_dir:
        save_results(results, filename, save_dir)
    
    return results


def plot_eos_fit(results, filename, save_dir=None):
    """Create clean EOS plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    V = results['volumes']
    E_ev = results['energies_ev']
    E_rel = (E_ev - E_ev.min()) * 1000  # meV
    
    # Energy vs Volume
    ax1.scatter(V, E_rel, s=60, color='black', zorder=5, label='DFT data')
    
    # Plot fit
    V_smooth = np.linspace(V.min()*0.98, V.max()*1.02, 200)
    
    if results['method'] == 'parabolic':
        # Plot parabolic fit
        E_smooth = np.polyval(results['poly_coeffs'], V_smooth)
        E_smooth_rel = (E_smooth - E_ev.min()) * 1000
        ax1.plot(V_smooth, E_smooth_rel, 'b-', linewidth=2, label='Parabolic fit')
    else:
        # Plot BM fit
        E_smooth = birch_murnaghan_energy_3rd(
            V_smooth, results['E0'], results['V0'], 
            results['K0'], results['K0_prime']
        )
        E_smooth_rel = (E_smooth - E_ev.min()) * 1000
        ax1.plot(V_smooth, E_smooth_rel, 'r-', linewidth=2, label='Birch-Murnaghan fit')
    
    ax1.axvline(results['V0'], color='green', linestyle='--', alpha=0.5, 
                label=f'V₀ = {results["V0"]:.1f} Å³')
    ax1.set_xlabel('Volume (Å³)')
    ax1.set_ylabel('E - E₀ (meV)')
    ax1.set_title('Energy vs Volume')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    if results['method'] == 'parabolic':
        E_fit = np.polyval(results['poly_coeffs'], V)
    else:
        E_fit = results['energies_fit']
    
    E_fit_rel = (E_fit - E_ev.min()) * 1000
    residuals = E_rel - E_fit_rel
    
    ax2.scatter(V, residuals, s=60, color='blue')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Volume (Å³)')
    ax2.set_ylabel('Residuals (meV)')
    ax2.set_title('Fit Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Add results box
    textstr = f'K₀ = {results["K0"]:.1f} GPa\n'
    textstr += f'Method: {results["method"]}\n'
    textstr += f'R² = {results["r_squared"]:.4f}'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    structure = os.path.basename(filename).split('_')[0]
    plt.suptitle(f'EOS Analysis: Structure {structure}')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{structure}_eos_fit.png'), 
                    dpi=150, bbox_inches='tight')
    
    plt.show()


def save_results(results, filename, save_dir):
    """Save results to text file"""
    os.makedirs(save_dir, exist_ok=True)
    structure = os.path.basename(filename).split('_')[0]
    
    with open(os.path.join(save_dir, f'{structure}_bulk_modulus.txt'), 'w') as f:
        f.write(f"Bulk Modulus Analysis Results\n")
        f.write(f"{'='*40}\n")
        f.write(f"Structure: {structure}\n")
        f.write(f"Data file: {filename}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"  Method: {results['method']}\n")
        f.write(f"  V₀ = {results['V0']:.3f} Å³\n")
        f.write(f"  K₀ = {results['K0']:.2f} GPa\n")
        f.write(f"  R² = {results['r_squared']:.4f}\n")
        
        f.write(f"\nData quality:\n")
        f.write(f"  Volume range: {results['volume_range_percent']:.1f}%\n")
        f.write(f"  Energy range: {results['energy_range_mev']:.1f} meV\n")


def batch_analysis(file_pattern='*_bulk_modulus_summary.txt', save_dir='results'):
    """Analyze multiple structures"""
    import glob
    
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found matching: {file_pattern}")
        return None
    
    print(f"Found {len(files)} structures to analyze\n")
    
    all_results = {}
    summary = []
    
    for file in files:
        structure = os.path.basename(file).split('_')[0]
        try:
            results = analyze_bulk_modulus(file, plot=True, save_dir=save_dir)
            all_results[structure] = results
            
            summary.append({
                'Structure': structure,
                'V0 (Å³)': results['V0'],
                'K0 (GPa)': results['K0'],
                'R²': results['r_squared'],
                'Method': results['method'],
                'V_range (%)': results['volume_range_percent']
            })
            
        except Exception as e:
            print(f"Error analyzing {file}: {str(e)}\n")
    
    # Print summary table
    if summary:
        summary_df = pd.DataFrame(summary)
        print("\n" + "="*60)
        print("SUMMARY OF ALL STRUCTURES")
        print("="*60)
        print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.2f}' if x > 0.1 else f'{x:.4f}'))
        
        # Save summary
        if save_dir:
            summary_df.to_csv(os.path.join(save_dir, 'bulk_modulus_summary.csv'), 
                             index=False)
    
    return all_results


# Quick access function
def quick_bulk_modulus(filename):
    """Get bulk modulus with minimal output"""
    df = parse_bulk_modulus_file(filename)
    results = fit_eos_robust(df['volume'].values, df['energy'].values)
    return results['K0'], results['V0']


# Diagnostic function for troubleshooting
def diagnose_fit(filename):
    """Detailed diagnostics for debugging"""
    df = parse_bulk_modulus_file(filename)
    V = df['volume'].values
    E_ry = df['energy'].values
    E_ev = E_ry * RY_TO_EV
    
    # Sort
    sort_idx = np.argsort(V)
    V = V[sort_idx]
    E_ev = E_ev[sort_idx]
    
    print(f"\nDiagnostic for: {filename}")
    print(f"Volumes: {V}")
    print(f"Energies (eV): {E_ev}")
    print(f"E range (meV): {(E_ev.max() - E_ev.min())*1000:.2f}")
    
    # Try parabolic fit
    K0, V0, r2, coeffs = parabolic_fit_k0(V, E_ev)
    print(f"\nParabolic fit:")
    print(f"  Coefficients: a={coeffs[0]:.6e}, b={coeffs[1]:.6e}, c={coeffs[2]:.6e}")
    print(f"  K0 = {K0:.1f} GPa")
    print(f"  V0 = {V0:.1f} Å³")
    print(f"  R² = {r2:.4f}")
    
    # Check curvature
    if coeffs[0] < 0:
        print("  WARNING: Negative curvature! (a < 0)")
    
    return {'V': V, 'E': E_ev, 'K0': K0, 'V0': V0, 'coeffs': coeffs}

def calculate_voigt_reuss_bounds(bulk_moduli, volumes=None, method='equal_volume'):
    """
    Calculate Voigt-Reuss bounds for bulk modulus
    
    Parameters:
    -----------
    bulk_moduli : array
        Bulk moduli of individual structures/microstates (GPa)
    volumes : array, optional
        Volumes of individual structures. If None, assumes equal volumes
    method : str
        'equal_volume' - assumes all structures have equal volume weight
        'weighted' - uses provided volumes for weighting
        
    Returns:
    --------
    dict with BR, BV, and statistics
    """
    bulk_moduli = np.array(bulk_moduli)
    n = len(bulk_moduli)
    
    if volumes is None or method == 'equal_volume':
        # Equal volume assumption - all Vi = V/n
        volumes = np.ones(n)
    else:
        volumes = np.array(volumes)
    
    # Normalize volumes
    V_total = volumes.sum()
    volume_fractions = volumes / V_total
    
    # Voigt bound (upper) - uniform strain
    # BV = Σ(Vi*Bi)/V = Σ(fi*Bi) where fi = Vi/V
    B_voigt = np.sum(volume_fractions * bulk_moduli)
    
    # Reuss bound (lower) - uniform stress  
    # BR = (Σ(Vi/Bi)/V)^-1 = (Σ(fi/Bi))^-1
    B_reuss = 1.0 / np.sum(volume_fractions / bulk_moduli)
    
    # Hill average (geometric mean of bounds)
    B_hill = (B_voigt + B_reuss) / 2
    
    return {
        'B_reuss': B_reuss,
        'B_voigt': B_voigt, 
        'B_hill': B_hill,
        'range': B_voigt - B_reuss,
        'range_percent': (B_voigt - B_reuss) / B_hill * 100,
        'n_structures': n,
        'bulk_moduli_stats': {
            'mean': bulk_moduli.mean(),
            'std': bulk_moduli.std(),
            'min': bulk_moduli.min(),
            'max': bulk_moduli.max()
        }
    }

def voigt_reuss_uncertainty(bulk_moduli, volumes=None, method='bootstrap', n_bootstrap=1000):
    """
    Estimate uncertainty in Voigt-Reuss bounds using bootstrap or analytical methods
    
    Parameters:
    -----------
    bulk_moduli : array
        Bulk moduli of individual structures
    volumes : array, optional
        Volumes of structures
    method : str
        'bootstrap' - bootstrap resampling
        'analytical' - central limit theorem (assumes equal volumes)
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns:
    --------
    dict with bounds and uncertainties
    """
    bulk_moduli = np.array(bulk_moduli)
    n = len(bulk_moduli)
    
    if method == 'analytical' and (volumes is None):
        # Analytical uncertainty using central limit theorem (equal volumes)
        V_mean = np.mean(bulk_moduli)
        VB_mean = np.mean(bulk_moduli**2)  # For equal volumes, Vi*Bi = Bi
        V_over_B_mean = np.mean(1/bulk_moduli)
        
        V_std = np.std(bulk_moduli)
        VB_std = np.std(bulk_moduli**2) 
        V_over_B_std = np.std(1/bulk_moduli)
        
        # Standard errors
        V_se = V_std / np.sqrt(n)
        VB_se = VB_std / np.sqrt(n)
        V_over_B_se = V_over_B_std / np.sqrt(n)
        
        # Bounds and uncertainties
        B_voigt = VB_mean / V_mean
        B_reuss = V_mean / V_over_B_mean
        
        # Error propagation (approximate)
        B_voigt_err = B_voigt * np.sqrt((VB_se/VB_mean)**2 + (V_se/V_mean)**2)
        B_reuss_err = B_reuss * np.sqrt((V_se/V_mean)**2 + (V_over_B_se/V_over_B_mean)**2)
        
        return {
            'B_reuss': B_reuss,
            'B_reuss_err': B_reuss_err,
            'B_voigt': B_voigt,
            'B_voigt_err': B_voigt_err,
            'method': 'analytical'
        }
    
    elif method == 'bootstrap':
        # Bootstrap resampling
        bounds_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            bm_sample = bulk_moduli[indices]
            vol_sample = volumes[indices] if volumes is not None else None
            
            result = calculate_voigt_reuss_bounds(bm_sample, vol_sample)
            bounds_bootstrap.append([result['B_reuss'], result['B_voigt']])
        
        bounds_bootstrap = np.array(bounds_bootstrap)
        
        return {
            'B_reuss': np.mean(bounds_bootstrap[:, 0]),
            'B_reuss_err': np.std(bounds_bootstrap[:, 0]),
            'B_voigt': np.mean(bounds_bootstrap[:, 1]), 
            'B_voigt_err': np.std(bounds_bootstrap[:, 1]),
            'method': 'bootstrap',
            'bootstrap_samples': bounds_bootstrap
        }

def analyze_ensemble_bulk_modulus(bulk_moduli, volumes=None, plot=True):
    """
    Complete analysis of ensemble bulk modulus with Voigt-Reuss bounds
    
    Parameters:
    -----------
    bulk_moduli : array
        Bulk moduli from individual structures
    volumes : array, optional
        Volumes of structures
    plot : bool
        Generate plots
        
    Returns:
    --------
    dict with complete analysis
    """
    # Basic bounds
    bounds = calculate_voigt_reuss_bounds(bulk_moduli, volumes)
    
    # Uncertainty analysis
    uncertainty = voigt_reuss_uncertainty(bulk_moduli, volumes, method='bootstrap')
    
    # Combine results
    results = {**bounds, **uncertainty}
    
    # Print summary
    print(f"\nVoigt-Reuss Bounds Analysis")
    print(f"Number of structures: {results['n_structures']}")
    print(f"Reuss bound (lower): {results['B_reuss']:.1f} ± {results['B_reuss_err']:.1f} GPa")
    print(f"Voigt bound (upper): {results['B_voigt']:.1f} ± {results['B_voigt_err']:.1f} GPa") 
    print(f"Hill average: {results['B_hill']:.1f} GPa")
    print(f"Range: {results['range']:.1f} GPa ({results['range_percent']:.1f}%)")
    print(f"Individual K0 range: {results['bulk_moduli_stats']['min']:.1f} - {results['bulk_moduli_stats']['max']:.1f} GPa")
    
    if plot:
        plot_voigt_reuss_analysis(bulk_moduli, results)
    
    return results

def plot_voigt_reuss_analysis(bulk_moduli, results):
    """
    Plot Voigt-Reuss analysis results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram with bounds
    ax1.hist(bulk_moduli, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax1.axvline(results['B_reuss'], color='red', linestyle='--', linewidth=2, 
                label=f'Reuss: {results["B_reuss"]:.1f}±{results["B_reuss_err"]:.1f} GPa')
    ax1.axvline(results['B_voigt'], color='blue', linestyle='--', linewidth=2,
                label=f'Voigt: {results["B_voigt"]:.1f}±{results["B_voigt_err"]:.1f} GPa')
    ax1.axvline(results['B_hill'], color='green', linestyle='-', linewidth=2,
                label=f'Hill: {results["B_hill"]:.1f} GPa')
    ax1.set_xlabel('Bulk Modulus (GPa)')
    ax1.set_ylabel('Density')
    ax1.set_title('Bulk Modulus Distribution with V-R Bounds')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Bounds comparison
    methods = ['Individual\nRange', 'Voigt-Reuss\nBounds']
    ranges = [bulk_moduli.max() - bulk_moduli.min(), results['range']]
    centers = [bulk_moduli.mean(), results['B_hill']]
    
    ax2.bar(methods, ranges, alpha=0.7, color=['gray', 'lightblue'])
    ax2.set_ylabel('Range (GPa)')
    ax2.set_title('Range Comparison')
    ax2.grid(alpha=0.3)
    
    # Add text annotations
    for i, (method, range_val, center) in enumerate(zip(methods, ranges, centers)):
        ax2.text(i, range_val + 1, f'{range_val:.1f} GPa\n(center: {center:.1f})', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


    