import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Unit conversions
RY_TO_EV = 13.6056980659  # Rydberg to eV
EV_TO_J = 1.602176634e-19  # eV to Joules
ANGSTROM3_TO_M3 = 1e-30    # Angstrom^3 to m^3
PA_TO_GPA = 1e-9           # Pa to GPa
EV_PER_A3_TO_GPA = 160.2176  # eV/Å³ to GPa

def birch_murnaghan_pressure_2nd_order(V, V0, K0):
    """
    2nd order Birch-Murnaghan pressure equation (K0' fixed at 4)
    """
    v_ratio = V0/V
    v_ratio_53 = v_ratio**(5/3)
    v_ratio_73 = v_ratio**(7/3)
    
    term1 = v_ratio_73 - v_ratio_53
    # K0_prime = 4 (fixed)
    
    P = (3*K0/2) * term1
    return P
    """
    3rd order Birch-Murnaghan pressure equation
    
    P = (3K₀/2) * [(V₀/V)^(7/3) - (V₀/V)^(5/3)] * [1 + 3/4(K₀' - 4)((V₀/V)^(2/3) - 1)]
    """
    # Calculate terms directly - no intermediate eta variable
    v_ratio = V0/V
    v_ratio_23 = v_ratio**(2/3)
    v_ratio_53 = v_ratio**(5/3)
    v_ratio_73 = v_ratio**(7/3)
    
    term1 = v_ratio_73 - v_ratio_53
    term2 = 1 + (3/4) * (K0_prime - 4) * (v_ratio_23 - 1)
    
    P = (3*K0/2) * term1 * term2
    return P

def parse_bulk_modulus_summary(summary_file):
    """Parse bulk_modulus_summary.txt file"""
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    
    # Filter out comment lines and empty lines
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    
    # Parse the data
    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 5:
            try:
                scale_label = parts[0]
                volume_scale = float(parts[1])
                actual_volume = float(parts[2]) if parts[2] != 'N/A' else np.nan
                energy = float(parts[3]) if parts[3] != 'N/A' else np.nan
                status = parts[4]
                time_str = parts[5] if len(parts) > 5 else ''
                
                data.append({
                    'Scale': scale_label,
                    'Volume_Scale': volume_scale,
                    'Actual_Volume': actual_volume,
                    'Energy': energy,
                    'Status': status,
                    'Time': time_str
                })
            except ValueError:
                continue
    
    return pd.DataFrame(data)

def calculate_pressures(volumes, energies):
    """Calculate pressures from energy-volume data using P = -dE/dV"""
    pressures = np.zeros_like(volumes)
    
    for i in range(len(volumes)):
        if i == 0:
            # Forward difference
            pressures[i] = -(energies[i+1] - energies[i]) / (volumes[i+1] - volumes[i])
        elif i == len(volumes) - 1:
            # Backward difference
            pressures[i] = -(energies[i] - energies[i-1]) / (volumes[i] - volumes[i-1])
        else:
            # Central difference
            pressures[i] = -(energies[i+1] - energies[i-1]) / (volumes[i+1] - volumes[i-1])
    
    return pressures

def fit_birch_murnaghan(summary_file, output_dir=None):
    """
    Fit Birch-Murnaghan equation of state using P-V approach
    """
    # Parse data
    df = parse_bulk_modulus_summary(summary_file)
    successful_data = df[df['Status'] == 'SUCCESS'].copy()
    
    if len(successful_data) < 4:
        raise ValueError(f"Need at least 4 successful data points. Got {len(successful_data)}")
    
    # Extract and sort data
    volumes = successful_data['Actual_Volume'].values
    energies_ry = successful_data['Energy'].values
    energies_ev = energies_ry * RY_TO_EV
    
    # Sort by volume
    sort_idx = np.argsort(volumes)
    volumes = volumes[sort_idx]
    energies_ev = energies_ev[sort_idx]
    
    # Make energies relative to minimum
    energies_ev_rel = energies_ev - np.min(energies_ev)
    
    # Calculate pressures P = -dE/dV (in eV/Å³)
    pressures_ev_per_a3 = calculate_pressures(volumes, energies_ev_rel)
    
    # Convert to GPa
    pressures_gpa = pressures_ev_per_a3 * EV_PER_A3_TO_GPA
    
    print(f"Calculated pressures (GPa): {pressures_gpa}")
    
    # No outlier filtering - use all data points
    
    # Initial parameter guesses
    V0_guess = volumes[np.argmin(energies_ev_rel)]
    
    # Estimate K0 from parabolic fit
    poly_coeffs = np.polyfit(volumes, energies_ev_rel, 2)
    K0_parabolic_ev_per_a3 = V0_guess * 2 * poly_coeffs[0]
    K0_parabolic_gpa = K0_parabolic_ev_per_a3 * EV_TO_J / ANGSTROM3_TO_M3 * PA_TO_GPA
    
    K0_guess = max(20.0, min(abs(K0_parabolic_gpa), 80.0))
    K0_prime_guess = 4.0
    
    print(f"Parabolic estimate: {K0_parabolic_gpa:.1f} GPa")
    print(f"Initial guess: V0={V0_guess:.1f} Å³, K0={K0_guess:.1f} GPa, K0'={K0_prime_guess:.1f}")
    
    # Set up fitting bounds for 2nd-order (only V0, K0 - no K0')
    lower_bounds = [min(volumes) * 0.95, 10.0]
    upper_bounds = [max(volumes) * 1.05, 100.0]
    initial_guess = [V0_guess, K0_guess]  # Only 2 parameters now
    
    # Ensure initial guess is within bounds
    for i in range(2):  # Only 2 parameters now
        initial_guess[i] = max(lower_bounds[i], min(initial_guess[i], upper_bounds[i]))
    
    # Perform fit
    try:
        popt, pcov = curve_fit(
            birch_murnaghan_pressure_2nd_order,  # Use 2nd-order function
            volumes,
            pressures_gpa,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000
        )
        
        V0_fit, K0_fit = popt  # Only 2 parameters returned
        K0_prime_fit = 4.0  # Fixed value for 2nd-order
        
        # Calculate fit quality
        pressures_pred = birch_murnaghan_pressure_2nd_order(volumes, *popt)
        
        ss_res = np.sum((pressures_gpa - pressures_pred) ** 2)
        ss_tot = np.sum((pressures_gpa - np.mean(pressures_gpa)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        rmse = np.sqrt(np.mean((pressures_gpa - pressures_pred) ** 2))
        
        # Package results
        results = {
            'bulk_modulus_gpa': K0_fit,
            'fit_params': {
                'V0': V0_fit,
                'K0_gpa': K0_fit,
                'K0_prime': K0_prime_fit
            },
            'r_squared': r_squared,
            'rmse': rmse,
            'successful_points': len(successful_data),
            'raw_data': df,
            'fit_data': {
                'volumes': volumes,
                'energies_ev_rel': energies_ev_rel,
                'pressures_gpa': pressures_gpa,
                'pressures_pred': pressures_pred
            }
        }
        
        # Save results if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, 'bulk_modulus_results.txt'), 'w') as f:
                f.write("Birch-Murnaghan Bulk Modulus Results\n")
                f.write("="*40 + "\n\n")
                f.write(f"Bulk Modulus: {K0_fit:.2f} GPa\n")
                f.write(f"V0: {V0_fit:.2f} Å³\n")
                f.write(f"K0': {K0_prime_fit:.2f}\n")
                f.write(f"R²: {r_squared:.4f}\n")
                f.write(f"RMSE: {rmse:.4f} GPa\n")
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Fitting failed: {str(e)}")

def plot_bulk_modulus_fit(summary_file, output_dir=None, show_plot=True):
    """
    Plot the Birch-Murnaghan fit
    """
    # Get fit results
    results = fit_birch_murnaghan(summary_file, output_dir)
    
    # Extract data
    volumes = results['fit_data']['volumes']
    energies_ev_rel = results['fit_data']['energies_ev_rel']
    pressures_gpa = results['fit_data']['pressures_gpa']
    pressures_pred = results['fit_data']['pressures_pred']
    
    # Create smooth curve for plotting
    V_smooth = np.linspace(volumes.min() * 0.96, volumes.max() * 1.04, 200)
    P_smooth = birch_murnaghan_pressure_2nd_order(V_smooth, 
                                                 results['fit_params']['V0'],
                                                 results['fit_params']['K0_gpa'])
    
    # Create dual plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Energy vs Volume plot
    ax1.scatter(volumes, energies_ev_rel, color='red', s=60, label='DFT Data')
    ax1.axvline(results['fit_params']['V0'], color='green', linestyle='--', alpha=0.7, 
                label=f"V₀ = {results['fit_params']['V0']:.1f} Å³")
    ax1.set_xlabel('Volume (Å³)')
    ax1.set_ylabel('Energy - E₀ (eV)')
    ax1.set_title('Energy vs Volume')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set reasonable margins
    vol_margin = (volumes.max() - volumes.min()) * 0.1
    eng_margin = (energies_ev_rel.max() - energies_ev_rel.min()) * 0.15
    ax1.set_xlim(volumes.min() - vol_margin, volumes.max() + vol_margin)
    ax1.set_ylim(energies_ev_rel.min() - eng_margin, energies_ev_rel.max() + eng_margin)
    
    # Pressure vs Volume plot (the actual fit)
    ax2.scatter(volumes, pressures_gpa, color='blue', s=60, label='Calculated P = -dE/dV')
    ax2.plot(V_smooth, P_smooth, 'r-', linewidth=2, label='Birch-Murnaghan Fit')
    ax2.axvline(results['fit_params']['V0'], color='green', linestyle='--', alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Volume (Å³)')
    ax2.set_ylabel('Pressure (GPa)')
    ax2.set_title('Pressure vs Volume Fit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add results text box
    textstr = f'K₀ = {results["bulk_modulus_gpa"]:.1f} GPa\n'
    textstr += f'K₀\' = {results["fit_params"]["K0_prime"]:.2f}\n'
    textstr += f'R² = {results["r_squared"]:.4f}\n'
    textstr += f'RMSE = {results["rmse"]:.3f} GPa'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'bulk_modulus_fit.png'), dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return results

def quick_bulk_modulus(results):
    """
    Calculate bulk modulus using simple K = V₀ * |dP/dV| formula
    
    Parameters:
    -----------
    results : dict
        Output from fit_birch_murnaghan() function
    
    Returns:
    --------
    dict with quick calculation results
    """
    # Extract data from results
    volumes = results['fit_data']['volumes']
    pressures = results['fit_data']['pressures_gpa']
    V0 = results['fit_params']['V0']
    
    # Calculate bulk modulus: K = V₀ * |dP/dV|
    # Since P decreases as V increases, slope is negative, so we need absolute value
    V_min = volumes.min()
    V_max = volumes.max()
    P_max = pressures.max()
    P_min = pressures.min()
    
    # Simple linear approximation of dP/dV
    dP_dV = (P_max - P_min) / (V_max - V_min)  # This will be negative
    K_quick = V0 * abs(dP_dV)  # Take absolute value to get positive bulk modulus
    
    # Alternative calculation using slope of entire dataset
    slope = np.polyfit(volumes, pressures, 1)[0]  # Linear fit slope
    K_quick_linear = V0 * abs(slope)
    
    quick_results = {
        'bulk_modulus_quick_gpa': K_quick,
        'bulk_modulus_linear_fit_gpa': K_quick_linear,
        'pressure_range_gpa': P_max - P_min,
        'volume_range_a3': V_max - V_min,
        'dP_dV': dP_dV,
        'comparison_with_bm': {
            'birch_murnaghan_gpa': results['bulk_modulus_gpa'],
            'quick_method_gpa': K_quick,
            'difference_gpa': abs(results['bulk_modulus_gpa'] - K_quick),
            'percent_difference': abs(results['bulk_modulus_gpa'] - K_quick) / results['bulk_modulus_gpa'] * 100
        }
    }
    
    print(f"Quick bulk modulus calculation:")
    print(f"  Range method: {K_quick:.1f} GPa")
    print(f"  Linear fit method: {K_quick_linear:.1f} GPa") 
    print(f"  Birch-Murnaghan: {results['bulk_modulus_gpa']:.1f} GPa")
    print(f"  Difference: {quick_results['comparison_with_bm']['difference_gpa']:.1f} GPa ({quick_results['comparison_with_bm']['percent_difference']:.1f}%)")
    
    return quick_results
