"""
Black Hole Thermodynamics: Schwarzschild/Kerr metrics adapted to TACC.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Bootstrap path setup for Colab compatibility
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc import black_holes, constants

def main():
    """Run the black hole thermodynamics experiment."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    out_dir = Path(__file__).resolve().parent / "out" / "black_holes"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running Black Hole Thermodynamics Experiment...")
    
    # Solar mass for reference
    M_sun = 1.989e30  # kg
    
    # Test different black hole masses
    mass_range = np.logspace(0, 9, 100) * M_sun  # 1 to 10^9 solar masses
    kappa_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Hawking temperature analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(kappa_values)))
    
    # Temperature vs mass for different κ
    for i, kappa in enumerate(kappa_values):
        T_hawking = np.array([black_holes.hawking_temperature_tacc(M, kappa) for M in mass_range])
        ax1.loglog(mass_range/M_sun, T_hawking, color=colors[i], linewidth=2, label=f'κ={kappa}')
    
    # GR comparison
    T_hawking_gr = np.array([black_holes.hawking_temperature_gr(M) for M in mass_range])
    ax1.loglog(mass_range/M_sun, T_hawking_gr, 'k--', linewidth=2, label='GR')
    
    ax1.set_xlabel('Mass (M☉)')
    ax1.set_ylabel('Hawking Temperature (K)')
    ax1.set_title('TACC Black Hole Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Entropy vs mass
    for i, kappa in enumerate(kappa_values):
        S_bh = np.array([black_holes.bekenstein_hawking_entropy_tacc(M, kappa) for M in mass_range])
        ax2.loglog(mass_range/M_sun, S_bh/constants.k_B, color=colors[i], linewidth=2, label=f'κ={kappa}')
    
    # GR comparison
    S_bh_gr = np.array([constants.k_B * constants.c**3 * (4*np.pi*black_holes.schwarzschild_radius(M)**2) / (4*constants.G*constants.hbar) for M in mass_range])
    ax2.loglog(mass_range/M_sun, S_bh_gr/constants.k_B, 'k--', linewidth=2, label='GR')
    
    ax2.set_xlabel('Mass (M☉)')
    ax2.set_ylabel('Entropy (k_B units)')
    ax2.set_title('TACC Black Hole Entropy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Evaporation time vs mass
    for i, kappa in enumerate(kappa_values):
        t_evap = np.array([black_holes.black_hole_evaporation_time_tacc(M, kappa) for M in mass_range])
        # Convert to years
        t_evap_years = t_evap / (365.25 * 24 * 3600)
        ax3.loglog(mass_range/M_sun, t_evap_years, color=colors[i], linewidth=2, label=f'κ={kappa}')
    
    # GR comparison
    t_evap_gr = np.array([5120 * np.pi * constants.G**2 * M**3 / (constants.hbar * constants.c**4) for M in mass_range])
    t_evap_gr_years = t_evap_gr / (365.25 * 24 * 3600)
    ax3.loglog(mass_range/M_sun, t_evap_gr_years, 'k--', linewidth=2, label='GR')
    
    # Add cosmological age reference
    t_universe = 13.8e9  # years
    ax3.axhline(y=t_universe, color='r', linestyle=':', alpha=0.7, label='Age of Universe')
    
    ax3.set_xlabel('Mass (M☉)')
    ax3.set_ylabel('Evaporation Time (years)')
    ax3.set_title('TACC Black Hole Evaporation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([1e30, 1e100])
    
    # Temperature deviations from GR
    M_test = 10 * M_sun  # 10 solar mass black hole
    kappa_fine = np.linspace(0.1, 5.0, 200)
    
    T_deviations = []
    for kappa in kappa_fine:
        T_tacc = black_holes.hawking_temperature_tacc(M_test, kappa)
        T_gr = black_holes.hawking_temperature_gr(M_test)
        deviation = (T_tacc - T_gr) / T_gr
        T_deviations.append(deviation)
    
    ax4.plot(kappa_fine, T_deviations, 'b-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='GR')
    ax4.axvline(x=2.0, color='r', linestyle=':', alpha=0.7, label='κ=2 (GR)')
    ax4.set_xlabel('κ (Constitutive Parameter)')
    ax4.set_ylabel('Fractional Temperature Deviation')
    ax4.set_title(f'Temperature Deviation (M = 10 M☉)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "black_hole_thermodynamics.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # M87* black hole analysis
    print("Analyzing M87* black hole...")
    m87_params = black_holes.m87_black_hole_parameters()
    
    # Compute properties for different κ values
    m87_results = {}
    for kappa in kappa_values:
        properties = black_holes.event_horizon_properties_tacc(m87_params['mass'], kappa)
        m87_results[kappa] = properties
    
    # Shadow size comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Shadow angular size vs κ
    shadow_angles = []
    for kappa in kappa_fine:
        theta = black_holes.black_hole_shadow_angular_size(
            m87_params['mass'], m87_params['distance'], kappa
        )
        # Convert to microarcseconds
        theta_uas = theta * (180/np.pi) * 3600 * 1e6
        shadow_angles.append(theta_uas)
    
    ax1.plot(kappa_fine, shadow_angles, 'b-', linewidth=2, label='TACC Prediction')
    
    # Observed value with error bar
    obs_angle_uas = m87_params['shadow_angle'] * (180/np.pi) * 3600 * 1e6
    obs_error_uas = m87_params['shadow_error'] * (180/np.pi) * 3600 * 1e6
    ax1.axhline(y=obs_angle_uas, color='r', linestyle='--', linewidth=2, label='EHT Observation')
    ax1.fill_between(kappa_fine, obs_angle_uas - obs_error_uas, obs_angle_uas + obs_error_uas,
                     alpha=0.3, color='red', label='Uncertainty')
    ax1.axvline(x=2.0, color='g', linestyle=':', alpha=0.7, label='GR (κ=2)')
    
    ax1.set_xlabel('κ (Constitutive Parameter)')
    ax1.set_ylabel('Shadow Angular Size (μas)')
    ax1.set_title('M87* Shadow Size vs κ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.1, 5.0])
    
    # Metric comparison at different radii
    r_range = np.logspace(0, 2, 100) * black_holes.schwarzschild_radius(m87_params['mass'])
    
    for i, kappa in enumerate([0.5, 1.0, 2.0, 3.0]):
        g_tt, g_rr = black_holes.tacc_metric_schwarzschild(r_range, m87_params['mass'], kappa)
        ax2.semilogx(r_range / black_holes.schwarzschild_radius(m87_params['mass']), 
                     -g_tt, color=colors[i], linewidth=2, label=f'κ={kappa}')
    
    ax2.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='Event Horizon')
    ax2.set_xlabel('r / r_s')
    ax2.set_ylabel('-g_tt (metric coefficient)')
    ax2.set_title('TACC Schwarzschild Metric')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, 100])
    
    plt.tight_layout()
    plt.savefig(out_dir / "m87_analysis.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Comparative analysis across mass scales
    print("Performing comparative analysis...")
    comparison_masses = [1, 10, 100, 1e6, 1e9] * np.array([M_sun])  # Different scales
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Temperature scaling
    for i, M in enumerate(comparison_masses):
        temperatures = []
        for kappa in kappa_fine:
            T = black_holes.hawking_temperature_tacc(M, kappa)
            temperatures.append(T)
        
        ax1.plot(kappa_fine, temperatures, linewidth=2, 
                label=f'{M/M_sun:.0e} M☉')
    
    ax1.axvline(x=2.0, color='k', linestyle=':', alpha=0.7, label='GR')
    ax1.set_xlabel('κ (Constitutive Parameter)')
    ax1.set_ylabel('Hawking Temperature (K)')
    ax1.set_title('Temperature Scaling with Mass')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Entropy scaling
    for i, M in enumerate(comparison_masses):
        entropies = []
        for kappa in kappa_fine:
            S = black_holes.bekenstein_hawking_entropy_tacc(M, kappa)
            entropies.append(S / constants.k_B)
        
        ax2.plot(kappa_fine, entropies, linewidth=2, 
                label=f'{M/M_sun:.0e} M☉')
    
    ax2.axvline(x=2.0, color='k', linestyle=':', alpha=0.7, label='GR')
    ax2.set_xlabel('κ (Constitutive Parameter)')
    ax2.set_ylabel('Entropy (k_B units)')
    ax2.set_title('Entropy Scaling with Mass')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Information capacity interpretation
    # Horizon computational capacity vs κ
    horizon_N = []
    horizon_B = []
    for kappa in kappa_fine:
        # Use stellar mass black hole as example
        M_stellar = 10 * M_sun
        r_s = black_holes.schwarzschild_radius(M_stellar)
        Phi_horizon = -constants.G * M_stellar / r_s
        N_h = 1.0 + Phi_horizon / constants.c**2  # This is 0.5
        B_h = black_holes.constitutive.B_of_N(N_h, kappa)
        horizon_N.append(N_h)
        horizon_B.append(B_h)
    
    ax3.plot(kappa_fine, horizon_B, 'b-', linewidth=2)
    ax3.axvline(x=2.0, color='r', linestyle=':', alpha=0.7, label='GR (κ=2)')
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('κ (Constitutive Parameter)')
    ax3.set_ylabel('B(N_horizon)')
    ax3.set_title('Horizon Computational Capacity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Information vs thermodynamic entropy
    M_info = 10 * M_sun
    info_entropies = []
    thermo_entropies = []
    
    for kappa in kappa_fine:
        # Information-theoretic entropy (computational perspective)
        properties = black_holes.event_horizon_properties_tacc(M_info, kappa)
        S_info = properties['entropy']
        
        # Standard thermodynamic entropy
        S_thermo = constants.k_B * constants.c**3 * properties['area'] / (4 * constants.G * constants.hbar)
        
        info_entropies.append(S_info / constants.k_B)
        thermo_entropies.append(S_thermo / constants.k_B)
    
    ax4.plot(kappa_fine, info_entropies, 'b-', linewidth=2, label='TACC (Info-theoretic)')
    ax4.plot(kappa_fine, thermo_entropies, 'r--', linewidth=2, label='Standard Thermodynamic')
    ax4.axvline(x=2.0, color='g', linestyle=':', alpha=0.7, label='GR (κ=2)')
    ax4.set_xlabel('κ (Constitutive Parameter)')
    ax4.set_ylabel('Entropy (k_B units)')
    ax4.set_title('Information vs Thermodynamic Entropy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "comparative_analysis.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Statistical analysis
    print("Computing statistical comparisons...")
    
    # Compare TACC vs GR for different masses
    comparison_results = {}
    for M in comparison_masses:
        gr_comparison = black_holes.compare_with_gr(M, kappa_values)
        comparison_results[M/M_sun] = gr_comparison
    
    # Save detailed results
    with open(out_dir / "results.txt", 'w') as f:
        f.write("TACC Black Hole Thermodynamics Results\n")
        f.write("======================================\n\n")
        
        f.write("M87* Black Hole Analysis:\n")
        f.write(f"  Mass: {m87_params['mass']/M_sun:.1e} M☉\n")
        f.write(f"  Distance: {m87_params['distance']/9.461e15/1e6:.1f} Mpc\n")
        f.write(f"  Observed shadow: {obs_angle_uas:.1f} ± {obs_error_uas:.1f} μas\n\n")
        
        f.write("TACC Properties for different κ values (M87*):\n")
        for kappa in kappa_values:
            props = m87_results[kappa]
            f.write(f"  κ={kappa}:\n")
            f.write(f"    Temperature: {props['temperature']:.2e} K\n")
            f.write(f"    Entropy: {props['entropy']/constants.k_B:.2e} k_B\n")
            f.write(f"    Evaporation time: {props['evaporation_time']/(365.25*24*3600*1e9):.2e} Gyr\n")
            f.write(f"    N_horizon: {props['N_horizon']:.3f}\n")
            f.write(f"    B_horizon: {props['B_horizon']:.3f}\n\n")
        
        f.write("Physical Interpretation:\n")
        f.write("- κ controls strength of computational capacity effects at horizon\n")
        f.write("- N_horizon = 0.5 for all black holes (universal result)\n")
        f.write("- B(N_horizon) = exp[-κ(1-0.5)] = exp[-κ/2] modifies thermodynamics\n")
        f.write("- Information-theoretic entropy links to computational limits\n")
        f.write("- Shadow size provides observational test of κ parameter\n\n")
        
        f.write("Key Insights:\n")
        f.write("- Black hole horizon represents maximum computational constraint\n")
        f.write("- Hawking radiation modified by computational capacity limits\n")
        f.write("- Information paradox potentially resolved through capacity bounds\n")
        f.write("- EHT observations can constrain TACC parameter space\n")
    
    print(f"\nBlack holes experiment completed! Results saved to: {out_dir}")
    print("Generated files:")
    print("- black_hole_thermodynamics.png: Temperature, entropy, evaporation time")
    print("- m87_analysis.png: M87* shadow size and metric comparison")
    print("- comparative_analysis.png: Multi-scale comparison and information theory")
    print("- results.txt: Detailed numerical results and physical interpretation")

if __name__ == "__main__":
    main()
