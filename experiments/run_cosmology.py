"""
Cosmological Expansion and Dark Energy: FLRW metrics adapted to TACC.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Bootstrap path setup for Colab compatibility
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc import cosmology, constitutive

def main():
    """Run the cosmological expansion experiment."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    out_dir = Path(__file__).resolve().parent / "out" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running Cosmological Expansion and Dark Energy Experiment...")
    
    # Generate synthetic supernova data
    print("Generating synthetic supernova data...")
    z_data, mu_data, mu_err = cosmology.generate_synthetic_supernova_data(
        z_max=2.0, n_points=100, kappa_true=2.0, noise_level=0.15
    )
    
    # Test different κ values on distance modulus
    kappa_values = np.linspace(0.5, 4.0, 20)
    z_test = np.linspace(0.01, 2.0, 100)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distance modulus vs redshift for different κ
    colors = plt.cm.viridis(np.linspace(0, 1, len(kappa_values)))
    for i, kappa in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]):
        mu_theory = cosmology.distance_modulus_tacc(z_test, kappa)
        ax1.plot(z_test, mu_theory, color=colors[i*3], linewidth=2, label=f'κ={kappa}')
    
    # Add synthetic data points
    ax1.errorbar(z_data[:20], mu_data[:20], yerr=mu_err[:20], fmt='ko', alpha=0.7, 
                 markersize=4, label='Synthetic SN Data')
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Distance Modulus μ (mag)')
    ax1.set_title('TACC Distance-Redshift Relation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 2])
    ax1.set_ylim([30, 50])
    
    # Hubble parameter evolution
    z_hubble = np.linspace(0, 3.0, 100)
    for i, kappa in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
        H_z = np.array([cosmology.hubble_parameter_tacc(z, kappa) for z in z_hubble])
        ax2.plot(z_hubble, H_z, color=colors[i*3], linewidth=2, label=f'κ={kappa}')
    
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('H(z) (km/s/Mpc)')
    ax2.set_title('Hubble Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 3])
    
    # Scale factor evolution
    t_cosmic = np.linspace(1, 13.8, 100)  # Cosmic time in Gyr
    for i, kappa in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
        try:
            a_t = cosmology.scale_factor_evolution(t_cosmic, kappa)
            ax3.plot(t_cosmic, a_t, color=colors[i*3], linewidth=2, label=f'κ={kappa}')
        except:
            # Skip if integration fails for extreme parameters
            continue
    
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Present (a=1)')
    ax3.set_xlabel('Cosmic Time (Gyr)')
    ax3.set_ylabel('Scale Factor a(t)')
    ax3.set_title('Scale Factor Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([1, 13.8])
    ax3.set_ylim([0, 1.2])
    
    # Constitutive law interpretation
    N_range = np.linspace(0.2, 1.5, 200)
    for i, kappa in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
        B_values = constitutive.B_of_N(N_range, kappa)
        ax4.plot(N_range, B_values, color=colors[i*3], linewidth=2, label=f'κ={kappa}')
    
    ax4.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='N=1 (Present)')
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('N (Computational Capacity)')
    ax4.set_ylabel('B(N) = exp[-κ(1-N)]')
    ax4.set_title('Constitutive Law: Cosmological Interpretation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0.2, 1.5])
    
    plt.tight_layout()
    plt.savefig(out_dir / "cosmological_expansion.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Fit to synthetic data
    print("Fitting TACC cosmology to synthetic supernova data...")
    fit_result = cosmology.fit_to_supernova_data(z_data, mu_data, mu_err)
    
    print(f"Best-fit parameters:")
    print(f"  κ = {fit_result['kappa']:.3f}")
    print(f"  H₀ = {fit_result['H0']:.1f} km/s/Mpc")
    print(f"  Ωₘ = {fit_result['Omega_m']:.3f}")
    print(f"  Ωᵈᵉ = {fit_result['Omega_de']:.3f}")
    print(f"  χ²/dof = {fit_result['chi2_red']:.2f}")
    
    # Create fit comparison plot
    z_fine = np.linspace(0.01, max(z_data), 200)
    mu_best_fit = cosmology.distance_modulus_tacc(
        z_fine, fit_result['kappa'], fit_result['H0'], 
        fit_result['Omega_m'], fit_result['Omega_de']
    )
    mu_lcdm = cosmology.distance_modulus_tacc(z_fine, 2.0)  # ΛCDM comparison
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Data vs fits
    ax1.errorbar(z_data, mu_data, yerr=mu_err, fmt='ko', alpha=0.7, 
                 markersize=4, label='Synthetic Data')
    ax1.plot(z_fine, mu_best_fit, 'r-', linewidth=2, 
             label=f'TACC Best Fit (κ={fit_result["kappa"]:.2f})')
    ax1.plot(z_fine, mu_lcdm, 'b--', linewidth=2, label='ΛCDM (κ=2.0)')
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Distance Modulus μ (mag)')
    ax1.set_title('TACC Cosmology Fit to Supernova Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    mu_data_interp = np.interp(z_fine, z_data, mu_data)
    residuals_tacc = mu_data_interp - mu_best_fit
    residuals_lcdm = mu_data_interp - mu_lcdm
    
    ax2.plot(z_fine, residuals_tacc, 'r-', linewidth=2, label='TACC Residuals')
    ax2.plot(z_fine, residuals_lcdm, 'b--', linewidth=2, label='ΛCDM Residuals')
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Residuals (mag)')
    ax2.set_title('Fit Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "supernova_fit.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Parameter space exploration
    print("Exploring parameter space...")
    kappa_grid = np.linspace(0.5, 4.0, 50)
    chi2_values = []
    
    for kappa in kappa_grid:
        mu_theory = cosmology.distance_modulus_tacc(z_data, kappa)
        residuals = (mu_data - mu_theory) / mu_err
        chi2 = np.sum(residuals**2)
        chi2_values.append(chi2)
    
    chi2_values = np.array(chi2_values)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(kappa_grid, chi2_values, 'b-', linewidth=2)
    ax.axvline(x=fit_result['kappa'], color='r', linestyle='--', 
               label=f'Best fit: κ={fit_result["kappa"]:.2f}')
    ax.axvline(x=2.0, color='g', linestyle=':', label='GR limit: κ=2.0')
    
    # 1-σ and 2-σ confidence levels
    chi2_min = np.min(chi2_values)
    ax.axhline(y=chi2_min + 1, color='orange', linestyle=':', alpha=0.7, label='1σ')
    ax.axhline(y=chi2_min + 4, color='orange', linestyle=':', alpha=0.5, label='2σ')
    
    ax.set_xlabel('κ (Constitutive Parameter)')
    ax.set_ylabel('χ²')
    ax.set_title('Parameter Constraint from Supernova Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 4.0])
    
    plt.tight_layout()
    plt.savefig(out_dir / "parameter_constraints.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Dark energy interpretation
    z_de = np.linspace(0, 2.0, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Effective dark energy density
    for kappa in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        rho_de_eff = []
        for z in z_de:
            a = 1.0 / (1.0 + z)
            B_factor = constitutive.B_of_N(a, kappa)
            rho_de_eff.append(0.7 * B_factor)  # Assuming Ω_Λ = 0.7
        
        ax1.plot(z_de, rho_de_eff, linewidth=2, label=f'κ={kappa}')
    
    ax1.axhline(y=0.7, color='k', linestyle='--', alpha=0.5, label='ΛCDM')
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Effective Dark Energy Density')
    ax1.set_title('TACC Dark Energy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Deceleration parameter
    for kappa in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        q_values = []
        for z in z_de:
            # Approximate deceleration parameter
            H_z = cosmology.hubble_parameter_tacc(z, kappa)
            # q = -ä/aH² (simplified calculation)
            if z < 2.0:
                H_z_plus = cosmology.hubble_parameter_tacc(z + 0.01, kappa)
                H_z_minus = cosmology.hubble_parameter_tacc(max(z - 0.01, 0), kappa)
                dH_dz = (H_z_plus - H_z_minus) / 0.02
                q = (1 + z) * dH_dz / H_z - 1
                q_values.append(q)
            else:
                q_values.append(0)  # Fallback
        
        ax2.plot(z_de[:-1], q_values[:-1], linewidth=2, label=f'κ={kappa}')
    
    ax2.axhline(y=-0.5, color='k', linestyle='--', alpha=0.5, label='ΛCDM Today')
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='q=0')
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Deceleration Parameter q')
    ax2.set_title('TACC Deceleration Parameter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-2, 2])
    
    plt.tight_layout()
    plt.savefig(out_dir / "dark_energy_evolution.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Save results to file
    with open(out_dir / "results.txt", 'w') as f:
        f.write("TACC Cosmological Expansion Results\n")
        f.write("===================================\n\n")
        f.write(f"Best-fit parameters from synthetic supernova data:\n")
        f.write(f"  κ = {fit_result['kappa']:.6f}\n")
        f.write(f"  H₀ = {fit_result['H0']:.3f} km/s/Mpc\n")
        f.write(f"  Ωₘ = {fit_result['Omega_m']:.6f}\n")
        f.write(f"  Ωᵈᵉ = {fit_result['Omega_de']:.6f}\n")
        f.write(f"  χ² = {fit_result['chi2']:.3f}\n")
        f.write(f"  χ²/dof = {fit_result['chi2_red']:.3f}\n")
        f.write(f"  Degrees of freedom = {fit_result['dof']}\n")
        f.write(f"  Fit successful: {fit_result['success']}\n\n")
        
        f.write("Physical Interpretation:\n")
        f.write("- κ controls how computational capacity affects cosmic expansion\n")
        f.write("- κ = 2.0 recovers standard ΛCDM cosmology\n")
        f.write("- κ ≠ 2.0 predicts deviations in distance-redshift relation\n")
        f.write("- TACC provides natural dark energy through computational constraints\n\n")
        
        f.write("Key Insights:\n")
        f.write("- Scale factor a(t) is interpreted as computational capacity N\n")
        f.write("- B(N) = exp[-κ(1-N)] modifies Friedmann equation\n")
        f.write("- Early universe (small a) had reduced computational capacity\n")
        f.write("- Present acceleration emerges from capacity constraints\n")
    
    print(f"\nCosmology experiment completed! Results saved to: {out_dir}")
    print("Generated files:")
    print("- cosmological_expansion.png: Distance modulus, Hubble parameter, scale factor")
    print("- supernova_fit.png: Fit to synthetic supernova data")
    print("- parameter_constraints.png: χ² constraints on κ")
    print("- dark_energy_evolution.png: Dark energy and deceleration parameter")
    print("- results.txt: Numerical results and physical interpretation")

if __name__ == "__main__":
    main()
