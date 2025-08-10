"""
PPN Parameter Extraction: Extract gamma and beta from metric and test against GR predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Bootstrap path setup for Colab compatibility
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc import ppn, constitutive

def main():
    """Run the PPN parameter extraction experiment."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    out_dir = Path(__file__).resolve().parent / "out" / "ppn"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running PPN Parameter Extraction Experiment...")
    
    # Test different values of kappa
    kappa_values = np.linspace(0.5, 3.0, 20)
    gamma_values = []
    beta_values = []
    
    for kappa in kappa_values:
        gamma, beta = ppn.extract_ppn_params(kappa)
        gamma_values.append(gamma)
        beta_values.append(beta)
    
    # Plot PPN parameters vs kappa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(kappa_values, gamma_values, 'bo-', linewidth=2, markersize=6)
    ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='GR Prediction (γ=1)')
    ax1.set_xlabel('κ (Constitutive Parameter)')
    ax1.set_ylabel('γ (PPN Parameter)')
    ax1.set_title('PPN γ vs κ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(kappa_values, beta_values, 'ro-', linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='GR Prediction (β=1)')
    ax2.set_xlabel('κ (Constitutive Parameter)')
    ax2.set_ylabel('β (PPN Parameter)')
    ax2.set_title('PPN β vs κ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "ppn_parameters.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Test N to Phi mapping
    N_values = np.linspace(0.95, 1.05, 100)
    Phi_values = ppn.expand_N_to_Phi(N_values)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(N_values, Phi_values, 'b-', linewidth=2)
    ax.set_xlabel('N (Computational Capacity)')
    ax.set_ylabel('Φ/c² (Newtonian Potential)')
    ax.set_title('Near-Unity Expansion: N ≈ 1 + Φ/c²')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "n_to_phi_mapping.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Test special relativity limit
    velocities = np.linspace(0, 0.9, 50)  # as fraction of c
    sr_factors = [ppn.sr_limit(v, c=1.0) for v in velocities]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(velocities, sr_factors, 'g-', linewidth=2, label='dτ/dt ≈ 1 - v²/(2c²)')
    
    # Compare with exact Lorentz factor for reference
    exact_factors = 1 / np.sqrt(1 - velocities**2)
    ax.plot(velocities, 1/exact_factors, 'r--', linewidth=2, label='Exact: 1/γ')
    
    ax.set_xlabel('v/c')
    ax.set_ylabel('Time Dilation Factor')
    ax.set_title('Special Relativity Limit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])
    
    plt.tight_layout()
    plt.savefig(out_dir / "sr_limit.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Test B(N) behavior for different kappa values
    N_range = np.linspace(0.5, 2.0, 200)
    test_kappas = [0.5, 1.0, 2.0, 3.0]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for kappa in test_kappas:
        B_values = constitutive.B_of_N(N_range, kappa)
        ax.plot(N_range, B_values, linewidth=2, label=f'κ={kappa}')
    
    ax.axvline(x=1.0, color='k', linestyle=':', alpha=0.7, label='N=1 (GR limit)')
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.7)
    ax.set_xlabel('N (Computational Capacity)')
    ax.set_ylabel('B(N) = exp[-κ(1-N)]')
    ax.set_title('Constitutive Law B(N) for Different κ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "constitutive_law.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    print("\nPPN Parameter Analysis:")
    print("======================")
    for i, kappa in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
        gamma, beta = ppn.extract_ppn_params(kappa)
        print(f"κ={kappa:.1f}: γ={gamma:.3f}, β={beta:.3f}")
        
        # Check deviation from GR
        gamma_dev = abs(gamma - 1.0)
        beta_dev = abs(beta - 1.0)
        print(f"         Deviation from GR: Δγ={gamma_dev:.3f}, Δβ={beta_dev:.3f}")
    
    # Find kappa value that gives gamma closest to 1
    optimal_kappa = 2.0  # gamma = kappa/2, so kappa=2 gives gamma=1
    gamma_opt, beta_opt = ppn.extract_ppn_params(optimal_kappa)
    print(f"\nOptimal κ for GR matching: κ={optimal_kappa}")
    print(f"Resulting parameters: γ={gamma_opt:.6f}, β={beta_opt:.6f}")
    
    # Save results to file
    with open(out_dir / "results.txt", 'w') as f:
        f.write("PPN Parameter Extraction Results\n")
        f.write("===============================\n\n")
        f.write("PPN parameters for different κ values:\n")
        for kappa in kappa_values:
            gamma, beta = ppn.extract_ppn_params(kappa)
            f.write(f"κ={kappa:.2f}: γ={gamma:.4f}, β={beta:.4f}\n")
        
        f.write(f"\nOptimal κ for GR matching: {optimal_kappa}\n")
        f.write(f"Optimal γ: {gamma_opt:.6f} (GR: 1.0)\n")
        f.write(f"Optimal β: {beta_opt:.6f} (GR: 1.0)\n")
        
        f.write("\nNote: In our model, β=1 always (built into the theory)\n")
        f.write("γ = κ/2, so κ=2 gives exact GR limit γ=1\n")
    
    print(f"\nExperiment completed! Results saved to: {out_dir}")
    print("Generated files:")
    print("- ppn_parameters.png: γ and β vs κ")
    print("- n_to_phi_mapping.png: N to Newtonian potential mapping")
    print("- sr_limit.png: Special relativity limit test")
    print("- constitutive_law.png: B(N) for different κ values")
    print("- results.txt: Numerical results and analysis")

if __name__ == "__main__":
    main()
