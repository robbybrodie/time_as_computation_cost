"""
Bandgaps Experiment: Fit DoF laws and beta parameters from microphysical data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Bootstrap path setup for Colab compatibility
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc.micro.bandgaps import fit_dof_law, fit_beta
from tacc import baselines

def main():
    """Run the bandgaps fitting experiment."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    out_dir = Path(__file__).resolve().parent / "out" / "bandgaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running Bandgaps Fitting Experiment...")
    
    # Generate synthetic data for demonstration
    N_values = np.linspace(0.5, 1.5, 100)
    true_a = 2.0
    DoF_values = np.exp(-true_a * (1 - N_values))
    DoF_values += np.random.normal(0, 0.02, len(DoF_values))
    
    # Fit the DoF law
    fitted_a = fit_dof_law()
    print(f"True parameter a: {true_a:.3f}")
    print(f"Fitted parameter a: {fitted_a:.3f}")
    print(f"Error: {abs(fitted_a - true_a):.3f}")
    
    # Plot DoF law fit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(N_values, DoF_values, alpha=0.6, label='Synthetic Data', s=30)
    N_fit = np.linspace(0.5, 1.5, 200)
    DoF_fit = np.exp(-fitted_a * (1 - N_fit))
    ax1.plot(N_fit, DoF_fit, 'r-', linewidth=2, label=f'Fitted: a={fitted_a:.3f}')
    ax1.set_xlabel('N (Computational Capacity)')
    ax1.set_ylabel('DoF (Degrees of Freedom)')
    ax1.set_title('DoF Law: DoF(N) = exp[-a(1-N)]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Beta parameter fitting
    DoF_range = np.linspace(0.1, 2.0, 100)
    true_beta = 1.5
    psi_values = DoF_range ** true_beta
    psi_values += np.random.normal(0, 0.05, len(psi_values))
    
    fitted_beta = fit_beta()
    print(f"True parameter beta: {true_beta:.3f}")
    print(f"Fitted parameter beta: {fitted_beta:.3f}")
    print(f"Error: {abs(fitted_beta - true_beta):.3f}")
    
    ax2.scatter(DoF_range, psi_values, alpha=0.6, label='Synthetic Data', s=30)
    psi_fit = DoF_range ** fitted_beta
    ax2.plot(DoF_range, psi_fit, 'r-', linewidth=2, label=f'Fitted: β={fitted_beta:.3f}')
    ax2.set_xlabel('DoF (Degrees of Freedom)')
    ax2.set_ylabel('ψ (Field Parameter)')
    ax2.set_title('Power Law: ψ = DoF^β')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "bandgaps_fits.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Compare with baseline models
    print("\nComparing with baseline models...")
    
    # Test polynomial fits
    poly2_params, _ = baselines.fit_baseline(baselines.poly2, N_values, DoF_values)
    poly3_params, _ = baselines.fit_baseline(baselines.poly3, N_values, DoF_values)
    exp_params, _ = baselines.fit_baseline(baselines.exponential, N_values, DoF_values)
    power_params, _ = baselines.fit_baseline(baselines.power_law, N_values, DoF_values)
    
    # Compute predictions
    poly2_pred = baselines.poly2(N_values, *poly2_params)
    poly3_pred = baselines.poly3(N_values, *poly3_params)
    exp_pred = baselines.exponential(N_values, *exp_params)
    power_pred = baselines.power_law(N_values, *power_params)
    
    # Compute AIC/BIC
    models = [
        ("Polynomial-2", poly2_pred, 3),
        ("Polynomial-3", poly3_pred, 4),
        ("Exponential", exp_pred, 2),
        ("Power Law", power_pred, 2)
    ]
    
    results = []
    for name, pred, n_params in models:
        aic, bic = baselines.compute_aic_bic(DoF_values, pred, n_params)
        results.append((name, aic, bic))
        print(f"{name}: AIC={aic:.2f}, BIC={bic:.2f}")
    
    # Plot baseline comparisons
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(N_values, DoF_values, alpha=0.6, label='Data', s=30)
    ax.plot(N_values, poly2_pred, '--', label='Polynomial-2')
    ax.plot(N_values, poly3_pred, '--', label='Polynomial-3')
    ax.plot(N_values, exp_pred, '--', label='Exponential')
    ax.plot(N_values, power_pred, '--', label='Power Law')
    ax.plot(N_fit, DoF_fit, 'r-', linewidth=3, label=f'Our Model (a={fitted_a:.3f})')
    
    ax.set_xlabel('N (Computational Capacity)')
    ax.set_ylabel('DoF (Degrees of Freedom)')
    ax.set_title('Model Comparison: DoF vs N')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "baseline_comparison.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Save results to file
    with open(out_dir / "results.txt", 'w') as f:
        f.write("Bandgaps Fitting Experiment Results\n")
        f.write("===================================\n\n")
        f.write(f"DoF Law Fit: a = {fitted_a:.6f} (true: {true_a})\n")
        f.write(f"Beta Fit: β = {fitted_beta:.6f} (true: {true_beta})\n\n")
        f.write("Baseline Model Comparison (AIC/BIC):\n")
        for name, aic, bic in results:
            f.write(f"{name}: AIC={aic:.2f}, BIC={bic:.2f}\n")
    
    print(f"\nExperiment completed! Results saved to: {out_dir}")
    print("Generated files:")
    print("- bandgaps_fits.png: DoF and beta parameter fits")
    print("- baseline_comparison.png: Model comparison plot")
    print("- results.txt: Numerical results")

if __name__ == "__main__":
    main()
