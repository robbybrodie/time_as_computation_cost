"""
Tension Bandgaps Experiment Runner
"""

import sys
from pathlib import Path

# Bootstrap path setup for Colab compatibility
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc.bandgaps.tension import run_demo, run_experiment


def main():
    """Run the tension bandgaps experiment."""
    print("Running Tension Bandgaps Experiment...")
    print("=" * 50)
    
    # Set up output directory
    output_dir = Path(__file__).resolve().parent / "out" / "tension_bandgaps"
    
    # Check for fitment state from interactive widgets
    default_a_true = 2.0
    default_beta_true = 1.5
    default_n_points = 50
    default_noise_sigma = 0.05
    
    try:
        from tacc.core.experiment_bridge import get_fitted_kappa, is_fitment_active, get_active_fitment_info
        
        if is_fitment_active():
            fitment_info = get_active_fitment_info()
            fitted_kappa = get_fitted_kappa()
            
            print(f"🎯 USING FITMENT: {fitment_info['name']}")
            print(f"   Fitted κ: {fitted_kappa:.4f}")
            print("   This affects synthetic data generation and fitting targets!")
            
            # Use fitted kappa as the primary parameter for synthetic data
            # kappa affects the underlying physics, so use it for a_true
            fitted_a_true = fitted_kappa
            # Scale beta based on kappa ratio
            fitted_beta_true = default_beta_true * (fitted_kappa / 2.0)
            
            # Adjust data quality based on how far from default we are
            kappa_deviation = abs(fitted_kappa - 2.0)
            fitted_noise_sigma = default_noise_sigma * (1.0 + kappa_deviation * 0.5)
            
            print(f"   Adjusted a_true: {fitted_a_true:.4f} (default: {default_a_true})")
            print(f"   Adjusted β_true: {fitted_beta_true:.4f} (default: {default_beta_true})")
            print(f"   Adjusted noise σ: {fitted_noise_sigma:.4f} (default: {default_noise_sigma})")
            
            fitted_n_points = default_n_points
            
        else:
            print("🔧 No active fitment - using default parameters")
            fitted_a_true = default_a_true
            fitted_beta_true = default_beta_true
            fitted_n_points = default_n_points
            fitted_noise_sigma = default_noise_sigma
            
    except ImportError:
        print("🔧 Fitment bridge not available - using default parameters")
        fitted_a_true = default_a_true
        fitted_beta_true = default_beta_true
        fitted_n_points = default_n_points
        fitted_noise_sigma = default_noise_sigma
    
    # Run experiment
    results = run_experiment(
        n_points=fitted_n_points,
        noise_sigma=fitted_noise_sigma,
        a_true=fitted_a_true,
        beta_true=fitted_beta_true,
        output_dir=str(output_dir)
    )
    
    # Print summary
    print(f"\nExperiment completed!")
    print(f"Parameters: n_points={results['parameters']['n_points']}, noise_sigma={results['parameters']['noise_sigma']}")
    print(f"True values: a_true={results['parameters']['a_true']}, beta_true={results['parameters']['beta_true']}")
    print(f"\nFitted parameters:")
    print(f"  a_hat = {results['fitted_params']['a_hat']:.3f}")
    print(f"  beta_hat = {results['fitted_params']['beta_hat']:.3f}")
    
    print(f"\nModel comparison (lower is better):")
    for model_name, metrics in results['model_comparison'].items():
        print(f"  {model_name.capitalize()}:")
        print(f"    AIC: {metrics['aic']:.2f}")
        print(f"    BIC: {metrics['bic']:.2f}")
        print(f"    CV MSE: {metrics['cv_score']:.4f} ± {metrics['cv_std']:.4f}")
    
    # Determine best model
    best_aic = min(results['model_comparison'], 
                   key=lambda x: results['model_comparison'][x]['aic'])
    print(f"\nBest model (by AIC): {best_aic.capitalize()}")
    
    if 'files' in results:
        print(f"\nOutput files:")
        for name, path in results['files'].items():
            print(f"  {name}: {path}")
    
    return results


if __name__ == "__main__":
    main()
