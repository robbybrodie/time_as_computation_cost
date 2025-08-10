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
    
    # Run experiment
    results = run_experiment(
        n_points=50,
        noise_sigma=0.05,
        a_true=2.0,
        beta_true=1.5,
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
