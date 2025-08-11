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
        
        # Check for individual experiment toggle (from notebook)
        use_fitment = True
        try:
            if 'experiment_toggles' in globals() and globals()['experiment_toggles'] is not None:
                toggle = globals()['experiment_toggles']['tension_bandgaps']
                use_fitment = toggle.value
                print(f"🎛️ Individual toggle: {'ON' if use_fitment else 'OFF'} for Tension Bandgaps")
        except:
            pass  # Fall back to global fitment setting
        
        if is_fitment_active() and use_fitment:
            fitment_info = get_active_fitment_info()
            fitted_kappa = get_fitted_kappa()
            
            print(f"🎯 USING FITMENT: {fitment_info['name']}")
            print(f"   Fitted κ: {fitted_kappa:.4f}")
            print("   This affects synthetic data generation and fitting targets!")
            
            # Use fitted kappa with MORE DRAMATIC parameter changes
            # kappa affects the underlying physics - make changes more visible
            fitted_a_true = fitted_kappa * 1.5  # Amplify the effect
            fitted_beta_true = default_beta_true * (fitted_kappa / 2.0) ** 1.5  # Power scaling
            
            # More dramatic data quality changes to make effects visible
            kappa_deviation = abs(fitted_kappa - 2.0)
            fitted_noise_sigma = default_noise_sigma * (1.0 + kappa_deviation * 2.0)  # 2x noise scaling
            
            # Also adjust sample size based on kappa - more complex datasets for higher kappa
            fitted_n_points = int(default_n_points * (1.0 + (fitted_kappa - 2.0) * 0.3))
            fitted_n_points = max(20, min(100, fitted_n_points))  # Keep reasonable bounds
            
            print(f"   Adjusted a_true: {fitted_a_true:.4f} (default: {default_a_true})")
            print(f"   a_true scaling: 1.5x κ for amplified physics effects")
            print(f"   Adjusted β_true: {fitted_beta_true:.4f} (default: {default_beta_true})")
            print(f"   β_true scaling: power law (κ/2)^1.5 for nonlinear effects")
            print(f"   Adjusted noise σ: {fitted_noise_sigma:.4f} (default: {default_noise_sigma})")
            print(f"   Noise scaling: 2x deviation impact for visible data quality changes")
            print(f"   Adjusted n_points: {fitted_n_points} (default: {default_n_points})")
            print(f"   Sample size scales with κ for complexity matching")
            
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
    
    # Print summary with fitment impact analysis
    print(f"\nExperiment completed!")
    print(f"Parameters: n_points={results['parameters']['n_points']}, noise_sigma={results['parameters']['noise_sigma']}")
    print(f"True values: a_true={results['parameters']['a_true']}, beta_true={results['parameters']['beta_true']}")
    print(f"\nFitted parameters:")
    print(f"  a_hat = {results['fitted_params']['a_hat']:.3f}")
    print(f"  beta_hat = {results['fitted_params']['beta_hat']:.3f}")
    
    # Show fitment impact comparison
    try:
        from tacc.core.experiment_bridge import is_fitment_active, get_fitted_kappa
        if is_fitment_active():
            fitted_kappa = get_fitted_kappa()
            print(f"\n🎯 FITMENT IMPACT ANALYSIS:")
            print(f"   Fitted κ: {fitted_kappa:.4f}")
            
            # Compare with defaults
            a_change_pct = ((fitted_a_true - default_a_true) / default_a_true) * 100
            beta_change_pct = ((fitted_beta_true - default_beta_true) / default_beta_true) * 100
            noise_change_pct = ((fitted_noise_sigma - default_noise_sigma) / default_noise_sigma) * 100
            n_points_change_pct = ((fitted_n_points - default_n_points) / default_n_points) * 100
            
            print(f"   a_true change: {a_change_pct:+.1f}% (1.5x κ amplification)")
            print(f"   β_true change: {beta_change_pct:+.1f}% ((κ/2)^1.5 power scaling)")
            print(f"   Noise σ change: {noise_change_pct:+.1f}% (2x deviation impact)")
            print(f"   Sample size change: {n_points_change_pct:+.1f}% (0.3x κ deviation scaling)")
            
            # Show fitting quality metrics
            fitting_error_a = abs(results['fitted_params']['a_hat'] - fitted_a_true) / fitted_a_true * 100
            fitting_error_beta = abs(results['fitted_params']['beta_hat'] - fitted_beta_true) / fitted_beta_true * 100
            
            print(f"   Fitting accuracy: a_hat error {fitting_error_a:.1f}%, β_hat error {fitting_error_beta:.1f}%")
            print(f"   Expected effects: Different data patterns, noise levels, and fitting challenges")
            
    except ImportError:
        pass
    
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
