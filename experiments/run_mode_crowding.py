"""
Mode Crowding Experiment Runner
"""

import sys
from pathlib import Path

# Bootstrap path setup for Colab compatibility
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc.crowding.mode_crowding import run_demo, run_experiment


def main():
    """Run the mode crowding experiment."""
    print("Running Mode Crowding Experiment...")
    print("=" * 50)
    
    # Set up output directory
    output_dir = Path(__file__).resolve().parent / "out" / "mode_crowding"
    
    # Check for fitment state from interactive widgets
    default_K = 10
    default_N_max = 1.0
    default_N_min = 0.01
    default_n_points = 100
    default_threshold = 0.1
    
    try:
        from tacc.core.experiment_bridge import get_fitted_kappa, is_fitment_active, get_active_fitment_info
        
        if is_fitment_active():
            fitment_info = get_active_fitment_info()
            fitted_kappa = get_fitted_kappa()
            
            print(f"🎯 USING FITMENT: {fitment_info['name']}")
            print(f"   Fitted κ: {fitted_kappa:.4f}")
            print("   This affects mode capacity and crowding behavior!")
            
            # Scale parameters based on fitted kappa
            # kappa affects the fundamental capacity scaling
            fitted_N_max = default_N_max * (fitted_kappa / 2.0)
            fitted_K = int(default_K * (1.0 + (fitted_kappa - 2.0) * 0.2))
            fitted_threshold = default_threshold * (fitted_kappa / 2.0)
            
            print(f"   Adjusted K: {fitted_K} (default: {default_K})")
            print(f"   Adjusted N_max: {fitted_N_max:.4f} (default: {default_N_max})")
            print(f"   Adjusted threshold: {fitted_threshold:.4f} (default: {default_threshold})")
            
            fitted_N_min = default_N_min
            fitted_n_points = default_n_points
            
        else:
            print("🔧 No active fitment - using default parameters")
            fitted_K = default_K
            fitted_N_max = default_N_max
            fitted_N_min = default_N_min
            fitted_n_points = default_n_points
            fitted_threshold = default_threshold
            
    except ImportError:
        print("🔧 Fitment bridge not available - using default parameters")
        fitted_K = default_K
        fitted_N_max = default_N_max
        fitted_N_min = default_N_min
        fitted_n_points = default_n_points
        fitted_threshold = default_threshold
    
    # Run experiment
    results = run_experiment(
        K=fitted_K,
        N_max=fitted_N_max,
        N_min=fitted_N_min,
        n_points=fitted_n_points,
        utility_type='random',
        threshold=fitted_threshold,
        output_dir=str(output_dir)
    )
    
    # Print summary
    print(f"\nExperiment completed!")
    print(f"Parameters: K={results['parameters']['K']}, N_range=({results['parameters']['N_min']}, {results['parameters']['N_max']})")
    print(f"Utility type: {results['parameters']['utility_type']}")
    
    print(f"\nKey metrics:")
    print(f"  PR_min = {results['metrics']['PR_min']:.3f}")
    print(f"  Gini_max = {results['metrics']['Gini_max']:.3f}")
    print(f"  Entropy range: [{results['metrics']['entropy_min']:.3f}, {results['metrics']['entropy_max']:.3f}]")
    
    print(f"\nCritical points (threshold = {results['parameters']['threshold']}):")
    if results['metrics']['N_c_pr'] is not None:
        print(f"  N_c (PR): {results['metrics']['N_c_pr']:.4f}")
    else:
        print(f"  N_c (PR): No critical point found")
    
    if results['metrics']['N_c_gini'] is not None:
        print(f"  N_c (Gini): {results['metrics']['N_c_gini']:.4f}")
    else:
        print(f"  N_c (Gini): No critical point found")
    
    if results['metrics']['N_c_entropy'] is not None:
        print(f"  N_c (Entropy): {results['metrics']['N_c_entropy']:.4f}")
    else:
        print(f"  N_c (Entropy): No critical point found")
    
    # Show critical analysis summary
    critical_analysis = results['critical_analysis']
    has_critical = any([
        critical_analysis['has_pr_critical'],
        critical_analysis['has_gini_critical'], 
        critical_analysis['has_entropy_critical']
    ])
    
    if has_critical:
        print(f"\nCrowding transition detected at capacity N ≈ {min(filter(None, [results['metrics']['N_c_pr'], results['metrics']['N_c_gini'], results['metrics']['N_c_entropy']])):.3f}")
    else:
        print(f"\nNo clear crowding transition detected (increase threshold or adjust N range)")
    
    if 'files' in results:
        print(f"\nOutput files:")
        for name, path in results['files'].items():
            print(f"  {name}: {path}")
    
    return results


if __name__ == "__main__":
    main()
