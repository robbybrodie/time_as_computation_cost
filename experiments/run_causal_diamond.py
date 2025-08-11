"""
Causal Diamond Experiment Runner
"""

import sys
from pathlib import Path

# Bootstrap path setup for Colab compatibility
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc.lattices.diamond import run_demo, run_experiment


def main():
    """Run the causal diamond experiment."""
    print("Running Causal Diamond Experiment...")
    print("=" * 50)
    
    # Set up output directory
    output_dir = Path(__file__).resolve().parent / "out" / "causal_diamond"
    
    # Check for fitment state from interactive widgets
    default_alpha = 0.1
    default_depth = 10
    
    try:
        from tacc.core.experiment_bridge import get_fitted_kappa, is_fitment_active, get_active_fitment_info
        
        # Check for individual experiment toggle (from notebook)
        use_fitment = True
        try:
            if 'experiment_toggles' in globals() and globals()['experiment_toggles'] is not None:
                toggle = globals()['experiment_toggles']['causal_diamond']
                use_fitment = toggle.value
                print(f"🎛️ Individual toggle: {'ON' if use_fitment else 'OFF'} for Causal Diamond")
        except:
            pass  # Fall back to global fitment setting
        
        if is_fitment_active() and use_fitment:
            fitment_info = get_active_fitment_info()
            fitted_kappa = get_fitted_kappa()
            
            print(f"🎯 USING FITMENT: {fitment_info['name']}")
            print(f"   Fitted κ: {fitted_kappa:.4f}")
            print("   This affects lattice connectivity and propagation!")
            
            # Scale parameters based on fitted kappa with MORE DRAMATIC effects
            # kappa affects the fundamental lattice structure and propagation
            fitted_alpha = default_alpha * (fitted_kappa / 2.0) ** 2  # Quadratic scaling for more sensitivity
            
            # More significant depth scaling based on kappa
            depth_multiplier = 1.0 + (fitted_kappa - 2.0) * 0.5  # 50% change per kappa unit
            fitted_depth = max(5, int(default_depth * depth_multiplier))  # Minimum depth of 5
            
            print(f"   Adjusted α: {fitted_alpha:.4f} (default: {default_alpha})")
            print(f"   α scaling: quadratic with κ for stronger coupling effects")
            print(f"   Adjusted depth: {fitted_depth} (default: {default_depth})")
            print(f"   Depth scaling: 50% change per κ unit for visible lattice differences")
            
        else:
            print("🔧 No active fitment - using default parameters")
            fitted_alpha = default_alpha
            fitted_depth = default_depth
            
    except ImportError:
        print("🔧 Fitment bridge not available - using default parameters")
        fitted_alpha = default_alpha
        fitted_depth = default_depth
    
    # Run experiment
    results = run_experiment(
        depth=fitted_depth,
        alpha=fitted_alpha,
        output_dir=str(output_dir)
    )
    
    # Print summary with fitment impact analysis
    print(f"\nExperiment completed!")
    print(f"Parameters: depth={results['parameters']['depth']}, alpha={results['parameters']['alpha']}")
    print(f"Metrics:")
    print(f"  Node count: {results['metrics']['node_count']}")
    print(f"  Edge count: {results['metrics']['edge_count']}")
    print(f"  Theoretical paths to top: {results['metrics']['theoretical_paths_top']}")
    print(f"  Actual paths to top: {results['metrics']['actual_paths_top']}")
    print(f"  Front symmetry deviation: {results['metrics']['front_symmetry_deviation']:.4f}")
    
    # Show fitment impact comparison
    try:
        from tacc.core.experiment_bridge import is_fitment_active, get_fitted_kappa
        if is_fitment_active():
            fitted_kappa = get_fitted_kappa()
            print(f"\n🎯 FITMENT IMPACT ANALYSIS:")
            print(f"   Fitted κ: {fitted_kappa:.4f}")
            
            # Compare with default results
            default_alpha_result = default_alpha
            default_depth_result = default_depth
            
            alpha_change_pct = ((fitted_alpha - default_alpha_result) / default_alpha_result) * 100
            depth_change_pct = ((fitted_depth - default_depth_result) / default_depth_result) * 100
            
            print(f"   α change: {alpha_change_pct:+.1f}% (quadratic κ scaling)")
            print(f"   Depth change: {depth_change_pct:+.1f}% (0.5x κ deviation scaling)")
            print(f"   Expected effects: Different lattice size and coupling strength")
            print(f"   → Node count scales ~depth³, edge connectivity scales with α")
            
    except ImportError:
        pass
    
    if 'files' in results:
        print(f"\nOutput files:")
        for name, path in results['files'].items():
            print(f"  {name}: {path}")
    
    return results


if __name__ == "__main__":
    main()
