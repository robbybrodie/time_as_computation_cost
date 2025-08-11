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
        
        if is_fitment_active():
            fitment_info = get_active_fitment_info()
            fitted_kappa = get_fitted_kappa()
            
            print(f"🎯 USING FITMENT: {fitment_info['name']}")
            print(f"   Fitted κ: {fitted_kappa:.4f}")
            print("   This affects lattice connectivity and propagation!")
            
            # Scale alpha based on fitted kappa (kappa affects the coupling strength)
            # Default kappa=2.0 gives alpha=0.1, so scale proportionally
            fitted_alpha = default_alpha * (fitted_kappa / 2.0)
            fitted_depth = int(default_depth * (1.0 + (fitted_kappa - 2.0) * 0.1))  # Slight depth adjustment
            
            print(f"   Adjusted α: {fitted_alpha:.4f} (default: {default_alpha})")
            print(f"   Adjusted depth: {fitted_depth} (default: {default_depth})")
            
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
    
    # Print summary
    print(f"\nExperiment completed!")
    print(f"Parameters: depth={results['parameters']['depth']}, alpha={results['parameters']['alpha']}")
    print(f"Metrics:")
    print(f"  Node count: {results['metrics']['node_count']}")
    print(f"  Edge count: {results['metrics']['edge_count']}")
    print(f"  Theoretical paths to top: {results['metrics']['theoretical_paths_top']}")
    print(f"  Actual paths to top: {results['metrics']['actual_paths_top']}")
    print(f"  Front symmetry deviation: {results['metrics']['front_symmetry_deviation']:.4f}")
    
    if 'files' in results:
        print(f"\nOutput files:")
        for name, path in results['files'].items():
            print(f"  {name}: {path}")
    
    return results


if __name__ == "__main__":
    main()
