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
    
    # Run experiment
    results = run_experiment(
        depth=10,
        alpha=0.1,
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
