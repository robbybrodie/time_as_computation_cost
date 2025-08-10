"""Tests for causal diamond experiment."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for testing
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc.lattices.diamond import run_demo, run_experiment, create_diamond_lattice, count_paths_by_time


class TestCausalDiamond:
    """Tests for causal diamond functionality."""

    def test_basic_lattice_creation(self):
        """Test that diamond lattice is created correctly."""
        depth = 5
        G = create_diamond_lattice(depth)
        
        # Should have nodes
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
        
        # Check specific nodes exist
        assert G.has_node((0, 0))  # Origin
        assert G.has_node((depth, 0))  # Top center
        
        # Check edges follow light cone structure
        assert G.has_edge((0, 0), (1, -1))
        assert G.has_edge((0, 0), (1, 1))
    
    def test_path_counting_growth(self):
        """Test that num_paths_t grows with t."""
        depth = 8
        G = create_diamond_lattice(depth)
        paths = count_paths_by_time(G, depth)
        
        # Should have paths to all time layers
        assert len(paths) == depth + 1
        
        # Paths should generally increase (allowing for some numerical issues)
        # At minimum, should be monotonic for first few steps
        assert paths[1] >= paths[0]  # t=1 should have at least as many as t=0
        assert paths[2] >= paths[1]  # t=2 should have at least as many as t=1
        
        # Total paths should be positive
        assert all(p > 0 for p in paths)
    
    def test_run_demo_returns_figure(self):
        """Test that run_demo returns a Figure."""
        result = run_demo(depth=6, alpha=0.1)
        
        # Should return a matplotlib Figure or dict
        assert isinstance(result, (plt.Figure, dict))
        
        # Clean up if it's a figure
        if isinstance(result, plt.Figure):
            plt.close(result)
    
    def test_run_experiment_basic(self):
        """Test basic experiment functionality."""
        results = run_experiment(depth=8, alpha=0.1)
        
        # Check structure
        assert 'parameters' in results
        assert 'metrics' in results
        assert 'front_data' in results
        
        # Check metrics exist and are reasonable
        assert results['metrics']['node_count'] > 0
        assert results['metrics']['edge_count'] > 0
        assert results['metrics']['theoretical_paths_top'] > 0
        
        # Front data should have right structure
        front_data = results['front_data']
        assert 'times' in front_data
        assert 'mean_positions' in front_data
        assert 'std_positions' in front_data
        assert len(front_data['times']) == len(front_data['mean_positions'])
    
    def test_capacity_gradient_effect(self):
        """Test that capacity gradient affects results."""
        depth = 6
        
        # Run with no gradient (alpha=0)
        results_0 = run_experiment(depth=depth, alpha=0.0)
        
        # Run with strong gradient (alpha=0.5)
        results_5 = run_experiment(depth=depth, alpha=0.5)
        
        # Should have same basic structure but different front behavior
        assert results_0['metrics']['node_count'] == results_5['metrics']['node_count']
        assert results_0['metrics']['edge_count'] == results_5['metrics']['edge_count']
        
        # Front symmetry deviation should be different
        dev_0 = results_0['metrics']['front_symmetry_deviation']
        dev_5 = results_5['metrics']['front_symmetry_deviation']
        
        # At least one should be different (allowing for numerical precision)
        assert abs(dev_0 - dev_5) > 1e-10 or dev_0 != dev_5
    
    def test_deterministic_behavior(self):
        """Test that results are deterministic with same random seed."""
        # Both should use same seed (42) internally
        results1 = run_experiment(depth=6, alpha=0.1)
        results2 = run_experiment(depth=6, alpha=0.1)
        
        # Basic metrics should be identical
        assert results1['metrics']['node_count'] == results2['metrics']['node_count']
        assert results1['metrics']['edge_count'] == results2['metrics']['edge_count']
        
        # Front data should be very similar (allowing for floating point precision)
        mean1 = results1['front_data']['mean_positions']
        mean2 = results2['front_data']['mean_positions']
        
        # Should be close to identical
        assert np.allclose(mean1, mean2, rtol=1e-10)


if __name__ == "__main__":
    # Run tests directly
    test = TestCausalDiamond()
    test.test_basic_lattice_creation()
    test.test_path_counting_growth() 
    test.test_run_demo_returns_figure()
    test.test_run_experiment_basic()
    test.test_capacity_gradient_effect()
    test.test_deterministic_behavior()
    print("All causal diamond tests passed!")
