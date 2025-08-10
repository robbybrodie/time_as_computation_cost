"""Tests for mode crowding experiment."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for testing
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc.crowding.mode_crowding import (
    run_demo, run_experiment, generate_utilities, softmax_occupancy,
    participation_ratio, gini_coefficient, shannon_entropy
)


class TestModeCrowding:
    """Tests for mode crowding functionality."""
    
    def test_utility_generation(self):
        """Test utility generation."""
        K = 10
        
        # Test random utilities
        utilities_random = generate_utilities(K, 'random')
        assert len(utilities_random) == K
        assert isinstance(utilities_random, np.ndarray)
        
        # Test spaced utilities
        utilities_spaced = generate_utilities(K, 'spaced')
        assert len(utilities_spaced) == K
        assert isinstance(utilities_spaced, np.ndarray)
        
        # Spaced utilities should be monotonic (sorted descending)
        assert np.all(np.diff(utilities_spaced) <= 0)
        
        # Both should be different (with high probability)
        assert not np.allclose(utilities_random, utilities_spaced)
    
    def test_softmax_occupancy(self):
        """Test softmax occupancy calculation."""
        utilities = np.array([2.0, 1.0, 0.0])
        K = len(utilities)
        
        # Test high capacity (N=1.0) - should be more uniform
        p_high = softmax_occupancy(utilities, 1.0)
        assert len(p_high) == K
        assert np.isclose(np.sum(p_high), 1.0)  # Should sum to 1
        assert np.all(p_high > 0)  # All positive
        
        # Test low capacity (N=0.01) - should crowd into highest utility
        p_low = softmax_occupancy(utilities, 0.01)
        assert len(p_low) == K
        assert np.isclose(np.sum(p_low), 1.0)  # Should sum to 1
        assert np.all(p_low > 0)  # All positive
        
        # Low capacity should be more concentrated on highest utility mode
        assert p_low[0] > p_high[0]  # First mode (highest utility) gets more weight
        assert p_low[2] < p_high[2]  # Last mode (lowest utility) gets less weight
    
    def test_participation_ratio(self):
        """Test participation ratio calculation."""
        # Equal probabilities - should give K
        K = 5
        p_equal = np.ones(K) / K
        pr_equal = participation_ratio(p_equal)
        assert np.isclose(pr_equal, K, rtol=1e-10)
        
        # One mode dominates - should give ~1
        p_dominated = np.array([0.99, 0.0025, 0.0025, 0.0025, 0.0025])
        pr_dominated = participation_ratio(p_dominated)
        assert pr_dominated < 1.1  # Should be close to 1
        assert pr_dominated > 1.0  # But greater than 1
    
    def test_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        # Equal probabilities - should give 0 (perfect equality)
        K = 4
        p_equal = np.ones(K) / K
        gini_equal = gini_coefficient(p_equal)
        assert gini_equal < 0.1  # Should be close to 0
        
        # One mode gets everything - should be high inequality
        p_unequal = np.array([1.0, 0.0, 0.0, 0.0])
        gini_unequal = gini_coefficient(p_unequal)
        assert gini_unequal > 0.5  # Should indicate high inequality
    
    def test_shannon_entropy(self):
        """Test Shannon entropy calculation."""
        # Equal probabilities - should give log(K)
        K = 4
        p_equal = np.ones(K) / K
        entropy_equal = shannon_entropy(p_equal)
        expected_entropy = np.log(K)
        assert np.isclose(entropy_equal, expected_entropy, rtol=1e-3)
        
        # One mode dominates - should give low entropy
        p_dominated = np.array([0.99, 0.003, 0.003, 0.004])
        entropy_dominated = shannon_entropy(p_dominated)
        assert entropy_dominated < entropy_equal
        assert entropy_dominated > 0  # Should still be positive
    
    def test_crowding_monotonicity(self):
        """Test that as N decreases, PR decreases and Gini increases."""
        K = 8
        utilities = generate_utilities(K, 'random')
        
        # Test a range of N values
        N_values = [1.0, 0.5, 0.1, 0.01]
        pr_values = []
        gini_values = []
        
        for N in N_values:
            p = softmax_occupancy(utilities, N)
            pr_values.append(participation_ratio(p))
            gini_values.append(gini_coefficient(p))
        
        # As N decreases, PR should generally decrease (within numerical tolerance)
        # Allow some tolerance for numerical issues
        for i in range(len(pr_values) - 1):
            assert pr_values[i] >= pr_values[i + 1] - 1e-10
        
        # As N decreases, Gini should generally increase (within numerical tolerance)
        for i in range(len(gini_values) - 1):
            assert gini_values[i] <= gini_values[i + 1] + 1e-10
    
    def test_run_demo_returns_figure(self):
        """Test that run_demo returns a Figure."""
        result = run_demo(K=5, n_points=20)
        
        # Should return a matplotlib Figure or dict
        assert isinstance(result, (plt.Figure, dict))
        
        # Clean up if it's a figure
        if isinstance(result, plt.Figure):
            plt.close(result)
    
    def test_run_experiment_basic(self):
        """Test basic experiment functionality."""
        results = run_experiment(K=8, n_points=30)
        
        # Check structure
        assert 'parameters' in results
        assert 'utilities' in results
        assert 'metrics' in results
        assert 'curves' in results
        assert 'critical_analysis' in results
        
        # Check metrics exist
        metrics = results['metrics']
        assert 'PR_min' in metrics
        assert 'Gini_max' in metrics
        assert 'entropy_min' in metrics
        assert 'entropy_max' in metrics
        
        # Check curves have right structure
        curves = results['curves']
        assert 'N_values' in curves
        assert 'participation_ratio' in curves
        assert 'gini_coefficient' in curves
        assert 'entropy' in curves
        
        # All curves should have same length
        n_points = len(curves['N_values'])
        assert len(curves['participation_ratio']) == n_points
        assert len(curves['gini_coefficient']) == n_points
        assert len(curves['entropy']) == n_points
    
    def test_deterministic_behavior(self):
        """Test that results are deterministic with same random seed."""
        # Both should use same seed (42) internally
        results1 = run_experiment(K=6, n_points=20)
        results2 = run_experiment(K=6, n_points=20)
        
        # Basic metrics should be identical
        assert np.isclose(results1['metrics']['PR_min'], results2['metrics']['PR_min'])
        assert np.isclose(results1['metrics']['Gini_max'], results2['metrics']['Gini_max'])
        
        # Curves should be identical
        assert np.allclose(results1['curves']['participation_ratio'],
                          results2['curves']['participation_ratio'])
        assert np.allclose(results1['curves']['gini_coefficient'],
                          results2['curves']['gini_coefficient'])


if __name__ == "__main__":
    # Run tests directly
    test = TestModeCrowding()
    test.test_utility_generation()
    test.test_softmax_occupancy()
    test.test_participation_ratio()
    test.test_gini_coefficient()
    test.test_shannon_entropy()
    test.test_crowding_monotonicity()
    test.test_run_demo_returns_figure()
    test.test_run_experiment_basic()
    test.test_deterministic_behavior()
    print("All mode crowding tests passed!")
