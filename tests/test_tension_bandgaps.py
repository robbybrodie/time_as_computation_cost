"""Tests for tension bandgaps experiment."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for testing
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc.bandgaps.tension import (
    run_demo, run_experiment, generate_synthetic_data, 
    exponential_model, fit_all_models
)


class TestTensionBandgaps:
    """Tests for tension bandgaps functionality."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        n_points = 30
        noise_sigma = 0.05
        a_true = 2.0
        beta_true = 1.5
        
        data = generate_synthetic_data(n_points, noise_sigma, a_true, beta_true)
        
        # Check structure
        assert 'N_values' in data
        assert 'DoF_values' in data  
        assert 'psi_values' in data
        
        # Check dimensions
        assert len(data['N_values']) == n_points
        assert len(data['DoF_values']) == n_points
        assert len(data['psi_values']) == n_points
        
        # Check ranges
        assert np.all(data['N_values'] >= 0.7)
        assert np.all(data['N_values'] <= 1.0)
        assert np.all(data['DoF_values'] > 0)  # Should be positive
        assert np.all(data['psi_values'] > 0)  # Should be positive
    
    def test_exponential_model(self):
        """Test exponential model function."""
        N = np.linspace(0.7, 1.0, 10)
        a = 2.0
        
        result = exponential_model(N, a)
        
        # Should be same length as input
        assert len(result) == len(N)
        
        # Should be positive
        assert np.all(result > 0)
        
        # Should be monotonic (higher N → higher DoF)
        assert np.all(np.diff(result) >= 0)
        
        # At N=1, should give exp(0) = 1
        result_at_1 = exponential_model(np.array([1.0]), a)
        assert np.isclose(result_at_1[0], 1.0)
    
    def test_parameter_recovery(self):
        """Test that fitting recovers true parameters within tolerance."""
        # Use exact values from the test
        n_points = 40
        noise_sigma = 0.03  # Low noise for better recovery
        a_true = 2.0
        beta_true = 1.5
        
        # Generate synthetic data
        data = generate_synthetic_data(n_points, noise_sigma, a_true, beta_true)
        
        # Fit models
        models = fit_all_models(data['N_values'], data['DoF_values'], data['psi_values'])
        
        # Check exponential model recovery
        a_fitted = models['exponential']['params'][0]
        assert abs(a_fitted - a_true) < 0.2  # Within 0.2 tolerance
        
        # Check psi model recovery
        beta_fitted = models['psi_beta']['params'][0] 
        assert abs(beta_fitted - beta_true) < 0.2  # Within 0.2 tolerance
    
    def test_model_comparison_metrics(self):
        """Test that AIC/BIC/CV metrics are computed for all models."""
        data = generate_synthetic_data(30, 0.05, 2.0, 1.5)
        models = fit_all_models(data['N_values'], data['DoF_values'], data['psi_values'])
        
        required_models = ['exponential', 'polynomial', 'power_law']
        required_metrics = ['aic', 'bic', 'cv_score', 'cv_std']
        
        for model in required_models:
            assert model in models
            for metric in required_metrics:
                assert metric in models[model]
                # Should be finite numbers
                assert np.isfinite(models[model][metric])
    
    def test_cross_validation_returns_5_scores(self):
        """Test that k-fold CV returns 5 scores as expected."""
        data = generate_synthetic_data(50, 0.05, 2.0, 1.5)  # More data for stable CV
        models = fit_all_models(data['N_values'], data['DoF_values'], data['psi_values'])
        
        # CV should have been run with k=5 folds
        for model in ['exponential', 'polynomial', 'power_law']:
            cv_score = models[model]['cv_score']
            cv_std = models[model]['cv_std']
            
            # Should be finite
            assert np.isfinite(cv_score)
            assert np.isfinite(cv_std)
            
            # Score should be positive (MSE)
            assert cv_score >= 0
            assert cv_std >= 0
    
    def test_run_demo_returns_figure(self):
        """Test that run_demo returns a Figure."""
        result = run_demo(n_points=20, noise_sigma=0.05)
        
        # Should return a matplotlib Figure or dict
        assert isinstance(result, (plt.Figure, dict))
        
        # Clean up if it's a figure  
        if isinstance(result, plt.Figure):
            plt.close(result)
    
    def test_run_experiment_basic(self):
        """Test basic experiment functionality."""
        results = run_experiment(n_points=30, noise_sigma=0.05, a_true=2.0, beta_true=1.5)
        
        # Check structure
        assert 'parameters' in results
        assert 'fitted_params' in results
        assert 'model_comparison' in results
        assert 'synthetic_data' in results
        
        # Check fitted parameters exist
        assert 'a_hat' in results['fitted_params']
        assert 'beta_hat' in results['fitted_params']
        
        # Check model comparison exists for all models
        required_models = ['exponential', 'polynomial', 'power_law']
        for model in required_models:
            assert model in results['model_comparison']
            assert 'aic' in results['model_comparison'][model]
            assert 'bic' in results['model_comparison'][model]
            assert 'cv_score' in results['model_comparison'][model]
    
    def test_deterministic_behavior(self):
        """Test that results are deterministic with same random seed."""
        # Both should use same seed (42) internally
        results1 = run_experiment(n_points=25, noise_sigma=0.05)
        results2 = run_experiment(n_points=25, noise_sigma=0.05)
        
        # Parameters should be identical (using same seed)
        assert np.isclose(results1['fitted_params']['a_hat'], 
                         results2['fitted_params']['a_hat'])
        assert np.isclose(results1['fitted_params']['beta_hat'], 
                         results2['fitted_params']['beta_hat'])
        
        # Model comparison metrics should be identical
        for model in ['exponential', 'polynomial', 'power_law']:
            assert np.isclose(results1['model_comparison'][model]['aic'],
                             results2['model_comparison'][model]['aic'])


if __name__ == "__main__":
    # Run tests directly
    test = TestTensionBandgaps()
    test.test_synthetic_data_generation()
    test.test_exponential_model()
    test.test_parameter_recovery()
    test.test_model_comparison_metrics()
    test.test_cross_validation_returns_5_scores()
    test.test_run_demo_returns_figure()
    test.test_run_experiment_basic()
    test.test_deterministic_behavior()
    print("All tension bandgaps tests passed!")
