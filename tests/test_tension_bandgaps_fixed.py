"""
Tests for the fixed Tension Bandgaps experiment.
Verifies data leakage prevention, AICc/BIC computation, and proper ML practices.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from tacc.experiments.tension_bandgaps import (
    TBGenParams, generate_tb_dataset, run_tension_bandgaps,
    fit_exponential, fit_polynomial, fit_powerlaw,
    predict, aic_bic_aicc, leakage_guard, apply_fitment
)


class TestTBGenParams:
    """Test the frozen generator parameters."""
    
    def test_frozen_dataclass(self):
        """Test that TBGenParams is frozen."""
        p = TBGenParams()
        
        # Should not be able to modify
        with pytest.raises(Exception):  # FrozenInstanceError
            p.seed = 999
    
    def test_default_values(self):
        """Test default parameter values."""
        p = TBGenParams()
        assert p.n_points == 50
        assert p.noise_sigma == 0.05
        assert p.a_true == 2.0
        assert p.beta_true == 1.5
        assert p.seed == 42


class TestDataGeneration:
    """Test frozen data generation."""
    
    def test_generate_dataset_deterministic(self):
        """Test that data generation is deterministic."""
        p = TBGenParams(seed=123)
        
        data1 = generate_tb_dataset(p)
        data2 = generate_tb_dataset(p)
        
        # Should be identical
        assert np.array_equal(data1["x"], data2["x"])
        assert np.array_equal(data1["y"], data2["y"])
        assert np.array_equal(data1["y_clean"], data2["y_clean"])
    
    def test_different_seeds_different_data(self):
        """Test that different seeds produce different data."""
        p1 = TBGenParams(seed=123)
        p2 = TBGenParams(seed=456)
        
        data1 = generate_tb_dataset(p1)
        data2 = generate_tb_dataset(p2)
        
        # x should be the same (deterministic linspace)
        assert np.array_equal(data1["x"], data2["x"])
        # y should be different (different noise)
        assert not np.array_equal(data1["y"], data2["y"])
    
    def test_parameter_preservation(self):
        """Test that generator parameters are preserved in output."""
        p = TBGenParams(n_points=25, noise_sigma=0.1, a_true=3.0, beta_true=2.0)
        data = generate_tb_dataset(p)
        
        gen_params = data["gen_params"]
        assert gen_params["n_points"] == 25
        assert gen_params["noise_sigma"] == 0.1
        assert gen_params["a_true"] == 3.0
        assert gen_params["beta_true"] == 2.0


class TestModelFitting:
    """Test individual model fitting functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        x = np.linspace(0.1, 1.0, 20)
        y = 2.0 * np.exp(1.5 * x) + np.random.normal(0, 0.1, len(x))
        return x, y
    
    def test_fit_exponential(self, sample_data):
        """Test exponential fitting."""
        x, y = sample_data
        params = fit_exponential(x, y)
        
        assert params["family"] == "exponential"
        assert "a" in params
        assert "b" in params
        assert not params.get("invalid", False)
    
    def test_fit_polynomial(self, sample_data):
        """Test polynomial fitting."""
        x, y = sample_data
        params = fit_polynomial(x, y, degree=2)
        
        assert params["family"] == "polynomial"
        assert "coefs" in params
        assert params["degree"] == 2
        assert len(params["coefs"]) == 3  # degree + 1
    
    def test_fit_powerlaw(self, sample_data):
        """Test power law fitting."""
        x, y = sample_data
        # Ensure positive values for power law
        y = np.abs(y) + 0.1
        params = fit_powerlaw(x, y)
        
        assert params["family"] == "power_law"
        if not params.get("invalid", False):
            assert "A" in params
            assert "B" in params
    
    def test_predict_exponential(self, sample_data):
        """Test exponential prediction."""
        x, y = sample_data
        params = fit_exponential(x, y)
        y_pred = predict(params, x)
        
        assert len(y_pred) == len(x)
        assert np.all(np.isfinite(y_pred))
    
    def test_predict_invalid_model(self, sample_data):
        """Test prediction with invalid model."""
        x, _ = sample_data
        params = {"family": "exponential", "invalid": True}
        y_pred = predict(params, x)
        
        assert len(y_pred) == len(x)
        assert np.all(np.isnan(y_pred))


class TestInformationCriteria:
    """Test AIC/BIC/AICc computation."""
    
    def test_aic_bic_aicc_basic(self):
        """Test basic AIC/BIC/AICc computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        k = 2  # 2 parameters
        
        aic, bic, aicc = aic_bic_aicc(y_true, y_pred, k)
        
        assert np.isfinite(aic)
        assert np.isfinite(bic)
        assert np.isfinite(aicc)
        assert bic > aic  # BIC penalizes complexity more
        assert aicc > aic  # AICc includes small-sample correction
    
    def test_aic_bic_aicc_nan_handling(self):
        """Test handling of NaN predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, np.nan, 3.1])
        k = 2
        
        aic, bic, aicc = aic_bic_aicc(y_true, y_pred, k)
        
        # Should handle NaN gracefully
        assert np.isfinite(aic) or aic == float("inf")
        assert np.isfinite(bic) or bic == float("inf")
        assert np.isfinite(aicc) or aicc == float("inf")
    
    def test_aic_bic_aicc_insufficient_data(self):
        """Test handling when n <= k."""
        y_true = np.array([1.0])
        y_pred = np.array([1.1])
        k = 2  # More parameters than data points
        
        aic, bic, aicc = aic_bic_aicc(y_true, y_pred, k)
        
        # Should return infinity
        assert aic == float("inf")
        assert bic == float("inf")
        assert aicc == float("inf")


class TestLeakageGuard:
    """Test data leakage detection."""
    
    def test_leakage_guard_no_change(self):
        """Test leakage guard with no parameter change."""
        gen_params_before = {"a_true": 2.0, "noise_sigma": 0.05}
        gen_params_after = {"a_true": 2.0, "noise_sigma": 0.05}
        y_train = np.array([1.0, 2.0, 3.0])
        yhat_train = np.array([1.1, 1.9, 3.1])
        noise_sigma = 0.05
        
        # Should not raise
        leakage_guard(gen_params_before, gen_params_after, y_train, yhat_train, noise_sigma)
    
    def test_leakage_guard_parameter_change(self):
        """Test leakage guard detects parameter change."""
        gen_params_before = {"a_true": 2.0, "noise_sigma": 0.05}
        gen_params_after = {"a_true": 2.5, "noise_sigma": 0.05}  # Changed!
        y_train = np.array([1.0, 2.0, 3.0])
        yhat_train = np.array([1.1, 1.9, 3.1])
        noise_sigma = 0.05
        
        with pytest.raises(RuntimeError, match="LEAKAGE.*generator parameters changed"):
            leakage_guard(gen_params_before, gen_params_after, y_train, yhat_train, noise_sigma)
    
    def test_leakage_guard_perfect_fit(self):
        """Test leakage guard detects suspiciously perfect fit."""
        gen_params_before = {"a_true": 2.0, "noise_sigma": 0.05}
        gen_params_after = {"a_true": 2.0, "noise_sigma": 0.05}
        y_train = np.array([1.0, 2.0, 3.0])
        yhat_train = np.array([1.0, 2.0, 3.0])  # Perfect fit
        noise_sigma = 0.05
        
        with pytest.raises(RuntimeError, match="LEAKAGE.*near-zero train error"):
            leakage_guard(gen_params_before, gen_params_after, y_train, yhat_train, noise_sigma)
    
    def test_leakage_guard_with_nan(self):
        """Test leakage guard handles NaN predictions."""
        gen_params_before = {"a_true": 2.0, "noise_sigma": 0.05}
        gen_params_after = {"a_true": 2.0, "noise_sigma": 0.05}
        y_train = np.array([1.0, 2.0, 3.0])
        yhat_train = np.array([1.1, np.nan, 3.1])
        noise_sigma = 0.05
        
        # Should not raise (NaN handling)
        leakage_guard(gen_params_before, gen_params_after, y_train, yhat_train, noise_sigma)


class TestFitment:
    """Test fitment application."""
    
    def test_apply_fitment_no_fit(self):
        """Test no_fit fitment."""
        dataset = {}
        family = "polynomial"
        opts = apply_fitment(dataset, family, "no_fit", {})
        assert opts == {}
    
    def test_apply_fitment_single_param(self):
        """Test single_param fitment."""
        dataset = {}
        family = "polynomial"
        fitment_params = {"degree": 4}
        opts = apply_fitment(dataset, family, "single_param", fitment_params)
        assert opts["degree"] == 4
    
    def test_apply_fitment_unknown(self):
        """Test unknown fitment."""
        dataset = {}
        family = "polynomial"
        opts = apply_fitment(dataset, family, "unknown_fitment", {})
        assert opts == {}


class TestFullExperiment:
    """Test complete experiment pipeline."""
    
    def test_run_experiment_basic(self):
        """Test basic experiment run."""
        gen_params = TBGenParams(n_points=20, seed=123)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            stamp = run_tension_bandgaps(
                gen_p=gen_params,
                families=("exponential", "polynomial"),
                fitment=("no_fit", {}),
                result_dir=temp_dir
            )
            
            # Check stamp structure
            assert "timestamp" in stamp
            assert "commit" in stamp
            assert "generator" in stamp
            assert "fitment" in stamp
            assert "families" in stamp
            assert "best_by_AICc" in stamp
            assert "train_size" in stamp
            assert "val_size" in stamp
            assert "test_size" in stamp
            
            # Check generator params preserved
            assert stamp["generator"]["n_points"] == 20
            assert stamp["generator"]["seed"] == 123
            
            # Check results file created
            results_file = Path(temp_dir) / "tension_bandgaps_seed123.json"
            assert results_file.exists()
            
            # Verify file content
            with open(results_file) as f:
                saved_stamp = json.load(f)
            assert saved_stamp["generator"] == stamp["generator"]
    
    def test_run_experiment_with_fitment(self):
        """Test experiment with fitment."""
        gen_params = TBGenParams(n_points=15, seed=456)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            stamp = run_tension_bandgaps(
                gen_p=gen_params,
                families=("polynomial",),
                fitment=("single_param", {"degree": 3}),
                result_dir=temp_dir
            )
            
            # Check fitment recorded
            assert stamp["fitment"]["name"] == "single_param"
            assert stamp["fitment"]["params"]["degree"] == 3
            
            # Generator should be unchanged
            assert stamp["generator"]["n_points"] == 15
            assert stamp["generator"]["seed"] == 456
    
    def test_experiment_train_val_test_split(self):
        """Test that train/val/test split is working."""
        gen_params = TBGenParams(n_points=30, seed=789)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            stamp = run_tension_bandgaps(
                gen_p=gen_params,
                families=("exponential",),
                fitment=("no_fit", {}),
                result_dir=temp_dir
            )
            
            # Check split sizes
            total_size = stamp["train_size"] + stamp["val_size"] + stamp["test_size"]
            assert total_size == 30
            assert stamp["train_size"] > 0
            assert stamp["val_size"] > 0
            assert stamp["test_size"] > 0
    
    def test_experiment_model_comparison(self):
        """Test model comparison and ranking."""
        gen_params = TBGenParams(n_points=40, seed=101112)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            stamp = run_tension_bandgaps(
                gen_p=gen_params,
                families=("exponential", "polynomial", "power_law"),
                fitment=("no_fit", {}),
                result_dir=temp_dir
            )
            
            # Should have results for all families
            valid_families = [f for f in stamp["families"] if not f.get("invalid")]
            assert len(valid_families) >= 1  # At least one should work
            
            # Each valid family should have metrics
            for family in valid_families:
                assert "val" in family
                assert "AICc" in family["val"]
                assert "AIC" in family["val"]
                assert "BIC" in family["val"]
                assert "test" in family
                assert "mse" in family["test"]
            
            # Best model should be identified
            if valid_families:
                assert stamp["best_by_AICc"] is not None


class TestCLIIntegration:
    """Test CLI integration."""
    
    def test_cli_integration_exists(self):
        """Test that CLI can import the experiment."""
        from tacc.cli.main import main
        # Just test import works - full CLI testing would require subprocess
        assert main is not None


if __name__ == "__main__":
    pytest.main([__file__])
