"""Test fitment API functionality."""

import pytest
from tacc.core.fitment import list_all, get
from tacc.core.microlaw import MicrolawInput
from tacc.core.nb import list_fitments


def test_fitment_registry():
    """Test that fitments are properly registered."""
    # Import to register fitments
    import tacc.fitments
    
    fitments = list_all()
    assert "no_fit" in fitments
    assert "single_param" in fitments
    
    # Test getting fitments
    no_fit = get("no_fit")
    assert no_fit.name == "no_fit"
    
    single_param = get("single_param")
    assert single_param.name == "single_param"


def test_no_fit_fitment():
    """Test no_fit fitment returns parameters unchanged."""
    import tacc.fitments
    
    no_fit = get("no_fit")
    
    # Create minimal dataset
    dataset = [
        {
            "x": {"phi": 0.1},
            "target": 1.05,
            "weight": 1.0
        },
        {
            "x": {"phi": 0.2}, 
            "target": 1.10,
            "weight": 1.0
        }
    ]
    
    ml_params = {"test_param": 1.5}
    bn_params = {"kappa": 2.0, "test_bn": 0.5}
    
    result = no_fit.fit(
        dataset,
        "ppn",
        ml_params,
        "exponential", 
        bn_params
    )
    
    # Should return parameters unchanged
    assert result["microlaw"] == ml_params
    assert result["bn"] == bn_params


def test_single_param_fitment():
    """Test single_param fitment updates the targeted parameter."""
    import tacc.fitments
    
    single_param = get("single_param")
    
    # Create synthetic dataset where kappa=3.0 should be optimal
    dataset = [
        {
            "x": {"phi": 0.0},  # N = 1.0
            "target": 1.0,      # B should be 1.0 when N=1 for any kappa
            "weight": 1.0
        },
        {
            "x": {"phi": 0.1},  # N = 1.1
            "target": 0.741,    # B ≈ exp(-3*(1-1.1)) = exp(0.3) ≈ 1.35, want lower target to drive kappa higher
            "weight": 1.0
        }
    ]
    
    ml_params = {}
    bn_params = {"kappa": 1.0}  # Start with kappa=1.0
    
    result = single_param.fit(
        dataset,
        "ppn",
        ml_params,
        "exponential",
        bn_params,
        extra={
            "target_param": "kappa",
            "init": 2.0,
            "max_iter": 10,
            "tol": 1e-3
        }
    )
    
    # Should have updated kappa
    assert "microlaw" in result
    assert "bn" in result
    assert result["microlaw"] == ml_params  # No microlaw params changed
    assert "kappa" in result["bn"]
    
    # Kappa should have changed from initial value
    final_kappa = result["bn"]["kappa"]
    assert final_kappa != 1.0  # Should have changed from initial


def test_notebook_integration():
    """Test integration with notebook bridge functions."""
    import tacc.fitments
    
    # Test list_fitments function
    fitments = list_fitments()
    assert "no_fit" in fitments
    assert "single_param" in fitments
    
    # Test get_fitment function
    from tacc.core.nb import get_fitment
    no_fit = get_fitment("no_fit")
    assert no_fit.name == "no_fit"


def test_fitment_error_handling():
    """Test error handling for invalid fitment names."""
    import tacc.fitments
    
    with pytest.raises(ValueError, match="Unknown fitment"):
        get("nonexistent_fitment")


if __name__ == "__main__":
    pytest.main([__file__])
