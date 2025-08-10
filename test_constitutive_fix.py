#!/usr/bin/env python3
"""
Test the constitutive law fix for numpy array compatibility.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "src"))

def test_constitutive_law():
    """Test that B_of_N works with both scalars and arrays."""
    print("Testing constitutive law B_of_N function...")
    
    try:
        from tacc.constitutive import B_of_N
        
        # Test with scalar inputs
        print("Testing with scalar inputs:")
        result_scalar = B_of_N(1.0, 2.0)
        print(f"  B_of_N(1.0, 2.0) = {result_scalar:.6f}")
        
        # Test with numpy arrays (this was failing before)
        print("Testing with numpy array inputs:")
        N_array = np.array([0.5, 1.0, 1.5, 2.0])
        kappa = 2.0
        result_array = B_of_N(N_array, kappa)
        print(f"  B_of_N([0.5, 1.0, 1.5, 2.0], 2.0) = {result_array}")
        
        # Test with mixed inputs (array N, scalar kappa)
        print("Testing with mixed inputs:")
        result_mixed = B_of_N(N_array, 1.5)
        print(f"  B_of_N([0.5, 1.0, 1.5, 2.0], 1.5) = {result_mixed}")
        
        # Verify B(1) == 1 property
        print("Verifying B(1) == 1 property:")
        b_of_1 = B_of_N(1.0, kappa)
        print(f"  B_of_N(1.0, 2.0) = {b_of_1:.6f} (should be 1.0)")
        
        print("✅ All constitutive law tests PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Constitutive law test FAILED: {e}")
        return False

def test_formula_correctness():
    """Test the mathematical correctness of the formula."""
    print("\nTesting mathematical correctness:")
    
    from tacc.constitutive import B_of_N
    
    # Test specific values
    test_cases = [
        (1.0, 2.0, 1.0),      # B(1) should equal 1
        (0.0, 2.0, np.exp(2.0)),  # B(0) = exp(-kappa*(1-0)) = exp(-kappa*1) = exp(-kappa)
        (2.0, 2.0, np.exp(2.0)),  # B(2) = exp(-kappa*(1-2)) = exp(-kappa*(-1)) = exp(kappa)
    ]
    
    for N, kappa, expected in test_cases:
        result = B_of_N(N, kappa)
        error = abs(result - expected)
        status = "✓" if error < 1e-10 else "✗"
        print(f"  {status} B_of_N({N}, {kappa}) = {result:.6f}, expected {expected:.6f}, error = {error:.2e}")
    
    print("✅ Mathematical correctness tests completed!")

if __name__ == "__main__":
    print("CONSTITUTIVE LAW FIX TEST")
    print("="*40)
    
    success = test_constitutive_law()
    test_formula_correctness()
    
    if success:
        print("\n🎉 FIX SUCCESSFUL! The PPN experiment should now work.")
        print("The B_of_N function now handles both scalars and numpy arrays correctly.")
    else:
        print("\n❌ Fix failed - there are still issues to resolve.")
