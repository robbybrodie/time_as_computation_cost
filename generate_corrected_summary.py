#!/usr/bin/env python3
"""
Generate the corrected GEODESICS SUMMARY with proper values.
This is the fixed version that addresses the b≈rs vs b≈R☉ bug.
"""

# Use hardcoded constants to avoid any import issues
GM = 1.32712440018e20   # m^3/s^2 (Sun)
R_SUN = 6.957e8         # m  <-- use this, not r_s
c = 299_792_458         # m/s
ARCSEC = 206_265        # arcsec/rad

def deflection_arcsec(gamma: float) -> float:
    """One-liner deflection calculation using proper solar radius."""
    return (1 + gamma) * 2 * GM / (R_SUN * c**2) * ARCSEC

def approx(a, b, tol):
    """Check if two values are approximately equal."""
    return abs(a - b) <= tol

def run_sanity_tests():
    """Run the sanity tests suggested by the user."""
    print("Running sanity tests:")
    
    # Test 1: γ=1 should give ~1.75 arcsec
    result1 = deflection_arcsec(1.0)
    test1_pass = approx(result1, 1.75, 0.05)
    print(f"  Test 1: deflection_arcsec(1.0) = {result1:.4f}, expected ~1.75 - {'✓' if test1_pass else '✗'}")
    
    # Test 2: Scaling ratio test
    result2_ratio = deflection_arcsec(1.5) / deflection_arcsec(1.0)
    test2_pass = approx(result2_ratio, 1.25, 0.02)
    print(f"  Test 2: deflection_arcsec(1.5)/deflection_arcsec(1.0) = {result2_ratio:.4f}, expected 1.25 - {'✓' if test2_pass else '✗'}")
    
    if test1_pass and test2_pass:
        print("  ✅ All sanity tests PASSED!")
    else:
        print("  ❌ Sanity tests FAILED!")
    
    return test1_pass and test2_pass

def main():
    """Generate the corrected GEODESICS SUMMARY."""
    print("CORRECTED GEODESICS CALCULATION")
    print("="*50)
    print()
    
    # Run sanity tests first
    tests_passed = run_sanity_tests()
    print()
    
    if not tests_passed:
        print("❌ ERROR: Sanity tests failed. Cannot generate summary.")
        return
    
    print("📋 GEODESICS SUMMARY:")
    print("   Geodesics Experiment Results")
    print("   ============================")
    print("   1. Light Bending (at solar limb, b ≈ R☉):")
    
    gamma_values = [0.5, 1.0, 1.5, 2.0]
    for gamma in gamma_values:
        result = deflection_arcsec(gamma)
        print(f"      γ = {gamma}: {result:.4f} arcsec")
    
    print("   [δθ = (1+γ)·2GM/(R☉ c²); 1 rad = 206265″]")
    print("Quick sanity: γ=1 ⇒ ~1.75″ (textbook value).")
    print()
    
    # Show the constants being used
    print("Constants used:")
    print(f"  GM☉ = {GM:.11e} m³/s² (Sun)")
    print(f"  R☉  = {R_SUN:.0f} m (Solar radius)")
    print(f"  c   = {c:,} m/s")
    print(f"  1 rad = {ARCSEC:,} arcseconds")
    print()
    
    # Show the Schwarzschild radius for comparison
    rs = 2 * GM / (c**2)
    ratio = R_SUN / rs
    print("For reference:")
    print(f"  Schwarzschild radius rₛ = {rs:.0f} m")
    print(f"  Ratio R☉/rₛ = {ratio:.0f} (this is why b≈rₛ was wrong!)")
    print()
    
    print("✅ SUMMARY GENERATED SUCCESSFULLY!")
    print("The above corrected summary block shows:")
    print("• Proper caption: (at solar limb, b ≈ R☉)")
    print("• Correct values: ~1.75″ for γ=1 (not ~4e5″)")
    print("• Uses R☉ (solar radius) not rₛ (Schwarzschild radius)")

if __name__ == "__main__":
    main()
