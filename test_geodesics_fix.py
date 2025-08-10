"""
Simple test to verify the geodesics fixes without requiring numpy/matplotlib.
"""

import sys
from pathlib import Path

# Bootstrap path setup
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "src"))

def test_geodesics_fix():
    """Test the geodesics fixes."""
    
    print("Testing Geodesics Fixes")
    print("=" * 30)
    
    try:
        from tacc import geodesics, ppn
        from tacc.constants import GM_SUN, R_SUN, c, ARCSEC_PER_RAD
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("This likely means other modules in the package have numpy dependencies.")
        print("Trying direct imports...")
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "tacc"))
            import geodesics
            import ppn
            import constants
            GM_SUN = constants.GM_SUN
            R_SUN = constants.R_SUN
            c = constants.c
            ARCSEC_PER_RAD = constants.ARCSEC_PER_RAD
            print("✓ Direct imports successful")
            # Update globals for the rest of the test
            globals()['geodesics'] = geodesics
            globals()['ppn'] = ppn
        except ImportError as e2:
            print(f"✗ Direct import also failed: {e2}")
            return False
    
    # Test unit tests
    print("\n1. Running unit tests...")
    try:
        result = geodesics.run_all_tests()
        if result:
            print("✓ All unit tests passed!")
        else:
            print("✗ Unit tests failed")
            return False
    except Exception as e:
        print(f"✗ Unit test error: {e}")
        return False
    
    # Test solar limb deflection
    print("\n2. Testing solar limb deflection (b = R☉)...")
    try:
        result = geodesics.solar_limb_deflection(gamma=1.0)
        deflection_arcsec = result['deflection_arcsec']
        print(f"   γ=1.0: {deflection_arcsec:.4f} arcsec")
        
        if 1.70 <= deflection_arcsec <= 1.80:
            print("✓ Solar limb deflection is in expected range (1.70-1.80 arcsec)")
        else:
            print(f"✗ Solar limb deflection {deflection_arcsec:.4f} is outside expected range")
            return False
            
    except Exception as e:
        print(f"✗ Solar limb test error: {e}")
        return False
    
    # Test Mercury precession
    print("\n3. Testing Mercury precession...")
    try:
        precession = geodesics.mercury_precession_arcsec_per_century(gamma=1.0)
        print(f"   γ=1.0: {precession:.2f} arcsec/century")
        
        if 42.0 <= precession <= 44.0:
            print("✓ Mercury precession is in expected range (42-44 arcsec/century)")
        else:
            print(f"✗ Mercury precession {precession:.2f} is outside expected range")
            return False
            
    except Exception as e:
        print(f"✗ Mercury precession test error: {e}")
        return False
    
    # Test PPN parameter relationships
    print("\n4. Testing PPN parameter relationships...")
    try:
        gr_limit = ppn.get_gr_limit()
        print(f"   GR limit: κ={gr_limit['kappa']}, γ={gr_limit['gamma']}, β={gr_limit['beta']}")
        
        gamma, beta = ppn.extract_ppn_params(kappa=2.0)
        if gamma == 1.0 and beta == 1.0:
            print("✓ κ=2 correctly gives γ=1, β=1")
        else:
            print(f"✗ κ=2 gives γ={gamma}, β={beta}, expected γ=1, β=1")
            return False
            
    except Exception as e:
        print(f"✗ PPN test error: {e}")
        return False
    
    # Test solar system compliance
    print("\n5. Testing solar system compliance...")
    try:
        compliance = ppn.validate_solar_system_compliance(kappa=2.0)
        print(f"   κ=2.0 compliance: {compliance['overall_compliant']}")
        
        if compliance['overall_compliant']:
            print("✓ κ=2.0 is compliant with Solar System constraints")
        else:
            print("✗ κ=2.0 is not compliant with Solar System constraints")
            return False
            
    except Exception as e:
        print(f"✗ Solar system compliance test error: {e}")
        return False
    
    print("\n" + "=" * 30)
    print("✓ ALL TESTS PASSED!")
    print("\nKey Results:")
    print(f"- Solar limb deflection (γ=1): {result['deflection_arcsec']:.4f} arcsec")
    print(f"- Mercury precession (γ=1): {precession:.2f} arcsec/century")
    print("- PPN relationship: κ=2 ⇒ γ=1, β=1 (GR limit)")
    print("- Solar System constraints satisfied")
    
    print(f"\nIMPORTANT: The bug has been fixed!")
    print(f"Previous deflection values were ~5 orders of magnitude too large")
    print(f"due to using b≈rs instead of b≈R☉ for solar limb calculations.")
    
    return True

if __name__ == "__main__":
    success = test_geodesics_fix()
    if not success:
        sys.exit(1)
