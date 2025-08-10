#!/usr/bin/env python3
"""
Find which geodesics implementation is producing the wrong values.
"""

import sys
from pathlib import Path

def test_hardcoded_values():
    """Test with hardcoded constants to verify correct calculations."""
    print("=== HARDCODED TEST (SHOULD BE CORRECT) ===")
    
    # Your corrected constants
    GM = 1.32712440018e20      # m^3/s^2
    R_sun = 6.957e8            # m
    c = 299_792_458            # m/s
    ARCSEC = 206_265           # arcsec/rad
    
    gamma_values = [0.5, 1.0, 1.5, 2.0]
    expected_values = [1.3134, 1.7512, 2.1890, 2.6268]
    
    for gamma, expected in zip(gamma_values, expected_values):
        calculated = (1 + gamma) * 2 * GM / (R_sun * c**2) * ARCSEC
        print(f"γ = {gamma}: {calculated:.4f} arcsec (expected: {expected:.4f}) - {'✓' if abs(calculated - expected) < 0.001 else '✗'}")
    print()

def test_tacc_implementation():
    """Test the main tacc implementation."""
    print("=== TACC IMPLEMENTATION TEST ===")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from tacc.geodesics import solar_limb_deflection
        
        gamma_values = [0.5, 1.0, 1.5, 2.0]
        for gamma in gamma_values:
            result = solar_limb_deflection(gamma)
            print(f"γ = {gamma}: {result['deflection_arcsec']:.4f} arcsec")
    except Exception as e:
        print(f"ERROR: {e}")
    print()

def test_capacity_implementation():
    """Test the capacity-physics implementation."""
    print("=== CAPACITY-PHYSICS IMPLEMENTATION TEST ===")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "capacity-physics" / "src"))
        from capacity.geodesics import solar_limb_deflection
        
        gamma_values = [0.5, 1.0, 1.5, 2.0]
        for gamma in gamma_values:
            result = solar_limb_deflection(gamma)
            print(f"γ = {gamma}: {result['deflection_arcsec']:.4f} arcsec")
    except Exception as e:
        print(f"ERROR: {e}")
    print()

def test_deprecated_function():
    """Test the deprecated null_geodesic_deflection function."""
    print("=== DEPRECATED FUNCTION TEST (SHOULD BE WRONG) ===")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from tacc.geodesics import null_geodesic_deflection
        from tacc.constants import GM_SUN, c, ARCSEC_PER_RAD
        
        gamma_values = [0.5, 1.0, 1.5, 2.0]
        
        # This is the WRONG way (using b/rs = 1.0)
        print("Using b/rs = 1.0 (WRONG - this is what was causing huge values):")
        for gamma in gamma_values:
            deflection_rad = null_geodesic_deflection(1.0, gamma, GM_SUN, c)
            deflection_arcsec = deflection_rad * ARCSEC_PER_RAD
            print(f"γ = {gamma}: {deflection_arcsec:.0f} arcsec (HUGE - WRONG!)")
        
        # This would be the correct way (using proper b/rs ratio)
        print("\nUsing proper b/rs ratio (R_sun/schwarzschild_radius):")
        rs = 2 * GM_SUN / (c**2)
        R_sun = 6.957e8
        proper_ratio = R_sun / rs  # This should be ~235,000
        print(f"Proper ratio R_sun/rs = {proper_ratio:.0f}")
        
        for gamma in gamma_values:
            deflection_rad = null_geodesic_deflection(proper_ratio, gamma, GM_SUN, c)
            deflection_arcsec = deflection_rad * ARCSEC_PER_RAD
            expected = [1.3134, 1.7512, 2.1890, 2.6268][gamma_values.index(gamma)]
            print(f"γ = {gamma}: {deflection_arcsec:.4f} arcsec (expected: {expected:.4f}) - {'✓' if abs(deflection_arcsec - expected) < 0.01 else '✗'}")
            
    except Exception as e:
        print(f"ERROR: {e}")
    print()

def check_constant_files():
    """Check what constants are being used."""
    print("=== CONSTANTS CHECK ===")
    
    # Check tacc constants
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from tacc.constants import GM_SUN, R_SUN, c, ARCSEC_PER_RAD
        print("TACC constants:")
        print(f"  GM_SUN = {GM_SUN}")
        print(f"  R_SUN = {R_SUN}")  
        print(f"  c = {c}")
        print(f"  ARCSEC_PER_RAD = {ARCSEC_PER_RAD}")
        
        # Calculate Schwarzschild radius ratio
        rs = 2 * GM_SUN / (c**2)
        ratio = R_SUN / rs
        print(f"  R_SUN/rs ratio = {ratio:.0f}")
        
    except Exception as e:
        print(f"TACC constants ERROR: {e}")
    
    # Check capacity constants
    try:
        sys.path.insert(0, str(Path(__file__).parent / "capacity-physics" / "src"))
        from capacity.constants import GM_SUN, R_SUN, c, ARCSEC_PER_RAD
        print("\nCAPACITY constants:")
        print(f"  GM_SUN = {GM_SUN}")
        print(f"  R_SUN = {R_SUN}")  
        print(f"  c = {c}")
        print(f"  ARCSEC_PER_RAD = {ARCSEC_PER_RAD}")
        
    except Exception as e:
        print(f"CAPACITY constants ERROR: {e}")
    print()

if __name__ == "__main__":
    print("COMPREHENSIVE GEODESICS DEBUG")
    print("="*50)
    
    test_hardcoded_values()
    check_constant_files() 
    test_tacc_implementation()
    test_capacity_implementation()
    test_deprecated_function()
    
    print("CONCLUSION:")
    print("- If hardcoded test shows ~1.75″ for γ=1, the math is correct")
    print("- If any implementation shows ~4e5″, that's the source of the bug")
    print("- The issue is likely using null_geodesic_deflection(1.0, ...) instead of proper functions")
