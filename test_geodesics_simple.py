#!/usr/bin/env python3
"""
Simple test to verify geodesics calculations match expected values.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "src"))

from tacc.geodesics import solar_limb_deflection
from tacc.constants import GM_SUN, R_SUN, c, ARCSEC_PER_RAD

def test_with_oneliner():
    """Test using the provided one-liner formula."""
    print("Testing with one-liner formula:")
    print("="*40)
    
    # One-liner from user
    GM = 1.32712440018e20      # m^3/s^2
    Rsun = 6.957e8             # m
    c_light = 299_792_458      # m/s
    ARCSEC = 206_265           # arcsec per radian
    deflection = lambda g: (1+g)*2*GM/(Rsun*c_light**2)*ARCSEC
    
    gamma_values = [0.5, 1.0, 1.5, 2.0]
    expected_values = [1.3134, 1.7512, 2.1890, 2.6268]
    
    print("Expected values from one-liner:")
    for g, expected in zip(gamma_values, expected_values):
        calculated = deflection(g)
        print(f"γ = {g}: {calculated:.4f} arcsec (expected: {expected:.4f})")
    
    print()

def test_with_our_functions():
    """Test using our geodesics functions."""
    print("Testing with our geodesics functions:")
    print("="*40)
    
    gamma_values = [0.5, 1.0, 1.5, 2.0]
    
    print("Results from our functions:")
    for gamma in gamma_values:
        result = solar_limb_deflection(gamma)
        print(f"γ = {gamma}: {result['deflection_arcsec']:.4f} arcsec")
    
    print()

def compare_constants():
    """Compare constants used in both calculations."""
    print("Comparing constants:")
    print("="*40)
    
    # User's one-liner constants
    GM_user = 1.32712440018e20
    Rsun_user = 6.957e8
    c_user = 299_792_458
    ARCSEC_user = 206_265
    
    print(f"GM_SUN:         {GM_SUN} (ours) vs {GM_user} (user) - Match: {GM_SUN == GM_user}")
    print(f"R_SUN:          {R_SUN} (ours) vs {Rsun_user} (user) - Match: {R_SUN == Rsun_user}")
    print(f"c:              {c} (ours) vs {c_user} (user) - Match: {c == c_user}")
    print(f"ARCSEC_PER_RAD: {ARCSEC_PER_RAD} (ours) vs {ARCSEC_user} (user) - Match: {ARCSEC_PER_RAD == ARCSEC_user}")
    print()

def generate_corrected_summary():
    """Generate the corrected GEODESICS SUMMARY block."""
    print("📋 GEODESICS SUMMARY:")
    print("   Geodesics Experiment Results")
    print("   ============================")
    print("   1. Light Bending (at solar limb, b ≈ R☉):")
    
    gamma_values = [0.5, 1.0, 1.5, 2.0]
    for gamma in gamma_values:
        result = solar_limb_deflection(gamma)
        print(f"      γ = {gamma}: {result['deflection_arcsec']:.4f} arcsec")
    
    print("   [computed via δθ = (1+γ)·2GM/(R☉ c²); 1 rad = 206265″]")
    print("Quick sanity: γ=1 ⇒ ~1.75″ (textbook value).")

if __name__ == "__main__":
    compare_constants()
    test_with_oneliner()
    test_with_our_functions()
    generate_corrected_summary()
