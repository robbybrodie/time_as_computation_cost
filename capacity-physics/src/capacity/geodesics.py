"""Geodesics: null/timelike integrators, lensing, Shapiro delay, perihelion precession."""

import numpy as np
from .constants import GM_SUN, R_SUN, c, ARCSEC_PER_RAD, AU
from .constants import MERCURY_SEMI_MAJOR_AXIS_AU, MERCURY_ECCENTRICITY
from .constants import OBSERVED_MERCURY_PRECESSION, SOLAR_LIMB_DEFLECTION_GR
try:
    from .constants import MERCURY_ORBITAL_PERIOD_DAYS
except ImportError:
    MERCURY_ORBITAL_PERIOD_DAYS = 88  # days - fallback value

def light_deflection_angle(b, gamma, GM=GM_SUN, c_light=c):
    """
    Compute light deflection angle for given impact parameter b and PPN parameter gamma.
    
    Uses the correct PPN formula: δθ = (1 + γ) * 2*GM / (b * c²)
    
    Args:
        b (float): Impact parameter in meters (e.g., R_SUN for solar limb)
        gamma (float): PPN parameter γ
        GM (float): Standard gravitational parameter in m³/s²
        c_light (float): Speed of light in m/s
        
    Returns:
        float: Deflection angle in radians
    """
    deflection_rad = (1 + gamma) * 2 * GM / (b * c_light**2)
    return deflection_rad

def solar_limb_deflection(gamma):
    """
    Compute light deflection at the solar limb (b = R☉).
    
    This is the classic test case where Einstein predicted 1.75 arcsec for γ=1.
    
    Args:
        gamma (float): PPN parameter γ
        
    Returns:
        dict: Contains deflection in radians and arcseconds
    """
    deflection_rad = light_deflection_angle(R_SUN, gamma)
    deflection_arcsec = deflection_rad * ARCSEC_PER_RAD
    
    return {
        'deflection_rad': deflection_rad,
        'deflection_arcsec': deflection_arcsec
    }

def null_geodesic_deflection(b_over_rs, gamma, GM=GM_SUN, c_light=c):
    """
    Compute light deflection angle for given impact parameter as ratio to Schwarzschild radius.
    
    DEPRECATED: This function uses b_over_rs parameterization which led to the original bug.
    Use light_deflection_angle() or solar_limb_deflection() instead for physical calculations.
    
    Args:
        b_over_rs (float): Impact parameter divided by Schwarzschild radius
        gamma (float): PPN parameter γ
        GM (float): Standard gravitational parameter in m³/s²
        c_light (float): Speed of light in m/s
        
    Returns:
        float: Deflection angle in radians
    """
    rs = 2 * GM / c_light**2  # Schwarzschild radius
    b = b_over_rs * rs
    return light_deflection_angle(b, gamma, GM, c_light)

def shapiro_delay(b, gamma, rE, rR, GM=GM_SUN, c_light=c):
    """
    Compute Shapiro time delay for given impact parameter and gamma.
    
    Args:
        b (float): Impact parameter in meters
        gamma (float): PPN parameter γ
        rE (float): Earth distance from Sun in meters
        rR (float): Receiver distance from Sun in meters
        GM (float): Standard gravitational parameter in m³/s²
        c_light (float): Speed of light in m/s
        
    Returns:
        float: Time delay in seconds
    """
    delay = (1 + gamma) * GM / c_light**3 * np.log(4 * rE * rR / b**2)
    return delay

def perihelion_precession(a_AU, e, beta, gamma, GM=GM_SUN, c_light=c):
    """
    Compute perihelion precession for Mercury-like orbit.
    
    Args:
        a_AU (float): Semi-major axis in AU
        e (float): Orbital eccentricity
        beta (float): PPN parameter β (usually 1.0 in our theory)
        gamma (float): PPN parameter γ
        GM (float): Standard gravitational parameter in m³/s²
        c_light (float): Speed of light in m/s
        
    Returns:
        float: Precession per orbit in radians
    """
    a = a_AU * AU
    precession = (2 - beta + 2 * gamma) * 3 * np.pi * GM / (a * (1 - e**2) * c_light**2)
    return precession

def mercury_precession_arcsec_per_century(gamma, beta=1.0):
    """
    Compute Mercury's perihelion precession in arcsec/century.
    
    Args:
        gamma (float): PPN parameter γ
        beta (float): PPN parameter β (default 1.0)
        
    Returns:
        float: Precession in arcseconds per century
    """
    precession_per_orbit = perihelion_precession(
        MERCURY_SEMI_MAJOR_AXIS_AU, MERCURY_ECCENTRICITY, beta, gamma
    )
    
    # Convert to arcsec/century
    orbits_per_century = 100 * 365.25 / MERCURY_ORBITAL_PERIOD_DAYS
    precession_arcsec_century = (
        precession_per_orbit * ARCSEC_PER_RAD * orbits_per_century
    )
    
    return precession_arcsec_century

def integrate_geodesic(*args, **kwargs):
    """Stub for numerical geodesic integrator (to be implemented)."""
    pass

# Unit tests / sanity checks
def test_solar_limb_deflection():
    """Test solar limb deflection for γ=1 should give ~1.75 arcsec."""
    result = solar_limb_deflection(gamma=1.0)
    deflection_arcsec = result['deflection_arcsec']
    
    tolerance = 0.05  # ±0.05 arcsec tolerance
    expected = SOLAR_LIMB_DEFLECTION_GR
    
    assert abs(deflection_arcsec - expected) <= tolerance, (
        f"Solar limb deflection test failed: got {deflection_arcsec:.3f} arcsec, "
        f"expected {expected:.3f} ± {tolerance:.3f} arcsec"
    )
    
    return True

def test_deflection_scaling():
    """Test deflection scaling: δθ(γ=0.5) : δθ(γ=1.0) : δθ(γ=1.5) ≈ 1.5 : 2.0 : 2.5"""
    gamma_values = [0.5, 1.0, 1.5]
    deflections = []
    
    for gamma in gamma_values:
        result = solar_limb_deflection(gamma)
        deflections.append(result['deflection_arcsec'])
    
    # Normalize by γ=1.0 case
    ratios = [d / deflections[1] for d in deflections]
    expected_ratios = [1.5, 2.0, 2.5]
    
    tolerance = 0.02
    for i, (actual, expected) in enumerate(zip(ratios, expected_ratios)):
        assert abs(actual - expected) <= tolerance, (
            f"Deflection scaling test failed at γ={gamma_values[i]}: "
            f"got ratio {actual:.3f}, expected {expected:.3f} ± {tolerance:.3f}"
        )
    
    return True

def test_mercury_precession():
    """Test Mercury precession should be 42-44 arcsec/century for γ≈1."""
    precession = mercury_precession_arcsec_per_century(gamma=1.0)
    
    min_expected = 42.0
    max_expected = 44.0
    
    assert min_expected <= precession <= max_expected, (
        f"Mercury precession test failed: got {precession:.1f} arcsec/century, "
        f"expected {min_expected}-{max_expected} arcsec/century"
    )
    
    return True

def run_all_tests():
    """Run all geodesics sanity checks."""
    tests = [
        ("Solar limb deflection", test_solar_limb_deflection),
        ("Deflection scaling", test_deflection_scaling), 
        ("Mercury precession", test_mercury_precession)
    ]
    
    print("Running geodesics sanity checks...")
    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}: PASSED")
        except AssertionError as e:
            print(f"✗ {name}: FAILED - {e}")
            return False
        except Exception as e:
            print(f"✗ {name}: ERROR - {e}")
            return False
    
    print("All geodesics tests passed!")
    return True
