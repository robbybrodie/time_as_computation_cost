"""Geodesics: null/timelike integrators, lensing, Shapiro delay, perihelion precession."""

import numpy as np

def null_geodesic_deflection(b_over_rs, gamma, GM, c=1.0):
    """
    Compute light deflection angle for given impact parameter b/rs and gamma.
    Uses PPN formula: alpha ≈ (1 + gamma) * 2 * GM / (b * c^2)
    """
    b = b_over_rs * 2 * GM / c**2
    alpha = (1 + gamma) * 2 * GM / (b * c**2)
    return alpha

def shapiro_delay(impact_b_over_rs, gamma, GM, rE, rR, c=1.0):
    """
    Compute Shapiro time delay for given impact parameter and gamma.
    Δt ≈ (1 + gamma) * GM / c^3 * ln(4 rE rR / b^2)
    """
    b = impact_b_over_rs * 2 * GM / c**2
    delay = (1 + gamma) * GM / c**3 * np.log(4 * rE * rR / b**2)
    return delay

def perihelion_precession(a_AU, e, beta, gamma, GM, c=1.0):
    """
    Compute perihelion precession for Mercury-like orbit.
    Δω ≈ (2 - beta + 2*gamma) * 3π * GM / (a * (1 - e^2) * c^2)
    """
    AU = 1.495978707e11  # meters
    a = a_AU * AU
    precession = (2 - beta + 2 * gamma) * 3 * np.pi * GM / (a * (1 - e**2) * c**2)
    return precession

def integrate_geodesic(*args, **kwargs):
    """Stub for numerical geodesic integrator (to be implemented)."""
    pass
