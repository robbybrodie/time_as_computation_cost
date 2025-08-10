"""
TACC Black Hole Thermodynamics: Schwarzschild/Kerr metrics adapted for computational capacity.
"""

import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from . import constants
from . import constitutive
from . import metric

def schwarzschild_radius(M):
    """
    Compute Schwarzschild radius for mass M.
    
    Parameters:
    -----------
    M : float
        Mass in kg
    
    Returns:
    --------
    r_s : float
        Schwarzschild radius in meters
    """
    return 2 * constants.G * M / constants.c**2

def tacc_metric_schwarzschild(r, M, kappa):
    """
    Compute TACC-modified Schwarzschild metric coefficients.
    
    Parameters:
    -----------
    r : float or array
        Radial coordinate in meters
    M : float
        Mass in kg
    kappa : float
        TACC constitutive parameter
    
    Returns:
    --------
    g_tt, g_rr : float or array
        Metric coefficients g_tt and g_rr
    """
    r_s = schwarzschild_radius(M)
    
    # Standard Schwarzschild potential
    Phi_N = -constants.G * M / r  # Newtonian potential
    
    # Computational capacity (near-unity expansion)
    N = 1.0 + Phi_N / constants.c**2
    
    # TACC modification
    B = constitutive.B_of_N(N, kappa)
    
    # Modified metric coefficients
    g_tt = -(1 - r_s/r) * B
    g_rr = 1 / (1 - r_s/r) * B
    
    return g_tt, g_rr

def hawking_temperature_tacc(M, kappa):
    """
    Compute Hawking temperature in TACC black hole thermodynamics.
    
    Parameters:
    -----------
    M : float
        Black hole mass in kg
    kappa : float
        TACC constitutive parameter
    
    Returns:
    --------
    T_H : float
        Hawking temperature in Kelvin
    """
    # Standard Hawking temperature
    T_H_standard = constants.hbar * constants.c**3 / (8 * np.pi * constants.G * M * constants.k_B)
    
    # TACC modification: temperature scales with computational capacity at horizon
    r_s = schwarzschild_radius(M)
    Phi_horizon = -constants.G * M / r_s  # This is -c²/2
    N_horizon = 1.0 + Phi_horizon / constants.c**2  # This is 0.5
    
    B_horizon = constitutive.B_of_N(N_horizon, kappa)
    
    # Modified temperature (scaling with B factor)
    T_H = T_H_standard * B_horizon
    
    return T_H

def hawking_temperature_gr(M):
    """Standard GR Hawking temperature for comparison."""
    return constants.hbar * constants.c**3 / (8 * np.pi * constants.G * M * constants.k_B)

def bekenstein_hawking_entropy_tacc(M, kappa):
    """
    Compute Bekenstein-Hawking entropy in TACC.
    
    Parameters:
    -----------
    M : float
        Black hole mass in kg
    kappa : float
        TACC constitutive parameter
    
    Returns:
    --------
    S_BH : float
        Bekenstein-Hawking entropy in J/K
    """
    r_s = schwarzschild_radius(M)
    A = 4 * np.pi * r_s**2  # Horizon area
    
    # Standard entropy
    S_BH_standard = constants.k_B * constants.c**3 * A / (4 * constants.G * constants.hbar)
    
    # TACC modification through computational capacity
    Phi_horizon = -constants.G * M / r_s
    N_horizon = 1.0 + Phi_horizon / constants.c**2
    B_horizon = constitutive.B_of_N(N_horizon, kappa)
    
    # Modified entropy (information-theoretic interpretation)
    S_BH = S_BH_standard * B_horizon
    
    return S_BH

def black_hole_evaporation_time_tacc(M, kappa):
    """
    Compute black hole evaporation time in TACC.
    
    Parameters:
    -----------
    M : float
        Initial black hole mass in kg
    kappa : float
        TACC constitutive parameter
    
    Returns:
    --------
    t_evap : float
        Evaporation time in seconds
    """
    # Stefan-Boltzmann law for black hole power
    # P = σ A T^4, where σ is Stefan-Boltzmann constant
    
    def mass_loss_rate(M_current):
        """Mass loss rate dm/dt"""
        if M_current <= 0:
            return 0
        
        T_H = hawking_temperature_tacc(M_current, kappa)
        r_s = schwarzschild_radius(M_current)
        A = 4 * np.pi * r_s**2
        
        # Power radiated (Stefan-Boltzmann)
        sigma_SB = 2 * np.pi**5 * constants.k_B**4 / (15 * constants.c**2 * constants.hbar**3)
        P = sigma_SB * A * T_H**4
        
        # Convert power to mass loss rate: E = mc²
        dmdt = -P / constants.c**2
        return dmdt
    
    # Standard result for comparison: t ~ M³
    # For TACC, we need to integrate the modified mass loss rate
    
    # Approximate analytical result using scaling
    T_H = hawking_temperature_tacc(M, kappa)
    T_H_standard = hawking_temperature_gr(M)
    
    # Standard evaporation time
    t_evap_standard = 5120 * np.pi * constants.G**2 * M**3 / (constants.hbar * constants.c**4)
    
    # TACC modification
    temperature_ratio = T_H / T_H_standard
    t_evap = t_evap_standard / temperature_ratio**4  # Power scales as T^4
    
    return t_evap

def event_horizon_properties_tacc(M, kappa):
    """
    Compute event horizon properties in TACC.
    
    Parameters:
    -----------
    M : float
        Black hole mass in kg
    kappa : float
        TACC constitutive parameter
    
    Returns:
    --------
    properties : dict
        Dictionary of horizon properties
    """
    r_s = schwarzschild_radius(M)
    
    # Computational capacity at horizon
    Phi_horizon = -constants.G * M / r_s
    N_horizon = 1.0 + Phi_horizon / constants.c**2
    B_horizon = constitutive.B_of_N(N_horizon, kappa)
    
    # Surface gravity
    kappa_surface = constants.c**4 / (4 * constants.G * M)  # Standard result
    kappa_surface_tacc = kappa_surface * B_horizon  # TACC modification
    
    # Area
    A = 4 * np.pi * r_s**2
    
    # Temperature
    T_H = hawking_temperature_tacc(M, kappa)
    
    # Entropy
    S_BH = bekenstein_hawking_entropy_tacc(M, kappa)
    
    return {
        'radius': r_s,
        'area': A,
        'temperature': T_H,
        'entropy': S_BH,
        'surface_gravity': kappa_surface_tacc,
        'N_horizon': N_horizon,
        'B_horizon': B_horizon,
        'evaporation_time': black_hole_evaporation_time_tacc(M, kappa)
    }

def black_hole_shadow_angular_size(M, distance, kappa):
    """
    Compute black hole shadow angular size in TACC.
    
    Parameters:
    -----------
    M : float
        Black hole mass in kg
    distance : float
        Distance to black hole in meters
    kappa : float
        TACC constitutive parameter
    
    Returns:
    --------
    theta_shadow : float
        Shadow angular size in radians
    """
    # Shadow radius is approximately 3√3/2 times Schwarzschild radius
    r_s = schwarzschild_radius(M)
    r_shadow = 3 * np.sqrt(3) / 2 * r_s
    
    # TACC modification through light bending
    # This is a simplified model - full ray tracing would be needed for precision
    Phi = -constants.G * M / r_shadow
    N = 1.0 + Phi / constants.c**2
    B = constitutive.B_of_N(N, kappa)
    
    # Modified shadow radius
    r_shadow_tacc = r_shadow * np.sqrt(B)  # Rough approximation
    
    # Angular size
    theta_shadow = r_shadow_tacc / distance
    
    return theta_shadow

def fit_black_hole_shadow_data(M_data, theta_data, theta_err, distance):
    """
    Fit kappa parameter to black hole shadow observations.
    
    Parameters:
    -----------
    M_data : array
        Black hole mass estimates in kg
    theta_data : array
        Observed shadow angular sizes in radians
    theta_err : array
        Uncertainties in angular size
    distance : float
        Distance to black hole in meters
    
    Returns:
    --------
    result : dict
        Best-fit kappa and goodness of fit
    """
    def chi_squared(kappa):
        """Chi-squared objective function"""
        if kappa <= 0:
            return 1e10
        
        theta_theory = np.array([black_hole_shadow_angular_size(M, distance, kappa) for M in M_data])
        residuals = (theta_data - theta_theory) / theta_err
        return np.sum(residuals**2)
    
    # Minimize chi-squared
    result = minimize_scalar(chi_squared, bounds=(0.1, 5.0), method='bounded')
    
    kappa_best = result.x
    chi2_min = result.fun
    
    return {
        'kappa': kappa_best,
        'chi2': chi2_min,
        'success': result.success
    }

def m87_black_hole_parameters():
    """
    Return M87* black hole parameters for analysis.
    
    Returns:
    --------
    params : dict
        M87* parameters
    """
    # M87* parameters from Event Horizon Telescope
    M_solar = 1.989e30  # kg
    M87_mass = 6.5e9 * M_solar  # kg
    M87_distance = 16.8e6 * 9.461e15  # meters (16.8 Mpc)
    
    # Shadow angular size from EHT (approximate)
    theta_shadow_observed = 42e-6 * (np.pi / 180) / 3600  # 42 microarcsec in radians
    theta_shadow_error = 3e-6 * (np.pi / 180) / 3600  # 3 microarcsec error
    
    return {
        'mass': M87_mass,
        'distance': M87_distance,
        'shadow_angle': theta_shadow_observed,
        'shadow_error': theta_shadow_error
    }

def compare_with_gr(M, kappa_values=None):
    """
    Compare TACC black hole properties with GR predictions.
    
    Parameters:
    -----------
    M : float
        Black hole mass in kg
    kappa_values : array, optional
        Array of kappa values to test
    
    Returns:
    --------
    comparison : dict
        Comparison of TACC vs GR properties
    """
    if kappa_values is None:
        kappa_values = np.linspace(0.5, 4.0, 20)
    
    # GR results
    gr_results = {
        'temperature': hawking_temperature_gr(M),
        'entropy': constants.k_B * constants.c**3 * (4 * np.pi * schwarzschild_radius(M)**2) / (4 * constants.G * constants.hbar),
        'evaporation_time': 5120 * np.pi * constants.G**2 * M**3 / (constants.hbar * constants.c**4)
    }
    
    # TACC results
    tacc_results = {
        'kappa': kappa_values,
        'temperature': [],
        'entropy': [],
        'evaporation_time': []
    }
    
    for kappa in kappa_values:
        tacc_results['temperature'].append(hawking_temperature_tacc(M, kappa))
        tacc_results['entropy'].append(bekenstein_hawking_entropy_tacc(M, kappa))
        tacc_results['evaporation_time'].append(black_hole_evaporation_time_tacc(M, kappa))
    
    tacc_results['temperature'] = np.array(tacc_results['temperature'])
    tacc_results['entropy'] = np.array(tacc_results['entropy'])
    tacc_results['evaporation_time'] = np.array(tacc_results['evaporation_time'])
    
    return {
        'gr': gr_results,
        'tacc': tacc_results
    }
