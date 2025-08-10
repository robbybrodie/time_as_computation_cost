"""
TACC Cosmology Module: Friedmann-Lemaître-Robertson-Walker (FLRW) metrics adapted for computational capacity.
"""

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
from . import constants
from . import constitutive

def hubble_parameter_tacc(z, kappa, H0=70.0, Omega_m=0.3, Omega_de=0.7):
    """
    Compute Hubble parameter H(z) in TACC cosmology.
    
    Parameters:
    -----------
    z : float or array
        Redshift
    kappa : float
        TACC constitutive parameter
    H0 : float
        Hubble constant in km/s/Mpc
    Omega_m : float
        Matter density parameter
    Omega_de : float
        Dark energy density parameter (or effective TACC parameter)
    
    Returns:
    --------
    H : float or array
        Hubble parameter at redshift z
    """
    a = 1.0 / (1.0 + z)  # Scale factor
    
    # Modified Friedmann equation with TACC effects
    # In TACC, computational capacity constraints modify expansion
    B_factor = constitutive.B_of_N(a, kappa)  # Use scale factor as proxy for N
    
    # Modified energy density evolution
    rho_m = Omega_m * (1 + z)**3
    rho_de_eff = Omega_de * B_factor  # TACC modification to dark energy
    
    H_squared = H0**2 * (rho_m + rho_de_eff)
    return np.sqrt(H_squared)

def luminosity_distance_tacc(z, kappa, H0=70.0, Omega_m=0.3, Omega_de=0.7):
    """
    Compute luminosity distance in TACC cosmology.
    
    Parameters:
    -----------
    z : float or array
        Redshift
    kappa : float
        TACC constitutive parameter
    H0 : float
        Hubble constant in km/s/Mpc
    Omega_m : float
        Matter density parameter
    Omega_de : float
        Dark energy density parameter
    
    Returns:
    --------
    d_L : float or array
        Luminosity distance in Mpc
    """
    c_km_s = constants.c * 1e-3  # Speed of light in km/s
    
    if np.isscalar(z):
        z_array = np.array([z])
    else:
        z_array = np.array(z)
    
    d_L = np.zeros_like(z_array)
    
    for i, z_val in enumerate(z_array):
        if z_val <= 0:
            d_L[i] = 0
        else:
            # Comoving distance integral
            def integrand(z_prime):
                return 1.0 / hubble_parameter_tacc(z_prime, kappa, H0, Omega_m, Omega_de)
            
            d_c, _ = integrate.quad(integrand, 0, z_val)
            d_c *= c_km_s  # Convert to Mpc
            
            # Luminosity distance
            d_L[i] = d_c * (1 + z_val)
    
    return d_L[0] if np.isscalar(z) else d_L

def distance_modulus_tacc(z, kappa, H0=70.0, Omega_m=0.3, Omega_de=0.7):
    """
    Compute distance modulus in TACC cosmology.
    
    Parameters:
    -----------
    z : float or array
        Redshift
    kappa : float
        TACC constitutive parameter
    H0 : float
        Hubble constant
    Omega_m : float
        Matter density parameter
    Omega_de : float
        Dark energy density parameter
    
    Returns:
    --------
    mu : float or array
        Distance modulus in magnitudes
    """
    d_L = luminosity_distance_tacc(z, kappa, H0, Omega_m, Omega_de)
    
    # Distance modulus: μ = 5 * log10(d_L / 10 pc)
    # d_L is in Mpc, so convert: 1 Mpc = 10^6 pc
    mu = 5 * np.log10(d_L * 1e6 / 10.0)
    
    return mu

def scale_factor_evolution(t, kappa, H0=70.0, Omega_m=0.3, Omega_de=0.7):
    """
    Compute scale factor evolution a(t) in TACC cosmology.
    
    Parameters:
    -----------
    t : float or array
        Time in Gyr
    kappa : float
        TACC constitutive parameter
    H0 : float
        Hubble constant
    Omega_m : float
        Matter density parameter
    Omega_de : float
        Dark energy density parameter
    
    Returns:
    --------
    a : float or array
        Scale factor (normalized to a=1 today)
    """
    # Convert H0 from km/s/Mpc to 1/Gyr
    H0_Gyr = H0 * 1.022e-3  # Conversion factor
    
    def daddt(t, a):
        """Scale factor derivative"""
        if a <= 0:
            return 0
        z = 1.0/a - 1.0
        H = hubble_parameter_tacc(z, kappa, H0, Omega_m, Omega_de) * 1.022e-3
        return a * H
    
    if np.isscalar(t):
        t_array = np.array([t])
    else:
        t_array = np.array(t)
    
    # Start from early time with small scale factor
    t0 = 0.1  # Gyr
    a0 = 0.01  # Small initial scale factor
    
    # Integrate forward to each requested time
    a_result = np.zeros_like(t_array)
    for i, t_val in enumerate(t_array):
        if t_val <= t0:
            a_result[i] = a0 * (t_val / t0)  # Linear extrapolation for very early times
        else:
            sol = integrate.solve_ivp(daddt, [t0, t_val], [a0], dense_output=True)
            a_result[i] = sol.sol(t_val)[0]
    
    # Normalize so that a=1 at present epoch (assume t_present = 13.8 Gyr)
    t_present = 13.8
    if t_present in t_array:
        norm_idx = np.argmin(np.abs(t_array - t_present))
        normalization = 1.0 / a_result[norm_idx]
    else:
        # Compute normalization separately
        sol = integrate.solve_ivp(daddt, [t0, t_present], [a0], dense_output=True)
        a_present = sol.sol(t_present)[0]
        normalization = 1.0 / a_present
    
    a_result *= normalization
    
    return a_result[0] if np.isscalar(t) else a_result

def fit_to_supernova_data(z_data, mu_data, mu_err=None):
    """
    Fit TACC cosmology parameters to supernova distance modulus data.
    
    Parameters:
    -----------
    z_data : array
        Redshift data
    mu_data : array
        Distance modulus data
    mu_err : array, optional
        Distance modulus uncertainties
    
    Returns:
    --------
    result : dict
        Best-fit parameters and goodness of fit
    """
    if mu_err is None:
        mu_err = np.ones_like(mu_data) * 0.1  # Default 0.1 mag uncertainty
    
    def chi_squared(params):
        """Chi-squared objective function"""
        kappa, H0, Omega_m = params
        Omega_de = 1.0 - Omega_m  # Flat universe assumption
        
        if kappa <= 0 or H0 <= 0 or Omega_m <= 0 or Omega_m >= 1:
            return 1e10  # Invalid parameter space
        
        mu_theory = distance_modulus_tacc(z_data, kappa, H0, Omega_m, Omega_de)
        residuals = (mu_data - mu_theory) / mu_err
        return np.sum(residuals**2)
    
    # Initial guess: ΛCDM-like parameters
    initial_params = [2.0, 70.0, 0.3]  # kappa, H0, Omega_m
    
    # Parameter bounds
    bounds = [(0.1, 5.0),    # kappa
              (50.0, 90.0),  # H0
              (0.1, 0.9)]    # Omega_m
    
    # Minimize chi-squared
    result = minimize(chi_squared, initial_params, bounds=bounds, method='L-BFGS-B')
    
    kappa_best, H0_best, Omega_m_best = result.x
    Omega_de_best = 1.0 - Omega_m_best
    chi2_min = result.fun
    
    # Compute reduced chi-squared
    dof = len(z_data) - 3  # 3 fitted parameters
    chi2_red = chi2_min / dof if dof > 0 else chi2_min
    
    return {
        'kappa': kappa_best,
        'H0': H0_best,
        'Omega_m': Omega_m_best,
        'Omega_de': Omega_de_best,
        'chi2': chi2_min,
        'chi2_red': chi2_red,
        'dof': dof,
        'success': result.success
    }

def generate_synthetic_supernova_data(z_max=2.0, n_points=50, kappa_true=2.0, noise_level=0.1):
    """
    Generate synthetic supernova distance modulus data for testing.
    
    Parameters:
    -----------
    z_max : float
        Maximum redshift
    n_points : int
        Number of data points
    kappa_true : float
        True value of kappa used to generate data
    noise_level : float
        Gaussian noise level in magnitudes
    
    Returns:
    --------
    z_data, mu_data, mu_err : arrays
        Synthetic redshift, distance modulus, and uncertainties
    """
    # Generate redshift array (weighted toward lower z like real surveys)
    z_data = np.random.exponential(0.3, n_points)
    z_data = z_data[z_data <= z_max]  # Truncate at z_max
    if len(z_data) < n_points:
        # Fill remaining with uniform distribution
        z_additional = np.random.uniform(0.01, z_max, n_points - len(z_data))
        z_data = np.concatenate([z_data, z_additional])
    
    z_data = z_data[:n_points]  # Ensure exact number of points
    z_data = np.sort(z_data)
    
    # Generate true distance moduli
    mu_true = distance_modulus_tacc(z_data, kappa_true)
    
    # Add Gaussian noise
    mu_noise = np.random.normal(0, noise_level, n_points)
    mu_data = mu_true + mu_noise
    
    # Error bars (assume constant for simplicity)
    mu_err = np.full_like(mu_data, noise_level)
    
    return z_data, mu_data, mu_err
