"""
TACC Gravitational Wave Propagation: Linearized perturbations in TACC metric.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize
from . import constants
from . import constitutive

def strain_amplitude_tacc(M1, M2, distance, frequency, kappa):
    """
    Compute gravitational wave strain amplitude in TACC.
    
    Parameters:
    -----------
    M1, M2 : float
        Binary component masses in kg
    distance : float
        Distance to source in meters
    frequency : float
        Gravitational wave frequency in Hz
    kappa : float
        TACC constitutive parameter
    
    Returns:
    --------
    h : float
        Strain amplitude
    """
    # Chirp mass
    M_total = M1 + M2
    mu = M1 * M2 / M_total  # Reduced mass
    M_chirp = (mu**(3/5)) * (M_total**(2/5))
    
    # Standard GR strain amplitude
    h_gr = (4/5) * (constants.G * M_chirp / constants.c**2)**(5/3) * (np.pi * frequency / constants.c)**(2/3) / distance
    
    # TACC modification through computational capacity
    # Estimate N from orbital characteristics
    r_orbit = (constants.G * M_total / (4 * np.pi**2 * frequency**2))**(1/3)  # Orbital radius
    Phi = -constants.G * M_total / r_orbit  # Gravitational potential
    N = 1.0 + Phi / constants.c**2
    
    B = constitutive.B_of_N(N, kappa)
    
    # Modified strain (computational capacity affects wave generation)
    h = h_gr * B
    
    return h

def phase_evolution_tacc(M1, M2, t, kappa, f_initial=10.0):
    """
    Compute phase evolution for binary inspiral in TACC.
    
    Parameters:
    -----------
    M1, M2 : float
        Binary component masses in kg
    t : float or array
        Time array in seconds
    kappa : float
        TACC constitutive parameter
    f_initial : float
        Initial frequency in Hz
    
    Returns:
    --------
    phase : float or array
        Gravitational wave phase
    frequency : float or array
        Instantaneous frequency
    """
    M_total = M1 + M2
    mu = M1 * M2 / M_total
    M_chirp = (mu**(3/5)) * (M_total**(2/5))
    
    # Time to coalescence from initial frequency
    t_coal = 5 * constants.c**5 / (256 * constants.G**(5/3) * (np.pi * M_chirp)**(5/3) * f_initial**(8/3))
    
    def frequency_evolution(t_val):
        """Frequency evolution with TACC corrections"""
        if t_val >= t_coal:
            return np.inf  # Coalescence
        
        tau = t_coal - t_val  # Time to coalescence
        
        # Standard 3.5PN frequency evolution
        f_gr = (1/(8*np.pi)) * (5*constants.G*M_chirp/constants.c**3)**(-3/8) * tau**(-3/8)
        
        # TACC modification
        r_orbit = (constants.G * M_total / (4 * np.pi**2 * f_gr**2))**(1/3)
        Phi = -constants.G * M_total / r_orbit
        N = 1.0 + Phi / constants.c**2
        B = constitutive.B_of_N(N, kappa)
        
        # Modified frequency evolution (computational constraints affect inspiral)
        f_tacc = f_gr * np.sqrt(B)
        
        return f_tacc
    
    if np.isscalar(t):
        t_array = np.array([t])
    else:
        t_array = np.array(t)
    
    frequency = np.array([frequency_evolution(t_val) for t_val in t_array])
    
    # Phase by integration
    phase = np.zeros_like(frequency)
    for i in range(1, len(t_array)):
        dt = t_array[i] - t_array[i-1]
        phase[i] = phase[i-1] + 2 * np.pi * frequency[i-1] * dt
    
    return (phase[0], frequency[0]) if np.isscalar(t) else (phase, frequency)

def waveform_tacc(M1, M2, distance, t, kappa, inclination=0.0, polarization=0.0):
    """
    Generate gravitational waveform in TACC.
    
    Parameters:
    -----------
    M1, M2 : float
        Binary component masses in kg
    distance : float
        Distance to source in meters
    t : array
        Time array in seconds
    kappa : float
        TACC constitutive parameter
    inclination : float
        Inclination angle in radians
    polarization : float
        Polarization angle in radians
    
    Returns:
    --------
    h_plus, h_cross : arrays
        Plus and cross polarizations
    """
    phase, frequency = phase_evolution_tacc(M1, M2, t, kappa)
    
    # Strain amplitude as function of time
    h_amp = np.array([strain_amplitude_tacc(M1, M2, distance, f, kappa) for f in frequency])
    
    # Polarization factors
    cos_i = np.cos(inclination)
    F_plus = (1 + cos_i**2) / 2
    F_cross = cos_i
    
    # Waveforms
    h_plus = h_amp * F_plus * np.cos(phase)
    h_cross = h_amp * F_cross * np.sin(phase)
    
    return h_plus, h_cross

def dispersion_relation_tacc(frequency, kappa, N_background=1.0):
    """
    Compute dispersion relation for gravitational waves in TACC.
    
    Parameters:
    -----------
    frequency : float or array
        Gravitational wave frequency in Hz
    kappa : float
        TACC constitutive parameter
    N_background : float
        Background computational capacity
    
    Returns:
    --------
    v_gw : float or array
        Gravitational wave speed (as fraction of c)
    """
    # In GR: v_gw = c exactly
    # In TACC: computational capacity can modify propagation
    
    B = constitutive.B_of_N(N_background, kappa)
    
    # Modified dispersion: ω² = k²c²B
    # This gives v_gw = c√B
    v_gw = constants.c * np.sqrt(B)
    
    return v_gw

def time_delay_tacc(frequency, distance, kappa, N_path=1.0):
    """
    Compute time delay for gravitational waves in TACC.
    
    Parameters:
    -----------
    frequency : float or array
        Gravitational wave frequency in Hz
    distance : float
        Distance traveled in meters
    kappa : float
        TACC constitutive parameter
    N_path : float
        Average computational capacity along path
    
    Returns:
    --------
    delta_t : float or array
        Time delay relative to light in seconds
    """
    v_gw = dispersion_relation_tacc(frequency, kappa, N_path)
    
    # Time delay relative to light
    t_light = distance / constants.c
    t_gw = distance / v_gw
    delta_t = t_gw - t_light
    
    return delta_t

def ligo_response_tacc(h_plus, h_cross, detector_angle=0.0):
    """
    Compute LIGO detector response in TACC.
    
    Parameters:
    -----------
    h_plus, h_cross : arrays
        Plus and cross polarizations
    detector_angle : float
        Detector orientation angle
    
    Returns:
    --------
    h_detector : array
        Detector strain response
    """
    # LIGO antenna patterns (simplified)
    F_plus = (1 + np.cos(detector_angle)**2) / 2
    F_cross = np.cos(detector_angle)
    
    h_detector = F_plus * h_plus + F_cross * h_cross
    
    return h_detector

def fit_gw150914_data(kappa_guess=2.0):
    """
    Fit TACC parameters to GW150914-like event.
    
    Parameters:
    -----------
    kappa_guess : float
        Initial guess for kappa
    
    Returns:
    --------
    result : dict
        Best-fit parameters and waveform comparison
    """
    # GW150914 approximate parameters
    M1 = 36 * 1.989e30  # kg (36 solar masses)
    M2 = 29 * 1.989e30  # kg (29 solar masses)
    distance = 410e6 * 9.461e15  # meters (410 Mpc)
    
    # Time array around merger
    t = np.linspace(-0.2, 0.05, 1000)  # seconds relative to merger
    
    def chi_squared(kappa):
        """Chi-squared for parameter fitting"""
        if kappa <= 0:
            return 1e10
        
        # Generate TACC waveform
        h_plus, h_cross = waveform_tacc(M1, M2, distance, t, kappa)
        h_detector = ligo_response_tacc(h_plus, h_cross)
        
        # Generate GR reference
        h_plus_gr, h_cross_gr = waveform_tacc(M1, M2, distance, t, 2.0)  # κ=2 is GR
        h_detector_gr = ligo_response_tacc(h_plus_gr, h_cross_gr)
        
        # Synthetic "observed" data (GR + noise)
        noise_level = np.max(np.abs(h_detector_gr)) * 0.1
        h_observed = h_detector_gr + np.random.normal(0, noise_level, len(t))
        
        # Chi-squared
        residuals = (h_observed - h_detector) / noise_level
        return np.sum(residuals**2)
    
    # Minimize chi-squared
    result = minimize(chi_squared, [kappa_guess], bounds=[(0.1, 5.0)], method='L-BFGS-B')
    
    kappa_best = result.x[0]
    
    # Generate best-fit waveform
    h_plus, h_cross = waveform_tacc(M1, M2, distance, t, kappa_best)
    h_detector = ligo_response_tacc(h_plus, h_cross)
    
    return {
        'kappa': kappa_best,
        'chi2': result.fun,
        'time': t,
        'h_plus': h_plus,
        'h_cross': h_cross,
        'h_detector': h_detector,
        'M1': M1,
        'M2': M2,
        'distance': distance,
        'success': result.success
    }

def multi_messenger_constraints(gw_time, em_time, distance, kappa):
    """
    Analyze multi-messenger constraints on TACC.
    
    Parameters:
    -----------
    gw_time : float
        Gravitational wave arrival time
    em_time : float
        Electromagnetic signal arrival time
    distance : float
        Distance to source in meters
    kappa : float
        TACC constitutive parameter
    
    Returns:
    --------
    constraints : dict
        Multi-messenger constraint analysis
    """
    # Time delay between GW and EM
    delta_t_observed = em_time - gw_time
    
    # Predicted delay in TACC
    frequency_gw = 100.0  # Hz (typical LIGO frequency)
    delta_t_predicted = time_delay_tacc(frequency_gw, distance, kappa)
    
    # Constraint on kappa
    residual = delta_t_observed - delta_t_predicted
    
    return {
        'delta_t_observed': delta_t_observed,
        'delta_t_predicted': delta_t_predicted,
        'residual': residual,
        'fractional_speed_difference': delta_t_predicted * constants.c / distance
    }

def generate_synthetic_gw_data(M1, M2, distance, kappa_true, noise_snr=20.0, duration=1.0):
    """
    Generate synthetic gravitational wave data for testing.
    
    Parameters:
    -----------
    M1, M2 : float
        Binary masses in kg
    distance : float
        Distance in meters
    kappa_true : float
        True kappa value for data generation
    noise_snr : float
        Signal-to-noise ratio
    duration : float
        Observation duration in seconds
    
    Returns:
    --------
    t, h_data, noise_level : arrays and float
        Time array, strain data, and noise level
    """
    # Time array
    sample_rate = 4096  # Hz (LIGO-like)
    n_samples = int(duration * sample_rate)
    t = np.linspace(-duration/2, duration/2, n_samples)
    
    # Generate waveform
    h_plus, h_cross = waveform_tacc(M1, M2, distance, t, kappa_true)
    h_clean = ligo_response_tacc(h_plus, h_cross)
    
    # Add noise
    signal_power = np.mean(h_clean**2)
    noise_power = signal_power / noise_snr**2
    noise = np.random.normal(0, np.sqrt(noise_power), len(t))
    
    h_data = h_clean + noise
    noise_level = np.sqrt(noise_power)
    
    return t, h_data, noise_level

def compare_waveforms_tacc_gr(M1, M2, distance, kappa_values=None, duration=0.5):
    """
    Compare TACC and GR waveforms for different kappa values.
    
    Parameters:
    -----------
    M1, M2 : float
        Binary masses in kg
    distance : float
        Distance in meters
    kappa_values : array, optional
        Array of kappa values to compare
    duration : float
        Waveform duration in seconds
    
    Returns:
    --------
    comparison : dict
        Waveform comparison data
    """
    if kappa_values is None:
        kappa_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    sample_rate = 2048  # Hz
    n_samples = int(duration * sample_rate)
    t = np.linspace(-duration/2, duration/2, n_samples)
    
    results = {
        'time': t,
        'kappa_values': kappa_values,
        'waveforms': {}
    }
    
    for kappa in kappa_values:
        h_plus, h_cross = waveform_tacc(M1, M2, distance, t, kappa)
        h_detector = ligo_response_tacc(h_plus, h_cross)
        
        results['waveforms'][kappa] = {
            'h_plus': h_plus,
            'h_cross': h_cross,
            'h_detector': h_detector
        }
    
    return results
