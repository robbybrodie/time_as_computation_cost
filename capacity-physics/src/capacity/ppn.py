"""PPN expansion: series near N=1, extract gamma, beta, test SR limit."""

import numpy as np

def expand_N_to_Phi(N):
    """Map N to Newtonian potential Phi/c^2 near N=1."""
    # N ≈ 1 + Phi/c^2
    return N - 1

def extract_ppn_params(kappa):
    """
    Extract PPN parameters gamma, beta from metric expansion.
    
    Our model's dictionary:
    - For spatial factor 1/B(N), gamma = κ/2
    - For N^2 = 1 + 2*Phi/c^2 + 2*beta*(Phi/c^2)^2, beta = 1
    
    General Relativity limit: κ = 2 ⇒ γ = 1, β = 1
    Solar System observational constraint: γ ≈ 1.0 (very precisely), so κ ≈ 2
    
    Args:
        kappa (float): Our model parameter κ
        
    Returns:
        tuple: (gamma, beta) PPN parameters
    """
    gamma = kappa / 2
    beta = 1.0
    return gamma, beta

def get_gr_limit():
    """
    Return the General Relativity limit values.
    
    Returns:
        dict: GR limit values with keys 'kappa', 'gamma', 'beta'
    """
    return {
        'kappa': 2.0,
        'gamma': 1.0, 
        'beta': 1.0
    }

def solar_system_constraints():
    """
    Return Solar System observational constraints on PPN parameters.
    
    Based on current observations:
    - γ = 1.0 ± 2.3×10⁻⁵ (Cassini spacecraft)
    - β = 1.0 ± 8×10⁻⁵ (lunar laser ranging)
    
    Returns:
        dict: Observational constraints
    """
    return {
        'gamma_measured': 1.0,
        'gamma_uncertainty': 2.3e-5,
        'beta_measured': 1.0, 
        'beta_uncertainty': 8e-5,
        'kappa_implied': 2.0,
        'kappa_uncertainty': 4.6e-5  # 2 * gamma_uncertainty
    }

def validate_solar_system_compliance(kappa):
    """
    Check if a given κ value is consistent with Solar System observations.
    
    Args:
        kappa (float): Model parameter κ to test
        
    Returns:
        dict: Validation results with compliance status and deviations
    """
    gamma, beta = extract_ppn_params(kappa)
    constraints = solar_system_constraints()
    
    gamma_deviation = abs(gamma - constraints['gamma_measured'])
    gamma_compliant = gamma_deviation <= 5 * constraints['gamma_uncertainty']  # 5-sigma
    
    beta_deviation = abs(beta - constraints['beta_measured'])
    beta_compliant = beta_deviation <= 5 * constraints['beta_uncertainty']  # 5-sigma
    
    return {
        'kappa': kappa,
        'gamma': gamma,
        'beta': beta,
        'gamma_deviation': gamma_deviation,
        'beta_deviation': beta_deviation,
        'gamma_compliant': gamma_compliant,
        'beta_compliant': beta_compliant,
        'overall_compliant': gamma_compliant and beta_compliant
    }

def sr_limit(v, c=1.0):
    """Special relativity limit: dτ/dt ≈ 1 - v^2/(2c^2)."""
    return 1 - v**2 / (2 * c**2)
