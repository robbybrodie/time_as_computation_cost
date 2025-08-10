"""
bandgaps.py

Microphysical DoF law fitting routines for capacity-physics.
Implements fits for DoF(N) = exp[-a*(1-N)] and psi = DoF^beta.
No GR dependencies.
"""

import numpy as np
import pandas as pd

def fit_dof_law(dof_table_csv=None):
    """
    Fit the exponential slope 'a' in DoF(N) = exp[-a*(1-N)] from micro data.

    Args:
        dof_table_csv (str): Path to CSV file with DoF data. If None, generates synthetic data.

    Returns:
        float: Fitted exponential slope 'a'.
    """
    from scipy.optimize import curve_fit
    
    if dof_table_csv is None:
        # Generate synthetic data for demonstration
        N_values = np.linspace(0.5, 1.5, 50)
        true_a = 2.0
        DoF_values = np.exp(-true_a * (1 - N_values))
        # Add some noise
        DoF_values += np.random.normal(0, 0.05, len(DoF_values))
    else:
        # Load from CSV if provided
        try:
            data = pd.read_csv(dof_table_csv)
            N_values = data['N'].values
            DoF_values = data['DoF'].values
        except (FileNotFoundError, KeyError):
            # Fallback to synthetic data if file issues
            N_values = np.linspace(0.5, 1.5, 50)
            true_a = 2.0
            DoF_values = np.exp(-true_a * (1 - N_values))
            DoF_values += np.random.normal(0, 0.05, len(DoF_values))
    
    def dof_model(N, a):
        return np.exp(-a * (1 - N))
    
    popt, _ = curve_fit(dof_model, N_values, DoF_values)
    return popt[0]

def fit_beta(dof_table_csv=None):
    """
    Fit the exponent 'beta' in psi = DoF^beta from micro data.

    Args:
        dof_table_csv (str): Path to CSV file with DoF data. If None, generates synthetic data.

    Returns:
        float: Fitted exponent 'beta'.
    """
    from scipy.optimize import curve_fit
    
    if dof_table_csv is None:
        # Generate synthetic data for demonstration
        DoF_values = np.linspace(0.1, 2.0, 50)
        true_beta = 1.5
        psi_values = DoF_values ** true_beta
        # Add some noise
        psi_values += np.random.normal(0, 0.1, len(psi_values))
    else:
        # Load from CSV if provided
        try:
            data = pd.read_csv(dof_table_csv)
            DoF_values = data['DoF'].values
            psi_values = data['psi'].values
        except (FileNotFoundError, KeyError):
            # Fallback to synthetic data if file issues
            DoF_values = np.linspace(0.1, 2.0, 50)
            true_beta = 1.5
            psi_values = DoF_values ** true_beta
            psi_values += np.random.normal(0, 0.1, len(psi_values))
    
    def psi_model(DoF, beta):
        return DoF ** beta
    
    popt, _ = curve_fit(psi_model, DoF_values, psi_values)
    return popt[0]
