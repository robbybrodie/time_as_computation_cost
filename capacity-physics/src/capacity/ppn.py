"""PPN expansion: series near N=1, extract gamma, beta, test SR limit."""

import numpy as np

def expand_N_to_Phi(N):
    """Map N to Newtonian potential Phi/c^2 near N=1."""
    # N ≈ 1 + Phi/c^2
    return N - 1

def extract_ppn_params(kappa):
    """
    Extract PPN parameters gamma, beta from metric expansion.
    For spatial factor 1/B(N), gamma = kappa / 2.
    For N^2 = 1 + 2*Phi/c^2 + 2*beta*(Phi/c^2)^2, beta = 1.
    """
    gamma = kappa / 2
    beta = 1.0
    return gamma, beta

def sr_limit(v, c=1.0):
    """Special relativity limit: dτ/dt ≈ 1 - v^2/(2c^2)."""
    return 1 - v**2 / (2 * c**2)
