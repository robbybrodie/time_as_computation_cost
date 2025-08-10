"""Constitutive law: B_of_N(N, kappa) = exp(-kappa * (1 - N))."""

import math
import numpy as np

def B_of_N(N, kappa):
    """Return B(N) = exp(-kappa * (1 - N)). B(1) == 1, monotone in N."""
    # Use numpy.exp to handle both scalars and arrays
    return np.exp(-kappa * (1 - N))
