"""Constitutive law: B_of_N(N, kappa) = exp(-kappa * (1 - N))."""

import math

def B_of_N(N, kappa):
    """Return B(N) = exp(-kappa * (1 - N)). B(1) == 1, monotone in N."""
    return math.exp(-kappa * (1 - N))
