"""Metric construction: builds ds^2 = -N^2 c^2 dt^2 + [1/B(N)] dx^2."""

from .constitutive import B_of_N

def g00(N, c=1.0):
    """Return g00 = -N^2 * c^2."""
    return -N**2 * c**2

def gij(N, kappa):
    """Return spatial metric factor: 1/B(N)."""
    return 1.0 / B_of_N(N, kappa)
