"""
Physical constants for TACC experiments including astrophysical applications.
"""

# Fundamental constants
c = 299_792_458             # m/s - Speed of light
G = 6.67430e-11            # m^3/kg/s^2 - Gravitational constant
hbar = 1.054571817e-34     # J⋅s - Reduced Planck constant
k_B = 1.380649e-23         # J/K - Boltzmann constant
h = 6.62607015e-34         # J⋅s - Planck constant

# Solar system parameters
GM_SUN = 1.32712440018e20   # m^3/s^2 - Standard gravitational parameter of the Sun
R_SUN = 6.957e8             # m - Solar radius
M_SUN = 1.98847e30         # kg - Solar mass

# Conversion factors
ARCSEC_PER_RAD = 206_265    # Arcseconds per radian

# Other useful constants
AU = 1.495978707e11         # m - Astronomical unit
SCHWARZSCHILD_RADIUS_SUN = 2 * GM_SUN / (c * c)  # m - Schwarzschild radius of the Sun (~2953 m)

# Mercury orbital parameters (for precession tests)
MERCURY_SEMI_MAJOR_AXIS_AU = 0.387  # AU
MERCURY_ECCENTRICITY = 0.206
MERCURY_ORBITAL_PERIOD_DAYS = 88    # days

# Observational reference values
OBSERVED_MERCURY_PRECESSION = 43.1  # arcsec/century - observed anomalous precession
SOLAR_LIMB_DEFLECTION_GR = 1.75     # arcsec - Einstein's prediction for γ=1
