"""Baselines: fit polynomial, exponential, power-law; compute AIC/BIC."""

import numpy as np
from scipy.optimize import curve_fit

def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def exponential(x, a, b):
    return a * np.exp(b * x)

def power_law(x, a, b):
    return a * x**b

def fit_baseline(model_func, x, y):
    """Fit a model to data (least squares)."""
    popt, pcov = curve_fit(model_func, x, y, maxfev=10000)
    return popt, pcov

def compute_aic_bic(y, y_pred, num_params):
    """Compute AIC and BIC for a fit."""
    n = len(y)
    resid = y - y_pred
    sse = np.sum(resid**2)
    aic = n * np.log(sse / n) + 2 * num_params
    bic = n * np.log(sse / n) + num_params * np.log(n)
    return aic, bic
