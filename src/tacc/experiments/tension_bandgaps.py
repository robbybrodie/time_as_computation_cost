"""
Tension Bandgaps Experiment - Fixed Version
No data leakage: frozen generator + proper train/val/test split + AICc/BIC + leakage guard
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import math
import json
import time
import subprocess
import pathlib


@dataclass(frozen=True)
class TBGenParams:
    n_points: int = 50
    noise_sigma: float = 0.05
    a_true: float = 2.0
    beta_true: float = 1.5
    x_min: float = 0.0
    x_max: float = 1.0
    seed: int = 42


def generate_tb_dataset(p: TBGenParams) -> Dict[str, Any]:
    """Generate frozen synthetic dataset - NEVER reads fitment parameters."""
    rng = np.random.default_rng(p.seed)
    x = np.linspace(p.x_min, p.x_max, p.n_points)
    # Baseline synthetic relation (keep as-is unless you intentionally change physics)
    y_clean = p.a_true * np.exp(p.beta_true * x)
    y = y_clean + rng.normal(0.0, p.noise_sigma, size=x.shape)
    return {
        "x": x, "y": y, "y_clean": y_clean,
        "gen_params": p.__dict__.copy(),
    }


def split_tvt(x, y, seed=0, ratios=(0.6, 0.2, 0.2)):
    """Train/validation/test split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(x.size)
    rng.shuffle(idx)
    n = x.size
    n_tr = int(ratios[0]*n)
    n_va = int(ratios[1]*n)
    i_tr, i_va, i_te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    return (x[i_tr], y[i_tr]), (x[i_va], y[i_va]), (x[i_te], y[i_te])


def fit_exponential(x, y):
    """Fit y ~ a * exp(b * x) via log transform with noise handling."""
    # Use only y>0 for log; otherwise use robust alt (e.g., clipped floor)
    mask = y > 0
    if np.sum(mask) < 3:
        return {"family": "exponential", "a": np.nan, "b": np.nan, "invalid": True}
    
    x_, ly = x[mask], np.log(y[mask])
    try:
        b, a0 = np.polyfit(x_, ly, 1)  # ly ≈ b*x + ln a
        a = math.exp(a0)
        return {"family": "exponential", "a": a, "b": b}
    except Exception:
        return {"family": "exponential", "a": np.nan, "b": np.nan, "invalid": True}


def fit_polynomial(x, y, degree=3):
    """Fit y ~ sum c_k x^k (keep degree small)."""
    try:
        coefs = np.polyfit(x, y, degree)
        return {"family": "polynomial", "coefs": coefs, "degree": degree}
    except Exception:
        return {"family": "polynomial", "coefs": [np.nan]*(degree+1), "degree": degree, "invalid": True}


def fit_powerlaw(x, y):
    """Proper power law: y ~ A * x^B => log y = log A + B log x."""
    # Restrict to x>0, y>0
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 3:
        return {"family": "power_law", "A": np.nan, "B": np.nan, "invalid": True}
    
    lx, ly = np.log(x[mask]), np.log(y[mask])
    try:
        B, logA = np.polyfit(lx, ly, 1)
        A = math.exp(logA)
        return {"family": "power_law", "A": A, "B": B}
    except Exception:
        return {"family": "power_law", "A": np.nan, "B": np.nan, "invalid": True}


def predict(params, x):
    """Make predictions using fitted model parameters."""
    fam = params["family"]
    if fam == "exponential":
        if params.get("invalid"):
            return np.full_like(x, np.nan)
        return params["a"] * np.exp(params["b"] * x)
    elif fam == "polynomial":
        if params.get("invalid"):
            return np.full_like(x, np.nan)
        coefs = params["coefs"]
        return np.polyval(coefs, x)
    elif fam == "power_law":
        if params.get("invalid"):
            return np.full_like(x, np.nan)
        A, B = params.get("A"), params.get("B")
        # Handle x <= 0 gracefully
        result = np.full_like(x, np.nan)
        mask = x > 0
        result[mask] = A * np.power(x[mask], B)
        return result
    else:
        raise ValueError(f"unknown family {fam}")


def mse(y_true, y_pred):
    """Mean squared error."""
    r = y_true - y_pred
    return float(np.mean(r*r))


def ic_counts(params):
    """Get number of parameters for each model family."""
    fam = params["family"]
    if fam == "exponential":
        k = 2
    elif fam == "power_law":
        k = 2
    elif fam == "polynomial":
        k = int(params["degree"]) + 1
    else:
        raise ValueError(f"unknown family {fam}")
    return k


def aic_bic_aicc(y, yhat, k):
    """Compute AIC, BIC, AICc for given predictions."""
    n = len(y)
    
    # Handle NaN predictions
    valid_mask = ~(np.isnan(y) | np.isnan(yhat))
    if np.sum(valid_mask) < k + 1:
        return float("inf"), float("inf"), float("inf")
    
    y_valid = y[valid_mask]
    yhat_valid = yhat[valid_mask]
    n_valid = len(y_valid)
    
    resid = y_valid - yhat_valid
    sse = float(np.dot(resid, resid))
    
    if sse <= 0 or n_valid <= k:
        return float("inf"), float("inf"), float("inf")
    
    sigma2 = sse / n_valid
    # Gaussian log-likelihood (up to additive const)
    ll = -0.5*n_valid*(1 + math.log(2*math.pi*sigma2))
    aic = 2*k - 2*ll
    bic = k*math.log(n_valid) - 2*ll
    # AICc small-sample correction
    aicc = aic + (2*k*(k+1))/(n_valid - k - 1) if n_valid - k - 1 > 0 else float("inf")
    return aic, bic, aicc


def apply_fitment(dataset, family, fitment_name="no_fit", fitment_params=None):
    """Apply fitment hook that only updates model hyper-params (no leakage)."""
    fitment_params = fitment_params or {}
    if fitment_name == "no_fit":
        return {}
    elif fitment_name == "single_param":
        # Example: choose polynomial degree; DO NOT touch dataset
        return {"degree": int(fitment_params.get("degree", 3))}
    elif fitment_name == "linear_fit":
        # Keep as no-op to avoid leakage
        return {}
    else:
        return {}


def leakage_guard(gen_params_before, gen_params_after, y_train, yhat_train, noise_sigma):
    """Detect data leakage."""
    # 1) Truth changed?
    if gen_params_before != gen_params_after:
        raise RuntimeError("LEAKAGE: generator parameters changed after fitment.")
    
    # 2) Too-perfect fit with noise? (tune threshold as needed)
    if noise_sigma > 0:
        valid_mask = ~(np.isnan(y_train) | np.isnan(yhat_train))
        if np.sum(valid_mask) > 0:
            err = mse(y_train[valid_mask], yhat_train[valid_mask])
            # Expected MSE ~ noise_sigma^2, so if << 1% of that, warn
            if err < 0.01 * (noise_sigma**2):
                raise RuntimeError("LEAKAGE: near-zero train error despite noise; check coupling.")


def run_tension_bandgaps(gen_p: TBGenParams, 
                        families=("exponential", "polynomial", "power_law"),
                        fitment=("no_fit", {}), 
                        result_dir="experiments/results") -> Dict[str, Any]:
    """
    Run tension bandgaps experiment with proper ML practices.
    
    Args:
        gen_p: Frozen generator parameters
        families: Model families to fit
        fitment: (fitment_name, fitment_params) tuple
        result_dir: Directory to save results
        
    Returns:
        Dict with full experimental results and stamp
    """
    # Generate frozen dataset
    data = generate_tb_dataset(gen_p)
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = split_tvt(
        data["x"], data["y"], seed=gen_p.seed
    )
    
    fitment_name, fitment_params = fitment
    results = []
    gen_params_before = dict(data["gen_params"])

    for fam in families:
        opts = apply_fitment(data, fam, fitment_name, fitment_params)
        
        # Fit on train
        if fam == "exponential":
            params = fit_exponential(x_tr, y_tr)
        elif fam == "polynomial":
            params = fit_polynomial(x_tr, y_tr, degree=opts.get("degree", 3))
        elif fam == "power_law":
            params = fit_powerlaw(x_tr, y_tr)
        else:
            continue
            
        if params.get("invalid"):
            results.append({"family": fam, "invalid": True})
            continue

        # Leakage guard on train predictions
        yhat_tr = predict(params, x_tr)
        leakage_guard(gen_params_before, data["gen_params"], y_tr, yhat_tr, gen_p.noise_sigma)

        # Metrics on val/test
        yhat_va = predict(params, x_va)
        yhat_te = predict(params, x_te)
        k = ic_counts(params)
        aic, bic, aicc = aic_bic_aicc(y_va, yhat_va, k)
        
        res = {
            "family": fam,
            "params": {k_: (v.tolist() if hasattr(v, "tolist") else v) 
                      for k_, v in params.items() if k_ != "invalid"},
            "val": {"mse": mse(y_va, yhat_va), "AIC": aic, "BIC": bic, "AICc": aicc},
            "test": {"mse": mse(y_te, yhat_te)},
        }
        results.append(res)

    # Rank by AICc
    valid_results = [r for r in results if not r.get("invalid")]
    ranked = sorted(valid_results, key=lambda r: r["val"]["AICc"])
    best = ranked[0] if ranked else None

    # Results stamp
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
        
    stamp = {
        "timestamp": int(time.time()),
        "commit": commit,
        "generator": data["gen_params"],
        "fitment": {"name": fitment_name, "params": fitment_params},
        "families": results,
        "best_by_AICc": best["family"] if best else None,
        "train_size": len(x_tr),
        "val_size": len(x_va),
        "test_size": len(x_te),
    }
    
    # Save stamp
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
    outp = pathlib.Path(result_dir) / f"tension_bandgaps_seed{gen_p.seed}.json"
    outp.write_text(json.dumps(stamp, indent=2))
    
    return stamp
