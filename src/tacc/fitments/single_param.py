"""Single parameter fitment: 1D optimizer for one scalar parameter."""

from typing import Dict, Any, Iterable, Callable
from copy import deepcopy
from math import isfinite
from tacc.core.fitment import Fitment, register
from tacc.core.microlaw import MicrolawInput
from tacc.core.nb import compute_BN


class SingleParam(Fitment):
    """1D optimizer for fitting a single scalar parameter (e.g., kappa or eta)."""
    
    name = "single_param"
    
    def fit(
        self, 
        dataset: Iterable[Dict[str, Any]],
        microlaw_name: str, 
        ml_params: Dict[str, Any],
        bn_family: str, 
        bn_params: Dict[str, Any],
        loss: Callable[[Dict[str, Any], Dict[str, float]], float] = None,
        extra: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Fit a single parameter using 1D optimization.
        
        Args:
            dataset: Iterable of data points
            microlaw_name: Name of microlaw to use
            ml_params: Initial microlaw parameters
            bn_family: Name of B(N) family to use  
            bn_params: Initial B(N) family parameters
            loss: Optional loss function (data_row, prediction) -> float
            extra: Configuration with keys:
                - target_param: which parameter to optimize (default "kappa")
                - init: initial value (default 1.0)
                - max_iter: maximum iterations (default 30)
                - step0: initial step size (default 0.5 * max(1.0, abs(init)))
                - tol: convergence tolerance (default 1e-6)
                
        Returns:
            Dict with 'microlaw' and 'bn' keys containing updated parameters
        """
        cfg = extra or {}
        target_param = cfg.get("target_param", "kappa")
        init = float(cfg.get("init", 1.0))
        max_iter = int(cfg.get("max_iter", 30))
        step0 = float(cfg.get("step0", 0.5 * max(1.0, abs(init))))
        tol = float(cfg.get("tol", 1e-6))
        
        # Convert dataset to list for multiple passes
        data_list = list(dataset)
        
        def score(val: float) -> float:
            """Compute loss for given parameter value."""
            params_ml = dict(ml_params)
            params_bn = dict(bn_params)
            
            # Decide whether this param lives in microlaw or BN
            if target_param in params_ml:
                params_ml[target_param] = val
            else:
                params_bn[target_param] = val
                
            total = 0.0
            for row in data_list:
                try:
                    x = MicrolawInput(**row["x"])
                    pred = compute_BN(microlaw_name, x, params_ml, bn_family, params_bn)
                    
                    if loss is not None:
                        loss_val = loss(row, pred)
                    else:
                        # Default squared error loss using 'B' as prediction
                        loss_val = (pred["B"] - row["target"])**2
                        
                    weight = float(row.get("weight", 1.0))
                    total += loss_val * weight
                    
                except Exception as e:
                    # Return high loss for invalid parameter values
                    return float('inf')
                    
            return float(total)
        
        # Initialize optimization
        k = init
        step = step0
        best_k, best_f = k, score(k)
        
        # Parabolic step optimization
        for iteration in range(max_iter):
            f0 = best_f
            
            try:
                f_plus = score(k + step)
                f_minus = score(k - step)
                
                # Parabolic step calculation
                denom = (f_plus + f_minus - 2*f0)
                if isfinite(denom) and denom != 0.0:
                    k_new = k - 0.5 * step * (f_plus - f_minus) / denom
                else:
                    # Fall back to gradient descent direction
                    k_new = k + (-step if f_plus > f_minus else step)
                    
                f_new = score(k_new)
                
                # Check convergence
                if abs(k_new - k) < tol:
                    k = k_new
                    best_f = f_new
                    best_k = k_new
                    break
                    
                # Update for next iteration
                k = k_new
                best_f = f_new
                best_k = k_new
                step *= 0.7  # Reduce step size
                
            except Exception:
                # If optimization fails, reduce step and continue
                step *= 0.5
                if step < tol:
                    break
        
        # Return updated parameters
        out_ml = dict(ml_params)
        out_bn = dict(bn_params)
        
        if target_param in out_ml:
            out_ml[target_param] = best_k
        else:
            out_bn[target_param] = best_k
            
        return {"microlaw": out_ml, "bn": out_bn}


# Register the fitment
register(SingleParam())
