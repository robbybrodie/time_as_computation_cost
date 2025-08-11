"""
Linear gradient fitment - fits parameters using simple gradient descent.
"""

from typing import Dict, Any, Iterable, Callable
from copy import deepcopy
from tacc.core.fitment import Fitment, register
from tacc.core.microlaw import MicrolawInput
from tacc.core.nb import compute_BN

class LinearFit(Fitment):
    """Simple linear gradient descent fitment for single parameter optimization."""
    
    name = "linear_fit"

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
        Linear gradient descent fitment - simpler than single_param's parabolic method.
        """
        cfg = extra or {}
        target_param = cfg.get("target_param", "kappa")
        init = float(cfg.get("init", 1.0))
        max_iter = int(cfg.get("max_iter", 30))
        learning_rate = float(cfg.get("learning_rate", 0.01))
        epsilon = float(cfg.get("epsilon", 1e-4))  # for numerical gradient

        def score(val: float) -> float:
            params_ml = dict(ml_params)
            params_bn = dict(bn_params)
            
            # Update the target parameter
            if target_param in params_ml:
                params_ml[target_param] = val
            else:
                params_bn[target_param] = val
            
            total = 0.0
            for row in dataset:
                x = MicrolawInput(**row["x"])
                pred = compute_BN(microlaw_name, x, params_ml, bn_family, params_bn)
                if loss:
                    total += loss(row, pred) * float(row.get("weight", 1.0))
                else:
                    total += (pred["B"] - row["target"])**2 * float(row.get("weight", 1.0))
            return float(total)

        # Simple gradient descent
        param_value = init
        
        for i in range(max_iter):
            # Numerical gradient
            loss_plus = score(param_value + epsilon)
            loss_minus = score(param_value - epsilon)
            gradient = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Update parameter
            old_param = param_value
            param_value = param_value - learning_rate * gradient
            
            # Check convergence
            if abs(param_value - old_param) < 1e-6:
                break
        
        # Return updated parameters
        out_ml = dict(ml_params)
        out_bn = dict(bn_params)
        
        if target_param in out_ml:
            out_ml[target_param] = param_value
        else:
            out_bn[target_param] = param_value
            
        return {"microlaw": out_ml, "bn": out_bn}

# Register the fitment
register(LinearFit())
