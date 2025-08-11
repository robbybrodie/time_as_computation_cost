"""B(N) families: response functions mapping N → geometry factor."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BNFamily(ABC):
    """Base class for B(N) families mapping computational capacity to geometry factors."""
    
    name: str
    
    @abstractmethod
    def compute_B(self, N: float, params: Dict[str, Any]) -> float:
        """
        Compute geometry factor B from computational capacity N.
        
        Args:
            N: Computational capacity (dimensionless)
            params: Family-specific parameters
            
        Returns:
            B: Geometry factor (dimensionless, B=1 corresponds to N=1)
        """
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and return cleaned parameters with defaults."""
        return params


class ExponentialBN(BNFamily):
    """Exponential B(N) family: B(N) = exp(-κ(1-N))."""
    
    name = "exponential"
    
    def compute_B(self, N: float, params: Dict[str, Any]) -> float:
        """B(N) = exp(-κ(1-N)) with parameter κ (kappa)."""
        kappa = params.get("kappa", 1.0)
        return np.exp(-kappa * (1 - N))
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate kappa parameter."""
        kappa = float(params.get("kappa", 1.0))
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        return {"kappa": kappa}


class PowerBN(BNFamily):
    """Power law B(N) family: B(N) = N^α."""
    
    name = "power"
    
    def compute_B(self, N: float, params: Dict[str, Any]) -> float:
        """B(N) = N^α with parameter α (alpha)."""
        alpha = params.get("alpha", 1.0)
        return N**alpha
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate alpha parameter."""
        alpha = float(params.get("alpha", 1.0))
        return {"alpha": alpha}


class LinearBN(BNFamily):
    """Linear B(N) family: B(N) = a*N + b."""
    
    name = "linear"
    
    def compute_B(self, N: float, params: Dict[str, Any]) -> float:
        """B(N) = a*N + b with parameters a and b."""
        a = params.get("a", 1.0)
        b = params.get("b", 0.0)
        return a * N + b
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a and b parameters."""
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        return {"a": a, "b": b}


class SigmoidBN(BNFamily):
    """Sigmoid B(N) family: B(N) = A / (1 + exp(-k*(N - N0))) + B0."""
    
    name = "sigmoid"
    
    def compute_B(self, N: float, params: Dict[str, Any]) -> float:
        """B(N) = A / (1 + exp(-k*(N - N0))) + B0 with sigmoid parameters."""
        A = params.get("A", 1.0)      # Amplitude
        k = params.get("k", 10.0)     # Steepness 
        N0 = params.get("N0", 0.5)    # Midpoint
        B0 = params.get("B0", 0.0)    # Baseline offset
        
        return A / (1 + np.exp(-k * (N - N0))) + B0
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sigmoid parameters."""
        A = float(params.get("A", 1.0))
        k = float(params.get("k", 10.0))
        N0 = float(params.get("N0", 0.5))
        B0 = float(params.get("B0", 0.0))
        
        if A <= 0:
            raise ValueError("Sigmoid amplitude A must be positive")
        if k <= 0:
            raise ValueError("Sigmoid steepness k must be positive")
        if not (0 < N0 < 1):
            raise ValueError("Sigmoid midpoint N0 must be in (0, 1)")
            
        return {"A": A, "k": k, "N0": N0, "B0": B0}


class PiecewiseBN(BNFamily):
    """Piecewise linear B(N) family with configurable breakpoints."""
    
    name = "piecewise"
    
    def compute_B(self, N: float, params: Dict[str, Any]) -> float:
        """B(N) as piecewise linear function."""
        breakpoints = params.get("breakpoints", [(0.0, 0.0), (1.0, 1.0)])
        
        # Sort breakpoints by N value
        breakpoints = sorted(breakpoints, key=lambda x: x[0])
        
        # Find the appropriate segment
        for i in range(len(breakpoints) - 1):
            N1, B1 = breakpoints[i]
            N2, B2 = breakpoints[i + 1]
            
            if N1 <= N <= N2:
                # Linear interpolation
                if N2 == N1:  # Avoid division by zero
                    return B1
                return B1 + (B2 - B1) * (N - N1) / (N2 - N1)
        
        # Extrapolation beyond range
        if N < breakpoints[0][0]:
            return breakpoints[0][1]
        else:
            return breakpoints[-1][1]
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate piecewise parameters."""
        breakpoints = params.get("breakpoints", [(0.0, 0.0), (1.0, 1.0)])
        
        if not isinstance(breakpoints, list) or len(breakpoints) < 2:
            raise ValueError("Piecewise breakpoints must be a list with at least 2 points")
        
        # Validate each breakpoint
        validated_points = []
        for point in breakpoints:
            if not (isinstance(point, (list, tuple)) and len(point) == 2):
                raise ValueError("Each breakpoint must be a (N, B) pair")
            N, B = float(point[0]), float(point[1])
            validated_points.append((N, B))
        
        # Sort by N values
        validated_points = sorted(validated_points, key=lambda x: x[0])
        
        return {"breakpoints": validated_points}


class PolynomialBN(BNFamily):
    """Polynomial B(N) family: B(N) = sum(c_i * N^i)."""
    
    name = "polynomial"
    
    def compute_B(self, N: float, params: Dict[str, Any]) -> float:
        """B(N) as polynomial with coefficients."""
        coefficients = params.get("coefficients", [0.0, 1.0])  # Default: B(N) = N
        
        result = 0.0
        for i, coeff in enumerate(coefficients):
            result += coeff * (N ** i)
        
        return result
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate polynomial coefficients."""
        coefficients = params.get("coefficients", [0.0, 1.0])
        
        if not isinstance(coefficients, list) or len(coefficients) == 0:
            raise ValueError("Polynomial coefficients must be a non-empty list")
        
        # Convert to floats
        validated_coeffs = [float(c) for c in coefficients]
        
        return {"coefficients": validated_coeffs}


class LogarithmicBN(BNFamily):
    """Logarithmic B(N) family: B(N) = a * log(b*N + c) + d."""
    
    name = "logarithmic"
    
    def compute_B(self, N: float, params: Dict[str, Any]) -> float:
        """B(N) = a * log(b*N + c) + d with logarithmic parameters."""
        a = params.get("a", 1.0)
        b = params.get("b", 1.0) 
        c = params.get("c", 0.0)
        d = params.get("d", 0.0)
        
        argument = b * N + c
        if argument <= 0:
            # Handle domain issues gracefully
            return d  # Return offset when log is undefined
        
        return a * np.log(argument) + d
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logarithmic parameters."""
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 1.0))
        c = float(params.get("c", 0.0))
        d = float(params.get("d", 0.0))
        
        # Check that argument will be positive for N in (0,1]
        if b <= 0 and c <= 0:
            raise ValueError("Logarithmic parameters must ensure b*N + c > 0 for N in (0,1]")
        
        return {"a": a, "b": b, "c": c, "d": d}


class CustomBN(BNFamily):
    """Custom user-defined B(N) family with function string evaluation."""
    
    name = "custom"
    
    def compute_B(self, N: float, params: Dict[str, Any]) -> float:
        """B(N) using custom function string."""
        func_str = params.get("function", "N")  # Default: B(N) = N
        func_params = params.get("func_params", {})
        
        # Safe namespace for evaluation
        safe_namespace = {
            "N": N,
            "np": np,
            "exp": np.exp,
            "log": np.log,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "sqrt": np.sqrt,
            "abs": np.abs,
            **func_params  # User-defined parameters
        }
        
        try:
            result = eval(func_str, {"__builtins__": {}}, safe_namespace)
            return float(result)
        except Exception as e:
            raise ValueError(f"Custom function evaluation failed: {e}")
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate custom function parameters."""
        func_str = params.get("function", "N")
        func_params = params.get("func_params", {})
        
        if not isinstance(func_str, str):
            raise ValueError("Custom function must be a string")
        
        if not isinstance(func_params, dict):
            raise ValueError("Function parameters must be a dictionary")
        
        # Test evaluation at a safe point
        try:
            safe_namespace = {
                "N": 0.5,
                "np": np,
                "exp": np.exp,
                "log": np.log,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "sqrt": np.sqrt,
                "abs": np.abs,
                **func_params
            }
            eval(func_str, {"__builtins__": {}}, safe_namespace)
        except Exception as e:
            raise ValueError(f"Custom function validation failed: {e}")
        
        return {"function": func_str, "func_params": func_params}


# Global registry
_BN_REGISTRY: Dict[str, BNFamily] = {}


def register_bn_family(family: BNFamily) -> None:
    """Register a B(N) family in the global registry."""
    _BN_REGISTRY[family.name] = family


def get_bn_family(name: str) -> BNFamily:
    """Get a B(N) family by name from the registry."""
    if name not in _BN_REGISTRY:
        raise ValueError(f"Unknown B(N) family: {name}")
    return _BN_REGISTRY[name]


def list_bn_families() -> Dict[str, BNFamily]:
    """List all registered B(N) families."""
    return dict(_BN_REGISTRY)


def get_bn_family_info() -> Dict[str, Dict[str, Any]]:
    """Get detailed information about all B(N) families."""
    info = {}
    for name, family in _BN_REGISTRY.items():
        info[name] = {
            "name": family.name,
            "class": family.__class__.__name__,
            "docstring": family.__doc__ or "No description available"
        }
    return info


# Register built-in B(N) families
register_bn_family(ExponentialBN())
register_bn_family(PowerBN())
register_bn_family(LinearBN())
register_bn_family(SigmoidBN())
register_bn_family(PiecewiseBN())
register_bn_family(PolynomialBN())
register_bn_family(LogarithmicBN())
register_bn_family(CustomBN())
