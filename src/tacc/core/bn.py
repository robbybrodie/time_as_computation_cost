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


# Register built-in B(N) families
register_bn_family(ExponentialBN())
register_bn_family(PowerBN())
register_bn_family(LinearBN())
