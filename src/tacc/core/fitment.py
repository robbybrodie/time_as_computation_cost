"""Fitment base classes and registry: calibration framework for microlaws and B(N) families."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Iterable


class Fitment(ABC):
    """Base class for fitments providing calibration for microlaws and B(N) families."""
    
    name: str
    
    @abstractmethod
    def fit(
        self,
        dataset: Iterable[Dict[str, Any]],
        microlaw_name: str,
        ml_params: Dict[str, Any],
        bn_family: str,
        bn_params: Dict[str, Any],
        loss: Any = None,
        extra: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Fit parameters to dataset using specified microlaw and B(N) family.
        
        Args:
            dataset: Iterable of data points, each containing:
                - 'x': kwargs to build MicrolawInput
                - 'target': experiment-specific target(s)
                - optional 'weight': data point weight (default 1.0)
            microlaw_name: Name of microlaw to use
            ml_params: Initial microlaw parameters
            bn_family: Name of B(N) family to use
            bn_params: Initial B(N) family parameters
            loss: Optional loss function (dataset_row, prediction) -> float
            extra: Optional extra configuration parameters
            
        Returns:
            Dict with keys 'microlaw' and 'bn' containing updated parameter maps
        """
        pass


# Global registry
_REGISTRY: Dict[str, Fitment] = {}


def register(f: Fitment) -> None:
    """Register a fitment in the global registry."""
    _REGISTRY[f.name] = f


def get(name: str) -> Fitment:
    """Get a fitment by name from the registry."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown fitment: {name}")
    return _REGISTRY[name]


def list_all() -> Dict[str, Fitment]:
    """List all registered fitments."""
    return dict(_REGISTRY)
