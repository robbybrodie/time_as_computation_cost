"""Microlaw base classes and registry: invariants → N."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class MicrolawInput:
    """Observer-local (orthonormal tetrad) input quantities for microlaws."""
    
    # Spacetime coordinates
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Metric components in orthonormal tetrad
    g00: float = -1.0  # timelike component
    gxx: float = 1.0   # spatial component
    gyy: float = 1.0   # spatial component
    gzz: float = 1.0   # spatial component
    
    # Additional physical quantities
    phi: float = 0.0   # gravitational potential / c^2
    v: float = 0.0     # velocity / c
    density: float = 0.0  # energy density
    pressure: float = 0.0  # pressure
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            't': self.t, 'x': self.x, 'y': self.y, 'z': self.z,
            'g00': self.g00, 'gxx': self.gxx, 'gyy': self.gyy, 'gzz': self.gzz,
            'phi': self.phi, 'v': self.v, 'density': self.density, 'pressure': self.pressure
        }


class Microlaw(ABC):
    """Base class for microlaws mapping physical invariants to computational capacity N."""
    
    name: str
    
    @abstractmethod
    def compute_N(self, inputs: MicrolawInput, params: Dict[str, Any]) -> float:
        """
        Compute computational capacity N from observer-local inputs.
        
        Args:
            inputs: Observer-local physical quantities
            params: Microlaw-specific parameters
            
        Returns:
            N: Computational capacity (dimensionless, N=1 corresponds to flat spacetime)
        """
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and return cleaned parameters with defaults."""
        return params


class PPNMicrolaw(Microlaw):
    """Post-Newtonian microlaw: N ≈ 1 + Φ/c² for weak fields."""
    
    name = "ppn"
    
    def compute_N(self, inputs: MicrolawInput, params: Dict[str, Any]) -> float:
        """N ≈ 1 + Φ/c² in post-Newtonian limit."""
        return 1.0 + inputs.phi
        
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """PPN microlaw has no parameters."""
        return {}


class SRMicrolaw(Microlaw):
    """Special relativity microlaw: N = 1/γ = √(1 - v²/c²)."""
    
    name = "sr"
    
    def compute_N(self, inputs: MicrolawInput, params: Dict[str, Any]) -> float:
        """N = 1/γ = √(1 - v²/c²) for special relativity."""
        v_over_c = inputs.v
        return np.sqrt(1.0 - v_over_c**2)
        
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """SR microlaw has no parameters."""
        return {}


# Global registry
_MICROLAW_REGISTRY: Dict[str, Microlaw] = {}


def register_microlaw(microlaw: Microlaw) -> None:
    """Register a microlaw in the global registry."""
    _MICROLAW_REGISTRY[microlaw.name] = microlaw


def get_microlaw(name: str) -> Microlaw:
    """Get a microlaw by name from the registry."""
    if name not in _MICROLAW_REGISTRY:
        raise ValueError(f"Unknown microlaw: {name}")
    return _MICROLAW_REGISTRY[name]


def list_microlaws() -> Dict[str, Microlaw]:
    """List all registered microlaws."""
    return dict(_MICROLAW_REGISTRY)


# Register built-in microlaws
register_microlaw(PPNMicrolaw())
register_microlaw(SRMicrolaw())
