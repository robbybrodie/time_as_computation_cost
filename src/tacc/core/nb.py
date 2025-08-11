"""Notebook bridge: unified interface for microlaws, B(N) families, and fitments."""

from __future__ import annotations
from typing import Dict, Any
from .microlaw import MicrolawInput, get_microlaw, list_microlaws
from .bn import get_bn_family, list_bn_families
from . import fitment as fitment_mod


def compute_BN(
    microlaw_name: str,
    inputs: MicrolawInput,
    ml_params: Dict[str, Any],
    bn_family: str,
    bn_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Unified computation: microlaw → N → B(N).
    
    Args:
        microlaw_name: Name of microlaw to use
        inputs: Observer-local physical quantities
        ml_params: Microlaw parameters
        bn_family: Name of B(N) family to use
        bn_params: B(N) family parameters
        
    Returns:
        Dict containing:
        - 'N': computational capacity
        - 'B': geometry factor
        - 'inputs': copy of inputs
        - 'microlaw': microlaw name and params
        - 'bn_family': B(N) family name and params
    """
    # Get microlaw and compute N
    microlaw = get_microlaw(microlaw_name)
    validated_ml_params = microlaw.validate_params(ml_params)
    N = microlaw.compute_N(inputs, validated_ml_params)
    
    # Get B(N) family and compute B
    bn_family_obj = get_bn_family(bn_family)
    validated_bn_params = bn_family_obj.validate_params(bn_params)
    B = bn_family_obj.compute_B(N, validated_bn_params)
    
    return {
        'N': N,
        'B': B,
        'inputs': inputs.to_dict(),
        'microlaw': {'name': microlaw_name, 'params': validated_ml_params},
        'bn_family': {'name': bn_family, 'params': validated_bn_params}
    }


def list_fitments() -> Dict[str, Any]:
    """List all available fitments."""
    return {k: v for k, v in fitment_mod.list_all().items()}


def get_fitment(name: str):
    """Get a fitment by name."""
    return fitment_mod.get(name)


def list_components() -> Dict[str, Dict[str, Any]]:
    """List all available components (microlaws, B(N) families, fitments)."""
    return {
        'microlaws': list_microlaws(),
        'bn_families': list_bn_families(),
        'fitments': list_fitments()
    }


# Convenience functions for common use cases
def create_inputs(**kwargs) -> MicrolawInput:
    """Create MicrolawInput from keyword arguments."""
    return MicrolawInput(**kwargs)


def quick_ppn_exponential(phi: float = 0.0, kappa: float = 2.0) -> Dict[str, Any]:
    """Quick computation using PPN microlaw and exponential B(N) family."""
    inputs = MicrolawInput(phi=phi)
    return compute_BN(
        microlaw_name="ppn",
        inputs=inputs,
        ml_params={},
        bn_family="exponential",
        bn_params={"kappa": kappa}
    )


def quick_sr_power(v: float = 0.0, alpha: float = 1.0) -> Dict[str, Any]:
    """Quick computation using SR microlaw and power B(N) family."""
    inputs = MicrolawInput(v=v)
    return compute_BN(
        microlaw_name="sr",
        inputs=inputs,
        ml_params={},
        bn_family="power",
        bn_params={"alpha": alpha}
    )
