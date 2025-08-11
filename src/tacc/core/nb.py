"""Notebook bridge: unified interface for microlaws, B(N) families, and fitments."""

from __future__ import annotations
from typing import Dict, Any, Optional
from .microlaw import MicrolawInput, get_microlaw, list_microlaws
from .bn import get_bn_family, list_bn_families
from . import fitment as fitment_mod
from .config import get_config
from .validation import validate_complete_setup
from .widgets import create_component_chooser, create_sanity_check_cell, display_system_info


def compute_BN(
    microlaw_name: Optional[str] = None,
    inputs: Optional[MicrolawInput] = None,
    ml_params: Optional[Dict[str, Any]] = None,
    bn_family: Optional[str] = None,
    bn_params: Optional[Dict[str, Any]] = None,
    use_config: bool = True
) -> Dict[str, Any]:
    """
    Unified computation: microlaw → N → B(N).
    
    Args:
        microlaw_name: Name of microlaw to use (defaults to config)
        inputs: Observer-local physical quantities (required)
        ml_params: Microlaw parameters (defaults to config)
        bn_family: Name of B(N) family to use (defaults to config)
        bn_params: B(N) family parameters (defaults to config)
        use_config: Whether to use config defaults for missing parameters
        
    Returns:
        Dict containing:
        - 'N': computational capacity
        - 'B': geometry factor
        - 'inputs': copy of inputs
        - 'microlaw': microlaw name and params
        - 'bn_family': B(N) family name and params
        - 'validation': validation results (if enabled)
    """
    # Load config defaults if needed
    if use_config:
        config = get_config()
        microlaw_name = microlaw_name or config.microlaw
        ml_params = ml_params if ml_params is not None else config.microlaw_params
        bn_family = bn_family or config.bn_family
        bn_params = bn_params if bn_params is not None else config.bn_params
    
    # Require inputs
    if inputs is None:
        raise ValueError("inputs parameter is required")
    
    # Get microlaw and compute N
    microlaw = get_microlaw(microlaw_name)
    validated_ml_params = microlaw.validate_params(ml_params)
    N = microlaw.compute_N(inputs, validated_ml_params)
    
    # Get B(N) family and compute B
    bn_family_obj = get_bn_family(bn_family)
    validated_bn_params = bn_family_obj.validate_params(bn_params)
    B = bn_family_obj.compute_B(N, validated_bn_params)
    
    result = {
        'N': N,
        'B': B,
        'inputs': inputs.to_dict(),
        'microlaw': {'name': microlaw_name, 'params': validated_ml_params},
        'bn_family': {'name': bn_family, 'params': validated_bn_params}
    }
    
    # Add validation if enabled
    if use_config and get_config().enable_validation:
        try:
            validation_result = validate_complete_setup(
                microlaw, inputs, validated_ml_params,
                bn_family_obj, validated_bn_params
            )
            result['validation'] = validation_result
        except Exception as e:
            result['validation'] = {'error': str(e)}
    
    return result


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
