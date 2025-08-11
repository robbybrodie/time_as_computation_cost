"""
Physics validation and guardrails for TACC.

Ensures that microlaws and B(N) families satisfy physical constraints:
- 0 < N ≤ 1 (computational capacity bounds)
- B(N) > 0 (positive geometry factors)
- Monotonicity where expected
- Parameter range validation
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from abc import ABC, abstractmethod
import warnings

from .microlaw import Microlaw, MicrolawInput
from .bn import BNFamily


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class ValidationWarning(UserWarning):
    """Warning for potential physics issues."""
    pass


@abstractmethod
class Validator(ABC):
    """Base class for validation rules."""
    
    @abstractmethod
    def validate(self, *args, **kwargs) -> Tuple[bool, str]:
        """
        Validate the given inputs.
        
        Returns:
            (is_valid, message): Tuple of validation result and message
        """
        pass


class MicrolawValidator(Validator):
    """Validates microlaw outputs and constraints."""
    
    def validate(self, 
                microlaw: Microlaw, 
                inputs: MicrolawInput, 
                params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate microlaw computation."""
        try:
            N = microlaw.compute_N(inputs, params)
        except Exception as e:
            return False, f"Microlaw computation failed: {e}"
        
        # Check N bounds: 0 < N ≤ 1
        if not isinstance(N, (int, float)):
            return False, f"N must be numeric, got {type(N)}"
        
        if np.isnan(N) or np.isinf(N):
            return False, f"N is not finite: {N}"
            
        if N <= 0:
            return False, f"N must be positive: N = {N}"
            
        if N > 1:
            # This might be acceptable in some extended theories
            warnings.warn(f"N > 1 detected: N = {N}. This exceeds flat spacetime capacity.", 
                         ValidationWarning)
        
        return True, "Valid"


class BNFamilyValidator(Validator):
    """Validates B(N) family outputs and properties."""
    
    def validate(self, 
                bn_family: BNFamily, 
                params: Dict[str, Any],
                N_test_points: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """Validate B(N) family computation and properties."""
        
        if N_test_points is None:
            N_test_points = np.linspace(0.01, 1.0, 50)
        
        try:
            B_values = np.array([bn_family.compute_B(N, params) for N in N_test_points])
        except Exception as e:
            return False, f"B(N) computation failed: {e}"
        
        # Check B values are finite and positive
        if not np.all(np.isfinite(B_values)):
            return False, f"B(N) contains non-finite values"
            
        if not np.all(B_values > 0):
            negative_count = np.sum(B_values <= 0)
            return False, f"B(N) must be positive, found {negative_count} non-positive values"
        
        # Check for reasonable range (heuristic)
        if np.max(B_values) > 1e6:
            warnings.warn(f"B(N) values very large (max: {np.max(B_values):.2e}). Check parameters.", 
                         ValidationWarning)
        
        if np.min(B_values) < 1e-6:
            warnings.warn(f"B(N) values very small (min: {np.min(B_values):.2e}). Check parameters.", 
                         ValidationWarning)
        
        return True, "Valid"


class MonotonicityValidator(Validator):
    """Validates monotonicity properties of B(N) families."""
    
    def validate(self, 
                bn_family: BNFamily, 
                params: Dict[str, Any],
                N_test_points: Optional[np.ndarray] = None,
                expected_monotonic: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate monotonicity of B(N).
        
        Args:
            expected_monotonic: "increasing", "decreasing", or None for no check
        """
        
        if N_test_points is None:
            N_test_points = np.linspace(0.01, 1.0, 100)
        
        try:
            B_values = np.array([bn_family.compute_B(N, params) for N in N_test_points])
        except Exception as e:
            return False, f"B(N) computation failed: {e}"
        
        # Compute differences
        dB_dN = np.diff(B_values) / np.diff(N_test_points)
        
        if expected_monotonic == "increasing":
            if not np.all(dB_dN >= -1e-10):  # Small tolerance for numerical errors
                violations = np.sum(dB_dN < -1e-10)
                return False, f"Expected increasing B(N), found {violations} decreasing segments"
                
        elif expected_monotonic == "decreasing":
            if not np.all(dB_dN <= 1e-10):  # Small tolerance for numerical errors
                violations = np.sum(dB_dN > 1e-10)
                return False, f"Expected decreasing B(N), found {violations} increasing segments"
        
        # General monotonicity check (warn if non-monotonic)
        if not (np.all(dB_dN >= -1e-10) or np.all(dB_dN <= 1e-10)):
            warnings.warn("B(N) is not monotonic. This may be physically unusual.", 
                         ValidationWarning)
        
        return True, "Valid"


class ParameterValidator(Validator):
    """Validates parameter ranges and relationships."""
    
    def __init__(self, param_constraints: Dict[str, Dict[str, Any]]):
        """
        Initialize with parameter constraints.
        
        Args:
            param_constraints: Dict mapping parameter names to constraint dicts
                e.g., {"kappa": {"min": 0, "max": 10, "type": float}}
        """
        self.constraints = param_constraints
    
    def validate(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameter values against constraints."""
        
        for param_name, constraints in self.constraints.items():
            if param_name not in params:
                if constraints.get("required", False):
                    return False, f"Required parameter '{param_name}' missing"
                continue
            
            value = params[param_name]
            
            # Type check
            expected_type = constraints.get("type")
            if expected_type and not isinstance(value, expected_type):
                return False, f"Parameter '{param_name}' should be {expected_type.__name__}, got {type(value).__name__}"
            
            # Range checks
            min_val = constraints.get("min")
            if min_val is not None and value < min_val:
                return False, f"Parameter '{param_name}' = {value} below minimum {min_val}"
                
            max_val = constraints.get("max")
            if max_val is not None and value > max_val:
                return False, f"Parameter '{param_name}' = {value} above maximum {max_val}"
            
            # Custom validation function
            custom_validator = constraints.get("validator")
            if custom_validator and callable(custom_validator):
                try:
                    is_valid = custom_validator(value)
                    if not is_valid:
                        return False, f"Parameter '{param_name}' failed custom validation"
                except Exception as e:
                    return False, f"Parameter '{param_name}' validation error: {e}"
        
        return True, "Valid"


class PhysicsValidator:
    """Main physics validation coordinator."""
    
    def __init__(self):
        self.microlaw_validator = MicrolawValidator()
        self.bn_validator = BNFamilyValidator()
        self.monotonicity_validator = MonotonicityValidator()
        
        # Default parameter constraints
        self.default_constraints = {
            # Exponential B(N) constraints
            "kappa": {"min": 0.01, "max": 100, "type": (int, float)},
            # Power law B(N) constraints  
            "alpha": {"min": -5, "max": 5, "type": (int, float)},
            # Linear B(N) constraints
            "a": {"min": -1000, "max": 1000, "type": (int, float)},
            "b": {"min": -1000, "max": 1000, "type": (int, float)},
        }
    
    def validate_microlaw(self, 
                         microlaw: Microlaw, 
                         inputs: MicrolawInput, 
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate microlaw computation."""
        is_valid, message = self.microlaw_validator.validate(microlaw, inputs, params)
        
        return {
            "component": "microlaw",
            "name": microlaw.name,
            "is_valid": is_valid,
            "message": message,
            "inputs": inputs.to_dict(),
            "params": params
        }
    
    def validate_bn_family(self, 
                          bn_family: BNFamily, 
                          params: Dict[str, Any],
                          check_monotonicity: Optional[str] = None) -> Dict[str, Any]:
        """Validate B(N) family computation and properties."""
        
        # Basic validation
        is_valid, message = self.bn_validator.validate(bn_family, params)
        
        result = {
            "component": "bn_family",
            "name": bn_family.name,
            "is_valid": is_valid,
            "message": message,
            "params": params
        }
        
        if not is_valid:
            return result
        
        # Monotonicity validation
        if check_monotonicity:
            mono_valid, mono_message = self.monotonicity_validator.validate(
                bn_family, params, expected_monotonic=check_monotonicity)
            result["monotonicity_valid"] = mono_valid
            result["monotonicity_message"] = mono_message
            
            if not mono_valid:
                result["is_valid"] = False
                result["message"] += f" | {mono_message}"
        
        return result
    
    def validate_parameters(self, 
                          params: Dict[str, Any], 
                          custom_constraints: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Validate parameter values."""
        
        constraints = self.default_constraints.copy()
        if custom_constraints:
            constraints.update(custom_constraints)
        
        validator = ParameterValidator(constraints)
        is_valid, message = validator.validate(params)
        
        return {
            "component": "parameters",
            "is_valid": is_valid,
            "message": message,
            "params": params,
            "constraints_applied": list(constraints.keys())
        }
    
    def validate_complete_setup(self, 
                               microlaw: Microlaw,
                               inputs: MicrolawInput,
                               ml_params: Dict[str, Any],
                               bn_family: BNFamily,
                               bn_params: Dict[str, Any],
                               check_monotonicity: Optional[str] = None) -> Dict[str, Any]:
        """Validate complete microlaw + B(N) setup."""
        
        results = {
            "overall_valid": True,
            "validations": []
        }
        
        # Validate microlaw parameters
        ml_param_result = self.validate_parameters(ml_params)
        results["validations"].append(ml_param_result)
        if not ml_param_result["is_valid"]:
            results["overall_valid"] = False
        
        # Validate B(N) parameters
        bn_param_result = self.validate_parameters(bn_params)
        results["validations"].append(bn_param_result)
        if not bn_param_result["is_valid"]:
            results["overall_valid"] = False
        
        # Validate microlaw computation
        ml_result = self.validate_microlaw(microlaw, inputs, ml_params)
        results["validations"].append(ml_result)
        if not ml_result["is_valid"]:
            results["overall_valid"] = False
            return results  # Can't proceed if microlaw fails
        
        # Validate B(N) family
        bn_result = self.validate_bn_family(bn_family, bn_params, check_monotonicity)
        results["validations"].append(bn_result)
        if not bn_result["is_valid"]:
            results["overall_valid"] = False
        
        return results


# Global validator instance
_physics_validator = PhysicsValidator()

# Convenience functions
def validate_microlaw(microlaw: Microlaw, inputs: MicrolawInput, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate microlaw computation."""
    return _physics_validator.validate_microlaw(microlaw, inputs, params)

def validate_bn_family(bn_family: BNFamily, params: Dict[str, Any], check_monotonicity: Optional[str] = None) -> Dict[str, Any]:
    """Validate B(N) family computation."""
    return _physics_validator.validate_bn_family(bn_family, params, check_monotonicity)

def validate_parameters(params: Dict[str, Any], custom_constraints: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Validate parameter values."""
    return _physics_validator.validate_parameters(params, custom_constraints)

def validate_complete_setup(microlaw: Microlaw, inputs: MicrolawInput, ml_params: Dict[str, Any],
                           bn_family: BNFamily, bn_params: Dict[str, Any], 
                           check_monotonicity: Optional[str] = None) -> Dict[str, Any]:
    """Validate complete setup."""
    return _physics_validator.validate_complete_setup(
        microlaw, inputs, ml_params, bn_family, bn_params, check_monotonicity)
