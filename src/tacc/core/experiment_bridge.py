"""
Bridge between notebook widgets and experiment runners.
Allows experiments to use fitted parameters from interactive widgets.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Global state file for sharing fitment results between widgets and experiments
_SHARED_STATE_FILE = Path(tempfile.gettempdir()) / "tacc_fitment_state.json"

def save_fitment_state(
    fitment_name: str,
    fitted_params: Dict[str, Any],
    fitment_config: Dict[str, Any] = None
) -> None:
    """Save fitment state to shared file for experiments to use."""
    state = {
        "fitment_name": fitment_name,
        "fitted_params": fitted_params,
        "fitment_config": fitment_config or {},
        "timestamp": str(Path().resolve())  # Simple timestamp
    }
    
    try:
        with open(_SHARED_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass  # Fail silently if can't write

def load_fitment_state() -> Optional[Dict[str, Any]]:
    """Load fitment state from shared file."""
    try:
        if _SHARED_STATE_FILE.exists():
            with open(_SHARED_STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def get_fitted_kappa(default_kappa: float = 2.0) -> float:
    """Get fitted kappa value from shared state, or default if not available."""
    state = load_fitment_state()
    if state and "fitted_params" in state:
        fitted_params = state["fitted_params"]
        if "bn" in fitted_params and "kappa" in fitted_params["bn"]:
            return float(fitted_params["bn"]["kappa"])
    return default_kappa

def get_fitted_params(default_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get all fitted parameters from shared state."""
    state = load_fitment_state()
    if state and "fitted_params" in state:
        return state["fitted_params"]
    return default_params or {"microlaw": {}, "bn": {"kappa": 2.0}}

def is_fitment_active() -> bool:
    """Check if a fitment has been run and is active."""
    state = load_fitment_state()
    return state is not None and state.get("fitment_name") != "no_fit"

def get_active_fitment_info() -> Dict[str, Any]:
    """Get information about the currently active fitment."""
    state = load_fitment_state()
    if state:
        return {
            "name": state.get("fitment_name", "none"),
            "config": state.get("fitment_config", {}),
            "has_fitted_params": bool(state.get("fitted_params"))
        }
    return {"name": "none", "config": {}, "has_fitted_params": False}

def clear_fitment_state() -> None:
    """Clear the shared fitment state."""
    try:
        if _SHARED_STATE_FILE.exists():
            _SHARED_STATE_FILE.unlink()
    except Exception:
        pass
