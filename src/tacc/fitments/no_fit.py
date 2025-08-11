"""No-fit fitment: pass-through fitment that returns parameters unchanged."""

from typing import Dict, Any, Iterable
from tacc.core.fitment import Fitment, register


class NoFit(Fitment):
    """Pass-through fitment that returns parameters unchanged."""
    
    name = "no_fit"
    
    def fit(
        self, 
        dataset: Iterable[Dict[str, Any]],
        microlaw_name: str, 
        ml_params: Dict[str, Any],
        bn_family: str, 
        bn_params: Dict[str, Any],
        loss=None, 
        extra: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Return parameters unchanged (pass-through)."""
        return {"microlaw": dict(ml_params), "bn": dict(bn_params)}


# Register the fitment
register(NoFit())
