import difflib
from typing import Dict, Any, Optional, List

# User-facing model aliases to internal provider mapping
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "llama-3.1": {
        "provider": "nvidia",
        "internal_model": "meta/llama-3.1-8b-instruct",
    },
    "axon-mock": {
        "provider": "mock",
        "internal_model": "axon-mock",
    }
}

DEFAULT_MODEL = "llama-3.1"

def resolve_model(alias: str) -> Optional[Dict[str, Any]]:
    """Resolves a model alias, returns the mapping or None."""
    return MODEL_REGISTRY.get(alias)

def get_available_models() -> List[str]:
    """Returns a list of all model aliases."""
    return list(MODEL_REGISTRY.keys())

def suggest_model(alias: str) -> Optional[str]:
    """Suggests the closest model alias using fuzzy matching."""
    matches = difflib.get_close_matches(alias, MODEL_REGISTRY.keys(), n=1, cutoff=0.3)
    return matches[0] if matches else None
