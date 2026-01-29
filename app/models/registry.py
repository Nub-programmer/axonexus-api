from typing import Dict, Any, Optional

# User-facing model aliases to internal model mapping
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

def resolve_model(alias: str) -> Optional[Dict[str, Any]]:
    """Resolves a user-facing model alias to its provider and internal model ID."""
    return MODEL_REGISTRY.get(alias)

def get_available_models():
    """Returns a list of available model aliases."""
    return list(MODEL_REGISTRY.keys())
