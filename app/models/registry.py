import difflib
from typing import Dict, Any, Optional, List
from app.core.config import get_settings

settings = get_settings()

# User-facing model aliases to internal provider mapping
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "axon-gpt-4o": {
        "provider": "openrouter",
        "internal_model": "openai/gpt-4o",
        "required_key": "openrouter_api_key"
    },
    "axon-gpt-4": {
        "provider": "openrouter",
        "internal_model": "openai/gpt-4",
        "required_key": "openrouter_api_key"
    },
    "axon-claude-sonnet": {
        "provider": "openrouter",
        "internal_model": "anthropic/claude-3-sonnet",
        "required_key": "openrouter_api_key"
    },
    "axon-gemini-pro": {
        "provider": "openrouter",
        "internal_model": "google/gemini-pro",
        "required_key": "openrouter_api_key"
    },
    "axon-llama-3-70b": {
        "provider": "groq",
        "internal_model": "llama3-70b-8192",
        "required_key": "groq_api_key"
    },
    "axon-llama-3-8b": {
        "provider": "groq",
        "internal_model": "llama3-8b-8192",
        "required_key": "groq_api_key"
    },
    "axon-mixtral": {
        "provider": "groq",
        "internal_model": "mixtral-8x7b-32768",
        "required_key": "groq_api_key"
    },
    "axon-mistral-large": {
        "provider": "mistral",
        "internal_model": "mistral-large-latest",
        "required_key": "mistral_api_key"
    },
    "axon-mistral-medium": {
        "provider": "mistral",
        "internal_model": "mistral-medium-latest",
        "required_key": "mistral_api_key"
    },
    "axon-llama-nvidia": {
        "provider": "nvidia",
        "internal_model": "meta/llama-3.1-8b-instruct",
        "required_key": "nvidia_api_key"
    },
    "axon-mistral-nvidia": {
        "provider": "nvidia",
        "internal_model": "mistralai/mistral-7b-instruct-v0.3",
        "required_key": "nvidia_api_key"
    },
    "axon-mock": {
        "provider": "mock",
        "internal_model": "axon-mock",
        "required_key": None
    }
}

DEFAULT_MODEL = "axon-mock"

def resolve_model(alias: str) -> Optional[Dict[str, Any]]:
    """Resolves a model alias, returns the mapping or None."""
    model_info = MODEL_REGISTRY.get(alias)
    if not model_info:
        return None
    
    # Check if required key is present
    required_key = model_info.get("required_key")
    if required_key and not getattr(settings, required_key):
        return None
        
    return model_info

def get_available_models() -> List[str]:
    """Returns a list of all model aliases that have configured keys."""
    available = []
    for alias, info in MODEL_REGISTRY.items():
        required_key = info.get("required_key")
        if not required_key or getattr(settings, required_key):
            available.append(alias)
    return available

def suggest_model(alias: str) -> Optional[str]:
    """Suggests the closest model alias using fuzzy matching among available models."""
    available_models = get_available_models()
    matches = difflib.get_close_matches(alias, available_models, n=1, cutoff=0.3)
    return matches[0] if matches else None
