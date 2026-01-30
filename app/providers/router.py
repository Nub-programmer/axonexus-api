import logging
from typing import Dict, Any, Optional
from app.providers import (
    MockProvider, 
    GroqProvider, 
    NVIDIAProvider, 
    OpenRouterProvider, 
    MistralProvider
)
from app.core.schemas import ChatRequest, ChatResponse
from app.models.registry import resolve_model

logger = logging.getLogger(__name__)

class ProviderRouter:
    def __init__(self):
        self.providers = {
            "mock": MockProvider(),
            "groq": GroqProvider(),
            "nvidia": NVIDIAProvider(),
            "openrouter": OpenRouterProvider(),
            "mistral": MistralProvider()
        }

    def route_chat(self, request: ChatRequest) -> ChatResponse:
        model_alias = request.model
        resolved = resolve_model(model_alias)
        
        if not resolved:
            # Attempt fuzzy matching internally if direct resolution fails
            from app.models.registry import suggest_model
            suggestion = suggest_model(model_alias)
            if suggestion:
                logger.info(f"Fuzzy matched '{model_alias}' to '{suggestion}'")
                model_alias = suggestion
                request.model = suggestion # Propagate resolved alias
                resolved = resolve_model(suggestion)
        
        if not resolved:
            raise ValueError(f"Model alias '{model_alias}' not found in registry")
            
        provider_name = resolved["provider"]
        internal_model = resolved["internal_model"]
        
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not implemented")
            
        logger.info(f"Routing '{model_alias}' to '{provider_name}' (internal: {internal_model})")
        try:
            return provider.chat_completion(request, model_name=internal_model)
        except ValueError as e:
            # Handle missing key or specific validation errors as 400
            raise e
        except Exception as e:
            logger.error(f"Provider error: {e}")
            raise RuntimeError(f"The model provider for '{model_alias}' encountered an issue: {str(e)}")

router = ProviderRouter()

def get_router() -> ProviderRouter:
    return router
