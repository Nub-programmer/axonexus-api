import logging
from typing import Dict, Any, Optional
from app.providers import MockProvider, GroqProvider, NVIDIAProvider
from app.core.schemas import ChatRequest, ChatResponse
from app.models.registry import resolve_model

logger = logging.getLogger(__name__)

class ProviderRouter:
    def __init__(self):
        self.providers = {
            "mock": MockProvider(),
            "groq": GroqProvider(),
            "nvidia": NVIDIAProvider()
        }

    def route_chat(self, request: ChatRequest) -> ChatResponse:
        model_alias = request.model
        resolved = resolve_model(model_alias)
        
        if not resolved:
            raise ValueError(f"Model alias '{model_alias}' not found in registry")
            
        provider_name = resolved["provider"]
        internal_model = resolved["internal_model"]
        
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not implemented")
            
        logger.info(f"Routing '{model_alias}' to '{provider_name}' (internal: {internal_model})")
        return provider.chat_completion(request, model_name=internal_model)

router = ProviderRouter()

def get_router() -> ProviderRouter:
    return router
