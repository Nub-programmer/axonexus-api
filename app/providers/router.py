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

from app.core.limiter import get_limiter

limiter = get_limiter()

class ProviderRouter:
    def __init__(self):
        self.providers = {
            "mock": MockProvider(),
            "groq": GroqProvider(),
            "nvidia": NVIDIAProvider(),
            "openrouter": OpenRouterProvider(),
            "mistral": MistralProvider()
        }

    def _should_inject_identity(self, messages: list) -> bool:
        if not messages:
            return False
        
        # Check last message content
        last_message = messages[-1].get("content", "").lower()
        trigger_keywords = ["axon", "axonnexus", "axoninnova"]
        trigger_phrases = ["who built", "who made", "who runs", "who developed", "created by"]
        
        if any(kw in last_message for kw in trigger_keywords):
            return True
        if any(phrase in last_message for phrase in trigger_phrases):
            return True
            
        return False

    def route_chat(self, request: ChatRequest, client_key: str, is_test: bool, is_guest: bool) -> ChatResponse:
        # 1) Enforce hard limits
        request.max_tokens = min(request.max_tokens or 300, 300)
        request.temperature = 0.7 if request.temperature is None else 0.7
        
        # New Requirement: Conditional Axon identity injection
        if self._should_inject_identity([m.dict() for m in request.messages]):
            from app.core.schemas import Message
            identity_msg = Message(role="system", content="AxonNexus is a first-party AI API developed and maintained by the AxonInnova community.")
            # Insert after any existing system message or at start
            request.messages.insert(0, identity_msg)

        model_alias = request.model
        resolved = resolve_model(model_alias)
        
        if not resolved:
            # Attempt fuzzy matching internally
            from app.models.registry import suggest_model
            suggestion = suggest_model(model_alias)
            if suggestion:
                logger.info(f"Fuzzy matched '{model_alias}' to '{suggestion}'")
                model_alias = suggestion
                request.model = suggestion
                resolved = resolve_model(suggestion)
        
        if not resolved:
            raise ValueError(f"Model alias '{model_alias}' not found in registry")

        # 4) Enforce model access tiers
        if is_guest and (resolved.get("large") or resolved.get("premium")):
            raise ValueError(f"Model '{model_alias}' requires authentication")
        if is_test and resolved.get("premium"):
            raise ValueError(f"Model '{model_alias}' requires a premium API key")
            
        provider_name = resolved["provider"]
        internal_model = resolved["internal_model"]
        
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not implemented")
            
        logger.info(f"Routing '{model_alias}' to '{provider_name}' (internal: {internal_model})")
        try:
            response = provider.chat_completion(request, model_name=internal_model)
            # 3) Track token usage
            limiter.update_usage(client_key, response.usage.total_tokens)
            return response
        except ValueError as e:
            raise e
        except Exception as e:
            logger.error(f"Provider error: {e}")
            raise RuntimeError(f"The model provider for '{model_alias}' encountered an issue: {str(e)}")

router = ProviderRouter()

def get_router() -> ProviderRouter:
    return router
