import logging
from typing import Optional, List
import random

from app.providers.base import BaseProvider
from app.core.schemas import ChatRequest, ChatResponse, Choice, Message, Usage
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class OpenAIProvider(BaseProvider):
    def __init__(self):
        self.keys = [
            settings.openai_key_1,
            settings.openai_key_2,
            settings.openai_key_3
        ]
        # Filter out empty keys
        self.active_keys = [k for k in self.keys if k]
        if not self.active_keys:
            logger.warning("No OpenAI keys found in environment")

    def chat_completion(self, request: ChatRequest, model_name: Optional[str] = None) -> ChatResponse:
        model = model_name or request.model
        logger.info(f"OpenAIProvider: handling model '{model}'")

        if not self.active_keys:
            raise ValueError("No OpenAI API keys configured")

        # In a real implementation, we would use httpx to call OpenAI API here
        # For this task, we will mock the response but show we can access the keys
        selected_key = random.choice(self.active_keys)
        # Obfuscate key in logs
        masked_key = f"{selected_key[:8]}...{selected_key[-4:]}"
        logger.info(f"Using OpenAI key: {masked_key}")

        last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")

        content = f"[OpenAI Response] Model: {model}. This is a simulated OpenAI response using a configured key."
        
        return ChatResponse(
            model=model,
            choices=[Choice(index=0, message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
