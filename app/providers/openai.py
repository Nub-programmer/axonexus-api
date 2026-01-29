import logging
import random
from typing import Optional

from openai import OpenAI
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
        # Strip 'openai:' prefix if present
        if model.startswith("openai:"):
            model = model[7:]
            
        logger.info(f"OpenAIProvider: sending request for model '{model}'")

        if not self.active_keys:
            raise ValueError("No OpenAI API keys configured")

        selected_key = random.choice(self.active_keys)
        # Obfuscate key in logs
        masked_key = f"{selected_key[:8]}...{selected_key[-4:]}"
        logger.info(f"Using OpenAI key: {masked_key}")

        try:
            client = OpenAI(api_key=selected_key)
            
            # Convert internal messages to OpenAI format
            openai_messages = [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ]

            response = client.chat.completions.create(
                model=model,
                messages=openai_messages
            )

            # Map OpenAI response to our internal ChatResponse schema
            choices = [
                Choice(
                    index=c.index,
                    message=Message(
                        role=c.message.role,
                        content=c.message.content
                    ),
                    finish_reason=c.finish_reason
                )
                for c in response.choices
            ]

            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )

            return ChatResponse(
                id=response.id,
                object="chat.completion",
                created=response.created,
                model=response.model,
                choices=choices,
                usage=usage
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise e
