import logging
from typing import Optional

from openai import OpenAI
from app.providers.base import BaseProvider
from app.core.schemas import ChatRequest, ChatResponse, Choice, Message, Usage
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class NVIDIAProvider(BaseProvider):
    def __init__(self):
        if not settings.nvidia_api_key:
            logger.error("NVIDIA_API_KEY is missing in environment")
            self.client = None
        else:
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=settings.nvidia_api_key
            )

    def chat_completion(self, request: ChatRequest, model_name: Optional[str] = None) -> ChatResponse:
        model = model_name or "meta/llama-3.1-8b-instruct"
        
        if not self.client:
            raise ValueError("NVIDIA API key not configured. Please check your .env file.")

        logger.info(f"NVIDIAProvider: sending request for model '{model}'")

        try:
            # Convert internal messages to OpenAI format (NVIDIA LLaMA API is OpenAI compatible)
            nvidia_messages = [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ]

            response = self.client.chat.completions.create(
                model=model,
                messages=nvidia_messages  # type: ignore
            )

            if not response.usage:
                raise ValueError("Provider API response missing usage information")

            choices = [
                Choice(
                    index=c.index,
                    message=Message(
                        role=c.message.role or "assistant",
                        content=c.message.content or ""
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
                model=request.model,  # Return the alias, not internal name
                choices=choices,
                usage=usage
            )

        except Exception as e:
            logger.error(f"Provider API error: {e}")
            raise e
