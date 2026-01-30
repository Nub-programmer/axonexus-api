import logging
from typing import Optional
from openai import OpenAI
from app.providers.base import BaseProvider
from app.core.schemas import ChatRequest, ChatResponse, Choice, Message, Usage
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class OpenRouterProvider(BaseProvider):
    def __init__(self):
        self.api_key = settings.openrouter_api_key
        if not self.api_key:
            self.client = None
        else:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )

    def chat_completion(self, request: ChatRequest, model_name: Optional[str] = None) -> ChatResponse:
        if not self.client:
            raise ValueError("OpenRouter API key not configured")
        
        model = model_name or "google/gemini-pro"
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in request.messages] # type: ignore
            )
            
            choices = [
                Choice(
                    index=c.index,
                    message=Message(role=c.message.role or "assistant", content=c.message.content or ""),
                    finish_reason=c.finish_reason or "stop"
                ) for c in response.choices
            ]
            
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0
            )

            return ChatResponse(
                id=response.id,
                created=response.created,
                model=request.model,
                choices=choices,
                usage=usage
            )
        except Exception as e:
            logger.error(f"OpenRouter error: {e}")
            raise e
