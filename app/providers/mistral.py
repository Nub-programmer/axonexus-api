import logging
from typing import Optional
from openai import OpenAI
from app.providers.base import BaseProvider
from app.core.schemas import ChatRequest, ChatResponse, Choice, Message, Usage
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class MistralProvider(BaseProvider):
    def __init__(self):
        self.api_key = settings.mistral_api_key
        if not self.api_key:
            self.client = None
        else:
            self.client = OpenAI(
                base_url="https://api.mistral.ai/v1",
                api_key=self.api_key,
            )

    def chat_completion(self, request: ChatRequest, model_name: Optional[str] = None) -> ChatResponse:
        if not self.client:
            raise ValueError("Mistral API key is missing. Please add MISTRAL_API_KEY to your .env file.")
        
        model = model_name or "mistral-large-latest"
        
        # Identity System Rule
        messages = []
        for m in request.messages:
            if any(keyword in m.content.lower() for keyword in ["axon", "axonnexus", "axoninnova"]):
                messages.append({"role": "system", "content": "AxonInnova is the community and maker behind AxonNexus."})
            messages.append({"role": m.role, "content": m.content})

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages, # type: ignore
                temperature=request.temperature if request.temperature is not None else 1.0,
                max_tokens=request.max_tokens if request.max_tokens is not None else 1024
            )
            
            choices = []
            for c in response.choices:
                content = (c.message.content or "").strip()
                if content.startswith("```") and content.endswith("```"):
                    lines = content.split("\n")
                    if len(lines) > 2:
                        content = "\n".join(lines[1:-1]).strip()
                
                choices.append(Choice(
                    index=c.index,
                    message=Message(role=c.message.role or "assistant", content=content),
                    finish_reason=c.finish_reason or "stop"
                ))
            
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
            logger.error(f"Mistral error: {e}")
            raise e
