import os
import logging
from typing import Optional

import httpx

from app.providers.base import BaseProvider
from app.core.schemas import ChatRequest, ChatResponse, Choice, Message, Usage

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseProvider):
    def __init__(self) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY")

    def chat_completion(self, request: ChatRequest, model_name: Optional[str] = None) -> ChatResponse:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        actual_model = model_name or request.model
        logger.info(f"GroqProvider: calling model '{actual_model}'")

        messages_payload = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        payload = {
            "model": actual_model,
            "messages": messages_payload,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(GROQ_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        choices = [
            Choice(
                index=choice["index"],
                message=Message(
                    role=choice["message"]["role"],
                    content=choice["message"]["content"]
                ),
                finish_reason=choice.get("finish_reason", "stop")
            )
            for choice in data["choices"]
        ]

        usage = Usage(
            prompt_tokens=data["usage"]["prompt_tokens"],
            completion_tokens=data["usage"]["completion_tokens"],
            total_tokens=data["usage"]["total_tokens"]
        )

        return ChatResponse(
            id=data["id"],
            object=data["object"],
            created=data["created"],
            model=data["model"],
            choices=choices,
            usage=usage
        )
