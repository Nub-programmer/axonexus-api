from abc import ABC, abstractmethod
from typing import Optional
from app.core.schemas import ChatRequest, ChatResponse


class BaseProvider(ABC):
    @abstractmethod
    def chat_completion(self, request: ChatRequest, model_name: Optional[str] = None) -> ChatResponse:
        pass
