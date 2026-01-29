from abc import ABC, abstractmethod
from app.core.schemas import ChatRequest, ChatResponse


class BaseProvider(ABC):
    @abstractmethod
    def chat_completion(self, request: ChatRequest) -> ChatResponse:
        pass
