from app.providers.base import BaseProvider
from app.core.schemas import ChatRequest, ChatResponse, Choice, Message, Usage


class MockProvider(BaseProvider):
    def chat_completion(self, request: ChatRequest) -> ChatResponse:
        last_user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_message = msg.content
                break

        mock_response_content = (
            f"[Mock Response] You requested model '{request.model}'. "
            f"Your message: '{last_user_message[:100]}{'...' if len(last_user_message) > 100 else ''}'. "
            f"This is a placeholder response from MockProvider."
        )

        response_message = Message(role="assistant", content=mock_response_content)

        prompt_tokens = sum(len(msg.content.split()) for msg in request.messages) * 4
        completion_tokens = len(mock_response_content.split()) * 4

        return ChatResponse(
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=response_message,
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
