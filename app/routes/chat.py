from fastapi import APIRouter, Depends

from app.core.schemas import ChatRequest, ChatResponse, Choice, Message, Usage
from app.core.auth import verify_api_key

router = APIRouter(prefix="/v1", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatResponse:
    last_user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_message = msg.content
            break
    
    mock_response_content = (
        f"[Mock Response] You requested model '{request.model}'. "
        f"Your message: '{last_user_message[:100]}{'...' if len(last_user_message) > 100 else ''}'. "
        f"This is a placeholder response. In production, this would be forwarded to the actual AI provider."
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
