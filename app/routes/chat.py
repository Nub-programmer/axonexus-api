from fastapi import APIRouter, Depends

from app.core.schemas import ChatRequest, ChatResponse
from app.core.auth import verify_api_key
from app.providers import MockProvider

router = APIRouter(prefix="/v1", tags=["Chat"])

provider = MockProvider()


@router.post("/chat", response_model=ChatResponse)
def create_chat_completion(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatResponse:
    return provider.chat_completion(request)
