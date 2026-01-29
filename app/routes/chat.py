import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.schemas import ChatRequest, ChatResponse
from app.core.auth import verify_api_key
from app.providers import MockProvider, GroqProvider, NVIDIAProvider
from app.core.models import resolve_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Chat"])

mock_provider = MockProvider()
groq_provider = GroqProvider()
nvidia_provider = NVIDIAProvider()


@router.post("/chat", response_model=ChatResponse)
def create_chat_completion(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatResponse:
    model_alias = request.model
    resolved = resolve_model(model_alias)

    if not resolved:
        logger.warning(f"Unsupported model requested: {model_alias}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_alias}' is not supported. Please use a valid model alias."
        )

    provider = resolved["provider"]
    internal_model = resolved["internal_model"]

    if provider == "mock":
        logger.info(f"Routing to MockProvider for model: {model_alias}")
        return mock_provider.chat_completion(request)

    if provider == "nvidia":
        logger.info(f"Routing to NVIDIAProvider for model: {internal_model}")
        try:
            return nvidia_provider.chat_completion(request, model_name=internal_model)
        except Exception as e:
            logger.error(f"Provider error: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The requested model provider returned an error."
            )

    if provider == "groq":
        logger.info(f"Routing to GroqProvider for model: {internal_model}")
        try:
            return groq_provider.chat_completion(request, model_name=internal_model)
        except Exception as e:
            logger.error(f"Provider error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The requested model provider returned an error."
            )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Model '{model_alias}' is correctly registered but its provider '{provider}' is not implemented."
    )
