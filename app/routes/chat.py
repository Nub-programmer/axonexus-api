import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.schemas import ChatRequest, ChatResponse
from app.core.auth import verify_api_key
from app.providers import MockProvider, GroqProvider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Chat"])

mock_provider = MockProvider()
groq_provider = GroqProvider()


@router.post("/chat", response_model=ChatResponse)
def create_chat_completion(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatResponse:
    model = request.model

    if model == "axon-mock":
        logger.info(f"Routing to MockProvider for model: {model}")
        return mock_provider.chat_completion(request)

    if model.startswith("groq:"):
        groq_model = model[5:]
        logger.info(f"Routing to GroqProvider for model: {groq_model}")
        try:
            return groq_provider.chat_completion(request, model_name=groq_model)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"GroqProvider error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Groq API error: {str(e)}"
            )

    logger.info(f"Unknown model '{model}', defaulting to MockProvider")
    return mock_provider.chat_completion(request)
