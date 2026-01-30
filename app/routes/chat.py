from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.openapi.docs import get_swagger_ui_html
from app.core.schemas import ChatRequest, ChatResponse, ModelListResponse, ModelInfo
from app.core.auth import verify_api_key
from app.providers.router import get_router
from app.models.registry import get_available_models, suggest_model
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Chat"])
provider_router = get_router()

@router.post("/chat", response_model=ChatResponse)
def create_chat_completion(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatResponse:
    try:
        return provider_router.route_chat(request)
    except ValueError as e:
        # 400 errors for validation, missing keys, or resolved suggestions
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        # Provider specific runtime failures
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred."
        )

@router.get("/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    """Lists all available model aliases."""
    models = [ModelInfo(id=alias) for alias in get_available_models()]
    return ModelListResponse(data=models)
