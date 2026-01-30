from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.openapi.docs import get_swagger_ui_html
from app.core.schemas import ChatRequest, ChatResponse, ModelListResponse, ModelInfo
from app.core.auth import verify_api_key, security
from app.providers.router import get_router
from app.models.registry import get_available_models, suggest_model
from app.core.limiter import get_limiter
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Chat"])
provider_router = get_router()
limiter = get_limiter()

@router.post("/chat", response_model=ChatResponse)
async def create_chat_completion(
    request_data: ChatRequest,
    fastapi_request: Request,
    api_key: str = Depends(verify_api_key)
) -> ChatResponse:
    # Identify client
    client_key = api_key or fastapi_request.client.host
    is_test = api_key == "axn_test_123"
    is_guest = api_key is None
    
    # 2) Rate limiting
    if not limiter.check_rate_limit(client_key, is_test, is_guest):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down."
        )
        
    # 3) Usage tracking
    if not limiter.check_usage_limit(client_key, is_test, is_guest):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Daily token limit exceeded."
        )

    try:
        return provider_router.route_chat(request_data, client_key, is_test, is_guest)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected internal error occurred.")

@router.get("/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    """Lists all available model aliases."""
    models = [ModelInfo(id=alias) for alias in get_available_models()]
    return ModelListResponse(data=models)
