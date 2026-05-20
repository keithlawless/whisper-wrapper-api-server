from __future__ import annotations

from fastapi import APIRouter, Request

from whisper_wrapper import API_VERSION, SERVICE_NAME, __version__
from whisper_wrapper.models import ModelManager
from whisper_wrapper.schemas import HealthResponse, ModelsResponse


def _mgr(request: Request) -> ModelManager:
    return request.app.state.model_manager


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    mgr = _mgr(request)
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=__version__,
        api_version=API_VERSION,
        backend="faster-whisper",
        loaded_models=mgr.loaded(),
        device=mgr.device(),
        compute_type=mgr.compute_type(),
    )


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    mgr = _mgr(request)
    return ModelsResponse(
        supported=mgr.supported(),
        loaded=mgr.loaded(),
        cached_on_disk=mgr.cached_on_disk(),
    )
