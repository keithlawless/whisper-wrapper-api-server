from __future__ import annotations

import contextlib

from fastapi import APIRouter, BackgroundTasks, Request

from whisper_wrapper.config import SUPPORTED_MODELS
from whisper_wrapper.errors import ModelNotAvailableError
from whisper_wrapper.models import ModelManager
from whisper_wrapper.schemas import PreloadResponse

router = APIRouter()


def _mgr(request: Request) -> ModelManager:
    return request.app.state.model_manager


async def _do_preload(mgr: ModelManager, size: str) -> None:
    with contextlib.suppress(Exception):
        await mgr.get(size)


@router.post("/v1/models/{size}/preload", response_model=PreloadResponse)
async def preload(size: str, request: Request, background: BackgroundTasks) -> PreloadResponse:
    if size not in SUPPORTED_MODELS:
        raise ModelNotAvailableError(
            f"unsupported model size '{size}'. supported: {', '.join(SUPPORTED_MODELS)}"
        )
    mgr = _mgr(request)
    if size in mgr.loaded():
        return PreloadResponse(status="loaded", size=size)
    background.add_task(_do_preload, mgr, size)
    return PreloadResponse(status="loading", size=size)
