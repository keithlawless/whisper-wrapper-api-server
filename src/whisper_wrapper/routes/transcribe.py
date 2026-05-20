from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Form, Request, UploadFile

from whisper_wrapper import audio as audio_mod
from whisper_wrapper import vad as vad_mod
from whisper_wrapper.config import Settings
from whisper_wrapper.errors import AudioDecodeError
from whisper_wrapper.logging import get_logger
from whisper_wrapper.models import ModelManager
from whisper_wrapper.schemas import TranscribeResponse

router = APIRouter()
log = get_logger("transcribe")


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _mgr(request: Request) -> ModelManager:
    return request.app.state.model_manager


@router.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),  # noqa: B008
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    vad: bool = Form(default=True),
    word_timestamps: bool = Form(default=False),
    initial_prompt: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    format: str | None = Form(default=None),
) -> TranscribeResponse:
    settings = _settings(request)
    mgr = _mgr(request)
    request_id = request.state.request_id

    model_size = (model or settings.default_model).strip()

    raw = await audio.read()
    if not raw:
        raise AudioDecodeError("empty upload")

    decoded = audio_mod.decode(
        raw,
        declared_format=format,
        filename=audio.filename,
    )
    audio_mod.enforce_max_duration(decoded)

    vad_applied = False
    if vad:
        decoded, vad_applied = await asyncio.to_thread(vad_mod.apply, decoded)

    duration = audio_mod.duration_seconds(decoded)

    if decoded.size == 0:
        return TranscribeResponse(
            text="",
            language=None,
            duration_sec=0.0,
            model=model_size,
            segments=[],
            words=None,
            vad_applied=vad_applied,
            request_id=request_id,
        )

    transcriber = await mgr.get(model_size)

    log.info(
        "transcribe_start",
        model=model_size,
        backend=mgr.backend_name(),
        language=language,
        duration_sec=round(duration, 2),
        vad_applied=vad_applied,
    )

    async with mgr.semaphore:
        result = await asyncio.to_thread(
            transcriber.transcribe,
            decoded,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            temperature=temperature,
        )

    log.info(
        "transcribe_done",
        model=model_size,
        backend=mgr.backend_name(),
        text_chars=len(result.text),
        segment_count=len(result.segments),
        detected_language=result.language,
    )

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        language_probability=result.language_probability,
        duration_sec=round(duration, 3),
        model=model_size,
        segments=result.segments,
        words=result.words,
        vad_applied=vad_applied,
        request_id=request_id,
    )
