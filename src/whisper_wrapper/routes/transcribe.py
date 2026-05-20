from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile

from whisper_wrapper import audio as audio_mod
from whisper_wrapper import vad as vad_mod
from whisper_wrapper.config import Settings
from whisper_wrapper.errors import AudioDecodeError
from whisper_wrapper.logging import get_logger
from whisper_wrapper.models import ModelManager
from whisper_wrapper.schemas import Segment, TranscribeResponse, WordTiming

router = APIRouter()
log = get_logger("transcribe")


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _mgr(request: Request) -> ModelManager:
    return request.app.state.model_manager


def _run_transcribe(
    model: Any,
    audio_arr,
    *,
    language: str | None,
    word_timestamps: bool,
    initial_prompt: str | None,
    temperature: float,
):
    """Sync wrapper around faster-whisper's generator-returning transcribe()."""
    segments, info = model.transcribe(
        audio_arr,
        language=language,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        temperature=temperature,
        condition_on_previous_text=False,
        no_speech_threshold=0.8,
        vad_filter=False,  # we handle VAD ourselves (or skip it)
    )
    # Materialize the generator inside the worker thread.
    materialized = list(segments)
    return materialized, info


@router.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),  # noqa: B008  (standard FastAPI dependency)
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

    # Empty after VAD = no speech detected. Return empty transcript.
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

    whisper_model = await mgr.get(model_size)

    log.info(
        "transcribe_start",
        model=model_size,
        language=language,
        duration_sec=round(duration, 2),
        vad_applied=vad_applied,
    )

    async with mgr.semaphore:
        segments_raw, info = await asyncio.to_thread(
            _run_transcribe,
            whisper_model,
            decoded,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            temperature=temperature,
        )

    segments: list[Segment] = []
    words_out: list[WordTiming] | None = [] if word_timestamps else None
    text_parts: list[str] = []
    for seg in segments_raw:
        text_parts.append(seg.text)
        segments.append(
            Segment(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text,
                no_speech_prob=getattr(seg, "no_speech_prob", None),
                avg_logprob=getattr(seg, "avg_logprob", None),
            )
        )
        if word_timestamps and getattr(seg, "words", None):
            for w in seg.words:
                words_out.append(  # type: ignore[union-attr]
                    WordTiming(
                        start=float(w.start),
                        end=float(w.end),
                        word=w.word,
                        probability=getattr(w, "probability", None),
                    )
                )

    text = "".join(text_parts).strip()

    log.info(
        "transcribe_done",
        model=model_size,
        text_chars=len(text),
        segment_count=len(segments),
        detected_language=getattr(info, "language", None),
    )

    return TranscribeResponse(
        text=text,
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        duration_sec=round(duration, 3),
        model=model_size,
        segments=segments,
        words=words_out,
        vad_applied=vad_applied,
        request_id=request_id,
    )
