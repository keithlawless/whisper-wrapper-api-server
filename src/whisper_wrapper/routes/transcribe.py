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

# Seconds of silence pad_silence() appends so Whisper flushes its last segment.
# Also the margin past the real audio beyond which any segment must be phantom
# (see _drop_segments_past).
_PAD_SILENCE_SEC = 1.0


def _drop_segments_past(result, cutoff_sec: float):
    """Drop segments/words starting at/after cutoff_sec; rebuild text.

    Whisper decode-loop hallucinations emit segments whose timestamps run past
    the end of the real (padded) audio — a 30s clip producing "Thank you."
    segments out to 56s. The padded tail is silence, so anything starting
    beyond it cannot be real speech. Rebuilding text from the survivors keeps
    the response consistent. Returns (result, dropped_count). This is a
    decode-parameter-independent backstop: it catches any future loop mode
    regardless of temperature/threshold settings.
    """
    kept = [s for s in result.segments if s.start < cutoff_sec]
    dropped = len(result.segments) - len(kept)
    if dropped:
        result.segments = kept
        result.text = "".join(s.text for s in kept).strip()
        if result.words is not None:
            result.words = [w for w in result.words if w.start < cutoff_sec]
    return result, dropped


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
    # VAD
    vad: bool = Form(default=True),
    vad_threshold: float = Form(default=0.5),
    vad_min_speech_ms: int = Form(default=250),
    vad_min_silence_ms: int = Form(default=100),
    vad_speech_pad_ms: int = Form(default=150),
    # Audio pre-processing
    normalize: bool = Form(default=True),
    pad_silence: bool = Form(default=True),
    # Model parameters
    word_timestamps: bool = Form(default=False),
    initial_prompt: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    no_speech_threshold: float = Form(default=0.8),
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

    if normalize:
        decoded = audio_mod.normalize(decoded)

    vad_applied = False
    if vad:
        decoded, vad_applied = await asyncio.to_thread(
            vad_mod.apply,
            decoded,
            threshold=vad_threshold,
            min_speech_ms=vad_min_speech_ms,
            min_silence_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
        )

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

    # Measure real audio duration before appending synthetic silence.
    duration = audio_mod.duration_seconds(decoded)

    if pad_silence:
        decoded = audio_mod.pad_silence(decoded, _PAD_SILENCE_SEC)

    transcriber = await mgr.get(model_size)

    log.info(
        "transcribe_start",
        model=model_size,
        backend=mgr.backend_name(),
        language=language,
        duration_sec=round(duration, 2),
        vad_applied=vad_applied,
        normalize=normalize,
        pad_silence=pad_silence,
        no_speech_threshold=no_speech_threshold,
    )

    async with mgr.semaphore:
        result = await asyncio.to_thread(
            transcriber.transcribe,
            decoded,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            temperature=temperature,
            no_speech_threshold=no_speech_threshold,
        )

    # Backstop against decode-loop hallucinations whose timestamps run past the
    # end of the real audio. Real speech ends by `duration`; the padded tail is
    # silence, so any segment starting beyond it is phantom.
    cutoff_sec = duration + (_PAD_SILENCE_SEC if pad_silence else 0.0)
    result, dropped = _drop_segments_past(result, cutoff_sec)
    if dropped:
        log.warning(
            "dropped_hallucinated_segments_past_audio",
            model=model_size,
            dropped=dropped,
            cutoff_sec=round(cutoff_sec, 2),
            request_id=request_id,
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
