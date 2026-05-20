from __future__ import annotations

import threading

import numpy as np

from whisper_wrapper.audio import TARGET_SR
from whisper_wrapper.logging import get_logger

log = get_logger("vad")

_lock = threading.Lock()
_model = None  # silero-vad jit model
_get_speech_timestamps = None
_load_failed = False


def _load() -> None:
    """Lazy-import silero-vad. If import fails (e.g. torch missing), disable VAD."""
    global _model, _get_speech_timestamps, _load_failed
    if _model is not None or _load_failed:
        return
    with _lock:
        if _model is not None or _load_failed:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore
        except Exception as e:
            log.warning("vad_unavailable", reason=str(e))
            _load_failed = True
            return
        try:
            _model = load_silero_vad()
            _get_speech_timestamps = get_speech_timestamps
        except Exception as e:
            log.warning("vad_load_failed", reason=str(e))
            _load_failed = True


def is_available() -> bool:
    _load()
    return _model is not None and not _load_failed


def apply(
    audio: np.ndarray,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
    speech_pad_ms: int = 150,
) -> tuple[np.ndarray, bool]:
    """Concatenate detected speech segments. Returns (filtered_audio, applied_bool).

    If VAD is unavailable, returns the original audio with applied=False.
    """
    _load()
    if _model is None:
        return audio, False

    # silero expects a torch tensor of float32 audio.
    try:
        import torch  # type: ignore
    except Exception as e:
        log.warning("vad_torch_missing", reason=str(e))
        return audio, False

    tensor = torch.from_numpy(audio.astype(np.float32, copy=False))
    timestamps = _get_speech_timestamps(  # type: ignore[misc]
        tensor,
        _model,
        sampling_rate=TARGET_SR,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    if not timestamps:
        # All silence: return a tiny slice so the model doesn't crash on zero-len.
        return audio[:0], True

    chunks = [audio[seg["start"] : seg["end"]] for seg in timestamps]
    return np.concatenate(chunks), True
