from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from whisper_wrapper.errors import AudioDecodeError, UnsupportedFormatError

TARGET_SR = 16_000
MAX_DURATION_SEC = 600  # hard cap to protect the server


def _ffmpeg_bin() -> str:
    """Resolve the bundled ffmpeg binary (via imageio-ffmpeg)."""
    import imageio_ffmpeg  # type: ignore

    return imageio_ffmpeg.get_ffmpeg_exe()


def _decode_with_ffmpeg(data: bytes) -> np.ndarray:
    """Pipe raw bytes through ffmpeg → float32 mono 16k numpy array."""
    cmd = [
        _ffmpeg_bin(),
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd, input=data, capture_output=True, check=True, timeout=120
        )
    except subprocess.CalledProcessError as e:
        raise AudioDecodeError(
            f"ffmpeg failed to decode audio: {e.stderr.decode('utf-8', errors='replace')[:300]}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise AudioDecodeError("ffmpeg decode timed out") from e

    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise AudioDecodeError("ffmpeg produced no audio samples")
    return audio


def _decode_with_soundfile(data: bytes) -> np.ndarray:
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError) as e:
        raise AudioDecodeError(f"soundfile failed to decode: {e}") from e

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)

    if sr != TARGET_SR:
        audio = _resample(audio, sr, TARGET_SR)
    return audio


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear resample. Fine for speech; we don't need pristine fidelity."""
    if src_sr == dst_sr:
        return audio
    duration = audio.shape[0] / float(src_sr)
    new_len = round(duration * dst_sr)
    if new_len <= 0:
        raise AudioDecodeError("audio is empty after resample")
    src_idx = np.linspace(0, audio.shape[0] - 1, new_len, dtype=np.float64)
    left = np.floor(src_idx).astype(np.int64)
    right = np.minimum(left + 1, audio.shape[0] - 1)
    frac = (src_idx - left).astype(np.float32)
    return ((1.0 - frac) * audio[left] + frac * audio[right]).astype(np.float32)


def decode(data: bytes, declared_format: str | None = None, filename: str | None = None) -> np.ndarray:
    """Decode an uploaded audio payload into float32 mono 16k.

    declared_format:
      - "raw_f32_16k": bytes are float32 LE, already mono 16k. Skip decode.
      - None / anything else: try soundfile first (WAV/FLAC/OGG), then ffmpeg.

    Returns a 1-D numpy float32 array.
    """
    if not data:
        raise AudioDecodeError("empty audio payload")

    if declared_format == "raw_f32_16k":
        if len(data) % 4 != 0:
            raise AudioDecodeError("raw_f32_16k payload size is not a multiple of 4 bytes")
        return np.frombuffer(data, dtype=np.float32).copy()

    # Try soundfile first for common containers (WAV/FLAC/OGG); ffmpeg as fallback.
    try:
        return _decode_with_soundfile(data)
    except AudioDecodeError:
        pass

    ext = (Path(filename).suffix.lower().lstrip(".") if filename else "")
    if declared_format and declared_format not in {"wav", "flac", "ogg", "mp3", "m4a", "aac", "webm", "opus"}:
        raise UnsupportedFormatError(f"unsupported declared format: {declared_format}")

    audio = _decode_with_ffmpeg(data)
    _ = ext  # currently unused; kept for future format-specific tweaks
    return audio


def duration_seconds(audio: np.ndarray) -> float:
    return float(audio.shape[0]) / float(TARGET_SR)


def enforce_max_duration(audio: np.ndarray) -> None:
    if duration_seconds(audio) > MAX_DURATION_SEC:
        from whisper_wrapper.errors import AudioTooLongError  # local import to avoid cycle
        raise AudioTooLongError(
            f"audio exceeds {MAX_DURATION_SEC}s cap ({duration_seconds(audio):.1f}s)"
        )
