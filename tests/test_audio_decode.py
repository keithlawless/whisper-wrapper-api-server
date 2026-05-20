from __future__ import annotations

import numpy as np
import pytest

from whisper_wrapper import audio
from whisper_wrapper.errors import AudioDecodeError, AudioTooLongError


def test_decode_wav_bytes(speechlike_wav_bytes: bytes) -> None:
    arr = audio.decode(speechlike_wav_bytes, filename="x.wav")
    assert arr.dtype == np.float32
    assert arr.ndim == 1
    assert arr.size > 0
    # 2s @ 16k
    assert abs(audio.duration_seconds(arr) - 2.0) < 0.05


def test_decode_raw_f32_16k_roundtrip() -> None:
    src = np.linspace(-1.0, 1.0, 16_000, dtype=np.float32)  # 1s
    arr = audio.decode(src.tobytes(), declared_format="raw_f32_16k")
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, src, atol=0)


def test_decode_raw_f32_16k_misaligned() -> None:
    with pytest.raises(AudioDecodeError):
        audio.decode(b"abc", declared_format="raw_f32_16k")


def test_decode_empty() -> None:
    with pytest.raises(AudioDecodeError):
        audio.decode(b"")


def test_resample_path_changes_length() -> None:
    sr = 8000
    src = np.linspace(-1.0, 1.0, sr, dtype=np.float32)
    upsampled = audio._resample(src, sr, audio.TARGET_SR)
    assert upsampled.shape[0] == audio.TARGET_SR  # 1 second at 16k


def test_enforce_max_duration() -> None:
    too_long = np.zeros(audio.TARGET_SR * (audio.MAX_DURATION_SEC + 1), dtype=np.float32)
    with pytest.raises(AudioTooLongError):
        audio.enforce_max_duration(too_long)
