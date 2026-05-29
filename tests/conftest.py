from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from whisper_wrapper.audio import TARGET_SR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def speechlike_wav_bytes() -> bytes:
    """Synthesize a short speech-band signal as WAV bytes.

    Not real speech — used for audio-decode / shape / VAD-applied tests.
    Tests that need actual recognition should use a real fixture WAV
    (committed under tests/fixtures/) and the tiny model.
    """
    duration = 2.0
    t = np.linspace(0, duration, int(TARGET_SR * duration), endpoint=False, dtype=np.float32)
    # 200Hz fundamental + harmonics, amplitude-modulated to look speechy.
    sig = 0.3 * (
        np.sin(2 * np.pi * 200 * t)
        + 0.5 * np.sin(2 * np.pi * 400 * t)
        + 0.25 * np.sin(2 * np.pi * 800 * t)
    )
    sig *= 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    buf = io.BytesIO()
    sf.write(buf, sig.astype(np.float32), TARGET_SR, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@pytest.fixture(scope="session")
def silence_wav_bytes() -> bytes:
    duration = 1.0
    samples = np.zeros(int(TARGET_SR * duration), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, TARGET_SR, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@pytest.fixture(scope="session")
def real_fixture_wav(fixtures_dir: Path) -> Path | None:
    """Return path to tests/fixtures/hello.wav if present, else None.

    Real-transcription tests skip when this fixture is missing.
    Drop a short (~2s) 16kHz mono WAV at tests/fixtures/hello.wav
    to enable end-to-end recognition tests.
    """
    p = fixtures_dir / "hello.wav"
    return p if p.exists() else None


@pytest.fixture
def tmp_model_cache(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("model-cache")
    return d


@pytest.fixture
def test_settings(tmp_model_cache: Path):
    from whisper_wrapper.config import Settings

    return Settings(
        host="127.0.0.1",
        port=0,
        model_cache_dir=tmp_model_cache,
        default_model="tiny",
        preload=(),
        max_concurrent=2,
        compute_type="int8",
        device="cpu",
        # Pin the backend so tests are deterministic regardless of host
        # hardware (auto-detect picks mlx-whisper on Apple Silicon).
        backend="faster-whisper",
        log_level="warning",
    )


@pytest.fixture
def test_app(test_settings):
    from whisper_wrapper.server import create_app

    return create_app(test_settings)


@pytest.fixture
async def client(test_app):
    from httpx import ASGITransport, AsyncClient

    async with (
        test_app.router.lifespan_context(test_app),
        AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c,
    ):
        yield c


def pytest_configure(config):
    # Reduce HuggingFace download chattiness in CI logs.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
