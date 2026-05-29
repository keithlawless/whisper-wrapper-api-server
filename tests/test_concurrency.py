from __future__ import annotations

import asyncio

import pytest

from whisper_wrapper.config import Settings
from whisper_wrapper.models import ModelManager


@pytest.mark.asyncio
async def test_semaphore_caps_in_flight(tmp_model_cache) -> None:
    settings = Settings(
        model_cache_dir=tmp_model_cache,
        max_concurrent=2,
        compute_type="int8",
        device="cpu",
    )
    mgr = ModelManager(settings)

    in_flight = 0
    peak = 0
    started = asyncio.Event()

    async def worker():
        nonlocal in_flight, peak
        async with mgr.semaphore:
            in_flight += 1
            peak = max(peak, in_flight)
            started.set()
            await asyncio.sleep(0.05)
            in_flight -= 1

    await asyncio.gather(*[worker() for _ in range(8)])
    assert peak <= 2


def test_mlx_backend_forces_serial_transcription(tmp_model_cache) -> None:
    """MLX/Metal is not thread-safe; concurrency must be capped at 1.

    Regression for the SIGTRAP ("trace trap") crash when two transcribe
    requests overlapped on the mlx-whisper backend.
    """
    settings = Settings(
        model_cache_dir=tmp_model_cache,
        backend="mlx-whisper",
        max_concurrent=4,
    )
    mgr = ModelManager(settings)
    assert mgr.effective_max_concurrent() == 1
    assert mgr.semaphore._value == 1


def test_faster_whisper_honors_max_concurrent(tmp_model_cache) -> None:
    settings = Settings(
        model_cache_dir=tmp_model_cache,
        backend="faster-whisper",
        max_concurrent=3,
    )
    mgr = ModelManager(settings)
    assert mgr.effective_max_concurrent() == 3


@pytest.mark.asyncio
async def test_model_manager_single_flight_lock(tmp_model_cache, monkeypatch) -> None:
    """Two concurrent get() calls for the same size must load the model once."""
    settings = Settings(model_cache_dir=tmp_model_cache, device="cpu", compute_type="int8")
    mgr = ModelManager(settings)

    call_count = 0

    def fake_load(size: str):
        nonlocal call_count
        call_count += 1
        # Simulate a slow load so the second call has time to enter the lock.
        import time

        time.sleep(0.1)
        return object()

    monkeypatch.setattr(mgr, "_load_sync", fake_load)

    a, b = await asyncio.gather(mgr.get("tiny"), mgr.get("tiny"))
    assert a is b
    assert call_count == 1
