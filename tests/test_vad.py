from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from whisper_wrapper import vad as vad_mod

pytest.importorskip("torch")


def test_apply_serializes_inference(monkeypatch) -> None:
    """Concurrent apply() calls must never run the Silero model reentrantly.

    Regression for the native "pointer being freed was not allocated" malloc
    abort: the shared Silero JIT model is not thread-safe, and apply() runs
    under asyncio.to_thread for every request, so overlapping calls are normal.
    """
    concurrent = 0
    max_concurrent = 0
    seen_lock = threading.Lock()

    def fake_get_speech_timestamps(_tensor, _model, **_kwargs):
        nonlocal concurrent, max_concurrent
        with seen_lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        # Linger so any unserialized caller would overlap here.
        time.sleep(0.05)
        with seen_lock:
            concurrent -= 1
        # No timestamps -> drives the all-silence branch (returns audio[:0]).
        return []

    # Pretend VAD loaded successfully with a sentinel model.
    monkeypatch.setattr(vad_mod, "_model", object())
    monkeypatch.setattr(vad_mod, "_load_failed", False)
    monkeypatch.setattr(vad_mod, "_get_speech_timestamps", fake_get_speech_timestamps)
    monkeypatch.setattr(vad_mod, "_load", lambda: None)

    audio = np.zeros(16000, dtype=np.float32)
    errors: list[BaseException] = []

    def worker():
        try:
            vad_mod.apply(audio)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    assert max_concurrent == 1, f"inference ran {max_concurrent}-way concurrent"
