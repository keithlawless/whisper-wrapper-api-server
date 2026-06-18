from __future__ import annotations

import pytest

from whisper_wrapper.config import Settings
from whisper_wrapper.models import ModelManager


def _mgr(tmp_model_cache, monkeypatch, *, cuda: bool, **settings_kwargs) -> ModelManager:
    monkeypatch.setattr(ModelManager, "_cuda_available", staticmethod(lambda: cuda))
    settings = Settings(
        model_cache_dir=tmp_model_cache,
        backend="faster-whisper",
        **settings_kwargs,
    )
    return ModelManager(settings)


def test_auto_device_selects_cuda_when_available(tmp_model_cache, monkeypatch) -> None:
    mgr = _mgr(tmp_model_cache, monkeypatch, cuda=True, device="auto")
    assert mgr.device() == "cuda"


def test_auto_device_falls_back_to_cpu_without_gpu(tmp_model_cache, monkeypatch) -> None:
    mgr = _mgr(tmp_model_cache, monkeypatch, cuda=False, device="auto")
    assert mgr.device() == "cpu"


def test_explicit_cuda_device_is_honored(tmp_model_cache, monkeypatch) -> None:
    # Even if detection would say no GPU, an explicit --device cuda is honored
    # (and surfaces a clear error later if the runtime is actually missing).
    mgr = _mgr(tmp_model_cache, monkeypatch, cuda=False, device="cuda")
    assert mgr.device() == "cuda"


def test_device_resolution_is_cached(tmp_model_cache, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_detect() -> bool:
        calls["n"] += 1
        return True

    monkeypatch.setattr(ModelManager, "_cuda_available", staticmethod(fake_detect))
    mgr = ModelManager(Settings(model_cache_dir=tmp_model_cache, backend="faster-whisper", device="auto"))
    assert mgr.device() == "cuda"
    assert mgr.device() == "cuda"
    assert calls["n"] == 1


@pytest.mark.parametrize("ct", ["float16", "int8_float16"])
def test_float16_compute_downgrades_on_cpu(tmp_model_cache, monkeypatch, ct) -> None:
    mgr = _mgr(tmp_model_cache, monkeypatch, cuda=False, device="cpu", compute_type=ct)
    assert mgr.compute_type() == "int8"


@pytest.mark.parametrize("ct", ["float16", "int8_float16"])
def test_float16_compute_preserved_on_cuda(tmp_model_cache, monkeypatch, ct) -> None:
    mgr = _mgr(tmp_model_cache, monkeypatch, cuda=True, device="cuda", compute_type=ct)
    assert mgr.compute_type() == ct


def test_int8_compute_preserved_on_cpu(tmp_model_cache, monkeypatch) -> None:
    mgr = _mgr(tmp_model_cache, monkeypatch, cuda=False, device="cpu", compute_type="int8")
    assert mgr.compute_type() == "int8"


def test_cuda_faster_whisper_still_serialized(tmp_model_cache, monkeypatch) -> None:
    """GPU does not relax serialization: concurrent CUDA calls on one model
    crash even harder than CPU, so in-flight transcriptions stay capped at 1."""
    mgr = _mgr(tmp_model_cache, monkeypatch, cuda=True, device="cuda", max_concurrent=4)
    assert mgr.effective_max_concurrent() == 1
    assert mgr.semaphore._value == 1


def test_cuda_unavailable_detection_swallows_errors(monkeypatch) -> None:
    """_cuda_available must never propagate import/runtime errors."""
    import builtins

    real_import = builtins.__import__

    def boom(name, *args, **kwargs):
        if name == "ctranslate2":
            raise ImportError("no ctranslate2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert ModelManager._cuda_available() is False
