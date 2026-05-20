from __future__ import annotations

import asyncio
from pathlib import Path

from whisper_wrapper.config import SUPPORTED_MODELS, Settings
from whisper_wrapper.errors import ModelDownloadError, ModelNotAvailableError
from whisper_wrapper.logging import get_logger

log = get_logger("models")


class ModelManager:
    """Lazy-loads faster-whisper models and gates parallel transcriptions.

    Concurrency design:
      - One WhisperModel per requested size, cached for the process lifetime.
      - Loads are single-flight via per-size asyncio.Lock so two parallel
        first-requests don't double-download/double-load.
      - The semaphore caps how many transcribe calls run simultaneously.
        CTranslate2 itself can use multiple threads per call (set internally),
        so this prevents oversubscription.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._models: dict[str, object] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(settings.resolved_max_concurrent())

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    @property
    def cache_dir(self) -> Path:
        return self._settings.model_cache_dir

    def device(self) -> str:
        d = self._settings.device
        return d if d != "auto" else "cpu"

    def compute_type(self) -> str:
        return self._settings.compute_type

    def supported(self) -> list[str]:
        return list(SUPPORTED_MODELS)

    def loaded(self) -> list[str]:
        return sorted(self._models.keys())

    def cached_on_disk(self) -> list[str]:
        """Best-effort: list sizes that have a snapshot directory under HF cache."""
        root = self._settings.model_cache_dir
        if not root.exists():
            return []
        hits: set[str] = set()
        # Hugging Face cache layout: models--Systran--faster-whisper-<size>
        for path in root.glob("models--*faster-whisper-*"):
            name = path.name
            # extract the size suffix
            for size in SUPPORTED_MODELS:
                if name.endswith(f"faster-whisper-{size}"):
                    hits.add(size)
                    break
        return sorted(hits)

    def _lock_for(self, size: str) -> asyncio.Lock:
        lock = self._locks.get(size)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[size] = lock
        return lock

    def _validate(self, size: str) -> None:
        if size not in SUPPORTED_MODELS:
            raise ModelNotAvailableError(
                f"unsupported model size '{size}'. supported: {', '.join(SUPPORTED_MODELS)}"
            )

    async def get(self, size: str):
        """Return a loaded faster_whisper.WhisperModel for `size`, loading if needed."""
        self._validate(size)
        existing = self._models.get(size)
        if existing is not None:
            return existing

        lock = self._lock_for(size)
        async with lock:
            existing = self._models.get(size)
            if existing is not None:
                return existing
            model = await asyncio.to_thread(self._load_sync, size)
            self._models[size] = model
            return model

    def _load_sync(self, size: str):
        from faster_whisper import WhisperModel  # imported lazily for fast CLI startup

        cache_dir = self._settings.model_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "model_load_start",
            size=size,
            device=self.device(),
            compute_type=self.compute_type(),
            cache_dir=str(cache_dir),
        )
        try:
            model = WhisperModel(
                size,
                device=self.device(),
                compute_type=self.compute_type(),
                download_root=str(cache_dir),
            )
        except Exception as e:
            log.error("model_load_failed", size=size, reason=str(e))
            raise ModelDownloadError(f"failed to load model '{size}': {e}") from e
        log.info("model_load_done", size=size)
        return model

    async def preload(self, sizes: list[str]) -> None:
        for size in sizes:
            try:
                await self.get(size)
            except Exception as e:
                log.warning("preload_failed", size=size, reason=str(e))
