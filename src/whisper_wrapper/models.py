from __future__ import annotations

import asyncio
import os
from pathlib import Path

from whisper_wrapper.backends import detect_backend
from whisper_wrapper.backends.base import Transcriber
from whisper_wrapper.config import MODEL_ALIASES, SUPPORTED_MODELS, Settings
from whisper_wrapper.errors import ModelDownloadError, ModelNotAvailableError
from whisper_wrapper.logging import get_logger

log = get_logger("models")


class ModelManager:
    """Lazy-loads Whisper models and gates parallel transcriptions.

    Concurrency design:
      - One Transcriber per requested size, cached for the process lifetime.
      - Loads are single-flight via per-size asyncio.Lock so two parallel
        first-requests don't double-download/double-load.
      - The semaphore caps how many transcribe calls run simultaneously.
        The MLX backend (Metal) is not thread-safe, so concurrency is
        forced to 1 there to avoid native SIGTRAP crashes.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._active_backend = (
            detect_backend() if settings.backend == "auto" else settings.backend
        )
        self._models: dict[str, Transcriber] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(self.effective_max_concurrent())

    def effective_max_concurrent(self) -> int:
        """How many transcribe calls may run at once for the active backend.

        mlx-whisper drives Metal, which is not thread-safe: concurrent
        transcribe() calls from worker threads abort the process with a
        SIGTRAP ("trace trap"). Serialize them by capping at 1.
        """
        if self._active_backend == "mlx-whisper":
            return 1
        return self._settings.resolved_max_concurrent()

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    @property
    def cache_dir(self) -> Path:
        return self._settings.model_cache_dir

    def backend_name(self) -> str:
        return self._active_backend

    def device(self) -> str:
        if self._active_backend == "mlx-whisper":
            return "apple-silicon"
        d = self._settings.device
        return d if d != "auto" else "cpu"

    def compute_type(self) -> str:
        if self._active_backend == "mlx-whisper":
            return "float16"
        return self._settings.compute_type

    def supported(self) -> list[str]:
        return list(SUPPORTED_MODELS)

    def loaded(self) -> list[str]:
        return sorted(self._models.keys())

    def cached_on_disk(self) -> list[str]:
        if self._active_backend == "mlx-whisper":
            return self._cached_mlx()
        return self._cached_faster_whisper()

    def _cached_faster_whisper(self) -> list[str]:
        root = self._settings.model_cache_dir
        if not root.exists():
            return []
        hits: set[str] = set()
        for path in root.glob("models--*faster-whisper-*"):
            name = path.name
            for size in SUPPORTED_MODELS:
                if name.endswith(f"faster-whisper-{size}"):
                    hits.add(size)
                    break
        return sorted(hits)

    def _cached_mlx(self) -> list[str]:
        """Check HF hub cache for downloaded mlx-community whisper model snapshots."""
        hf_cache = (
            Path(os.environ["HF_HOME"])
            if "HF_HOME" in os.environ
            else Path.home() / ".cache" / "huggingface"
        ) / "hub"
        if not hf_cache.exists():
            return []
        hits: set[str] = set()
        for size in SUPPORTED_MODELS:
            # HF hub stores repos as models--<org>--<repo-name>
            if (hf_cache / f"models--mlx-community--whisper-{size}-mlx").exists():
                hits.add(size)
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

    async def get(self, size: str) -> Transcriber:
        """Return a loaded Transcriber for `size`, loading if needed."""
        size = MODEL_ALIASES.get(size, size)
        self._validate(size)
        existing = self._models.get(size)
        if existing is not None:
            return existing

        lock = self._lock_for(size)
        async with lock:
            existing = self._models.get(size)
            if existing is not None:
                return existing
            transcriber = await asyncio.to_thread(self._load_sync, size)
            self._models[size] = transcriber
            return transcriber

    def _load_sync(self, size: str) -> Transcriber:
        log.info(
            "model_load_start",
            size=size,
            backend=self._active_backend,
            device=self.device(),
            compute_type=self.compute_type(),
        )
        try:
            transcriber = self._load_backend(size)
        except Exception as e:
            log.error("model_load_failed", size=size, backend=self._active_backend, reason=str(e))
            raise ModelDownloadError(f"failed to load model '{size}': {e}") from e
        log.info("model_load_done", size=size, backend=self._active_backend)
        return transcriber

    def _load_backend(self, size: str) -> Transcriber:
        if self._active_backend == "mlx-whisper":
            return self._load_mlx(size)
        return self._load_faster_whisper(size)

    def _load_faster_whisper(self, size: str) -> Transcriber:
        from faster_whisper import WhisperModel

        from whisper_wrapper.backends.faster_whisper_backend import FasterWhisperTranscriber

        cache_dir = self._settings.model_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        model = WhisperModel(
            size,
            device=self._settings.device if self._settings.device != "auto" else "cpu",
            compute_type=self._settings.compute_type,
            download_root=str(cache_dir),
        )
        return FasterWhisperTranscriber(model)

    def _load_mlx(self, size: str) -> Transcriber:
        from whisper_wrapper.backends.mlx_backend import MLX_REPOS, MlxTranscriber

        repo = MLX_REPOS.get(size)
        if repo is None:
            raise ModelDownloadError(f"no MLX model repo configured for size '{size}'")
        # Download model weights to HF hub cache now so the first transcribe is fast.
        try:
            import mlx.core as mx  # type: ignore
            import mlx_whisper.load_models as lm  # type: ignore

            lm.load_model(repo, dtype=mx.float16)
        except Exception as e:
            raise ModelDownloadError(
                f"failed to load MLX model '{size}' from {repo}: {e}"
            ) from e
        return MlxTranscriber(repo)

    async def preload(self, sizes: list[str]) -> None:
        for size in sizes:
            try:
                await self.get(size)
            except Exception as e:
                log.warning("preload_failed", size=size, reason=str(e))
