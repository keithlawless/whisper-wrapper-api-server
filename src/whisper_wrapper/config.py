from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_data_dir

from whisper_wrapper import SERVICE_NAME

SUPPORTED_MODELS = ("tiny", "base", "small", "medium", "large-v3")
DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"


def _default_model_cache_dir() -> Path:
    return Path(user_data_dir(SERVICE_NAME, appauthor=False)) / "models"


@dataclass
class Settings:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    model_cache_dir: Path = field(default_factory=_default_model_cache_dir)
    default_model: str = "base"
    preload: tuple[str, ...] = ()
    max_concurrent: int = 0  # 0 = auto (min(cpu, 4))
    compute_type: str = "int8"
    device: str = "auto"
    auth_token: str | None = None
    log_level: str = "info"

    def resolved_max_concurrent(self) -> int:
        if self.max_concurrent > 0:
            return self.max_concurrent
        cpu = os.cpu_count() or 2
        return max(1, min(cpu, 4))

    def is_local_only(self) -> bool:
        return self.host in {"127.0.0.1", "localhost", "::1"}


def from_env() -> dict:
    """Read WHISPER_WRAPPER_* env vars into a dict of override kwargs."""
    out: dict = {}
    prefix = "WHISPER_WRAPPER_"
    mapping = {
        "HOST": ("host", str),
        "PORT": ("port", int),
        "MODEL_CACHE_DIR": ("model_cache_dir", Path),
        "DEFAULT_MODEL": ("default_model", str),
        "PRELOAD": ("preload", lambda s: tuple(x.strip() for x in s.split(",") if x.strip())),
        "MAX_CONCURRENT": ("max_concurrent", int),
        "COMPUTE_TYPE": ("compute_type", str),
        "DEVICE": ("device", str),
        "AUTH_TOKEN": ("auth_token", str),
        "LOG_LEVEL": ("log_level", str),
    }
    for env_name, (field_name, caster) in mapping.items():
        raw = os.environ.get(prefix + env_name)
        if raw is None or raw == "":
            continue
        out[field_name] = caster(raw)
    return out
