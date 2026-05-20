from __future__ import annotations

import platform


def detect_backend() -> str:
    """Return 'mlx-whisper' on Apple Silicon with mlx-whisper installed, else 'faster-whisper'."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401

            return "mlx-whisper"
        except ImportError:
            pass
    return "faster-whisper"
