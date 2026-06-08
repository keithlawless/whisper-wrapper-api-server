from __future__ import annotations

import os
import threading


def configure_tqdm_lock() -> None:
    """Stop tqdm from creating a multiprocessing semaphore.

    huggingface_hub / mlx_whisper render a "Fetching N files" tqdm progress bar
    during snapshot_download (which mlx_whisper calls on every transcribe). On
    first use tqdm lazily creates a global ``multiprocessing.RLock()`` — a named
    POSIX semaphore — to coordinate bars across processes. It never releases it,
    so the interpreter prints "leaked semaphore objects to clean up at shutdown"
    when the process exits.

    We never run progress bars across processes, so swap in a plain thread lock.
    Idempotent: set_lock just replaces tqdm's class-level lock. Also silence HF
    progress bars outright — this is a daemon, the bars are only log noise.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import tqdm
    except Exception:
        return
    tqdm.tqdm.set_lock(threading.RLock())
