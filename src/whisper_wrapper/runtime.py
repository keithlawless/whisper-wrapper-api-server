from __future__ import annotations

import faulthandler
import os
import threading


def enable_faulthandler() -> None:
    """Dump every thread's Python stack to stderr on a native fault.

    Native aborts inside CTranslate2 / PyTorch (heap corruption, access
    violations) otherwise kill the interpreter silently — the process just
    drops back to the shell prompt with no traceback. faulthandler installs
    OS-level handlers (SIGSEGV/SIGABRT/SIGFPE/SIGBUS/SIGILL on POSIX, plus the
    equivalent structured-exception handlers on Windows) that print where each
    thread was before the process dies, so the next crash leaves a clue about
    which worker was mid-inference. Idempotent.
    """
    if not faulthandler.is_enabled():
        faulthandler.enable()


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
