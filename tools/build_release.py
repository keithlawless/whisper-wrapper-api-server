"""Build a whisper-wrapper release binary for the current platform.

Invoked from .gitlab-ci.yml. Produces dist/whisper-wrapper(.exe) and copies
it to dist/whisper-wrapper-<os>-<arch>(.exe) for artifact upload.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        os_tag = "macos"
    elif system == "linux":
        os_tag = "linux"
    elif system == "windows":
        os_tag = "windows"
    else:
        os_tag = system

    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    arch = arch_map.get(machine, machine)
    return f"{os_tag}-{arch}"


def main() -> int:
    spec = ROOT / "whisper_wrapper.spec"
    cmd = [sys.executable, "-m", "PyInstaller", "--clean", "-y", str(spec)]
    print("→", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        return result.returncode

    suffix = ".exe" if platform.system() == "Windows" else ""
    src = ROOT / "dist" / f"whisper-wrapper{suffix}"
    if not src.exists():
        print(f"error: expected build output not found at {src}", file=sys.stderr)
        return 1

    dst = ROOT / "dist" / f"whisper-wrapper-{_platform_tag()}{suffix}"
    shutil.copy2(src, dst)
    print(f"built {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
