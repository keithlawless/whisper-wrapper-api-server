from __future__ import annotations

import argparse
import sys
from pathlib import Path

from whisper_wrapper import API_VERSION, SERVICE_NAME, __version__
from whisper_wrapper.config import (
    SUPPORTED_MODELS,
    Settings,
    from_env,
)


def _parse_preload(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="whisper-wrapper", description="Local Whisper API server")
    sub = p.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--model-cache-dir", type=Path, default=None)
    serve.add_argument("--default-model", default=None, choices=SUPPORTED_MODELS)
    serve.add_argument("--preload", default=None, help="comma-separated model sizes to preload")
    serve.add_argument("--max-concurrent", type=int, default=None)
    serve.add_argument(
        "--compute-type",
        default=None,
        choices=["int8", "int8_float16", "int8_float32", "float16", "float32"],
    )
    serve.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    serve.add_argument("--auth-token", default=None)
    serve.add_argument(
        "--backend",
        default=None,
        choices=["auto", "faster-whisper", "mlx-whisper"],
    )
    serve.add_argument("--log-level", default=None, choices=["debug", "info", "warning", "error"])

    dl = sub.add_parser("download", help="Pre-download a model and exit")
    dl.add_argument("size", choices=SUPPORTED_MODELS)
    dl.add_argument("--model-cache-dir", type=Path, default=None)

    sub.add_parser("version", help="Print version info and exit")

    return p


def _build_settings(args: argparse.Namespace) -> Settings:
    env = from_env()
    overrides: dict = {}
    for key in (
        "host",
        "port",
        "model_cache_dir",
        "default_model",
        "max_concurrent",
        "compute_type",
        "device",
        "backend",
        "auth_token",
        "log_level",
    ):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    preload_cli = getattr(args, "preload", None)
    if preload_cli is not None:
        overrides["preload"] = _parse_preload(preload_cli)

    merged = {**env, **overrides}
    return Settings(**merged)


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = _build_settings(args)
    if not settings.is_local_only() and not settings.auth_token:
        print(
            f"error: --host={settings.host} requires --auth-token (or "
            "WHISPER_WRAPPER_AUTH_TOKEN) when not bound to loopback",
            file=sys.stderr,
        )
        return 2

    from whisper_wrapper.server import create_app

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        access_log=False,  # we log via the middleware
    )
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from faster_whisper import WhisperModel

    settings = _build_settings(args)
    cache_dir = settings.model_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.size} to {cache_dir} ...", file=sys.stderr)
    WhisperModel(args.size, device="cpu", compute_type="int8", download_root=str(cache_dir))
    print("done", file=sys.stderr)
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"{SERVICE_NAME} {__version__} (api {API_VERSION})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "download":
        return _cmd_download(args)
    if args.command == "version":
        return _cmd_version(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
