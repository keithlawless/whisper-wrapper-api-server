from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from whisper_wrapper import API_VERSION, SERVICE_NAME, __version__
from whisper_wrapper.config import Settings
from whisper_wrapper.errors import AuthRequiredError, install_handlers
from whisper_wrapper.logging import configure as configure_logging
from whisper_wrapper.logging import get_logger, install_middleware
from whisper_wrapper.models import ModelManager
from whisper_wrapper.routes import admin as admin_routes
from whisper_wrapper.runtime import configure_tqdm_lock, enable_faulthandler
from whisper_wrapper.routes import health as health_routes
from whisper_wrapper.routes import transcribe as transcribe_routes


def create_app(settings: Settings) -> FastAPI:
    enable_faulthandler()
    configure_tqdm_lock()
    configure_logging(settings.log_level)
    log = get_logger("server")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mgr = ModelManager(settings)
        app.state.settings = settings
        app.state.model_manager = mgr
        log.info(
            "server_startup",
            service=SERVICE_NAME,
            version=__version__,
            api_version=API_VERSION,
            host=settings.host,
            port=settings.port,
            max_concurrent=mgr.effective_max_concurrent(),
            model_cache_dir=str(settings.model_cache_dir),
        )
        if settings.preload:
            await mgr.preload(list(settings.preload))
        yield
        log.info("server_shutdown")

    app = FastAPI(
        title="whisper-wrapper",
        version=__version__,
        lifespan=lifespan,
    )

    install_middleware(app)
    install_handlers(app)

    if not settings.is_local_only():
        # Bearer token guard for non-localhost binds. localhost binds remain open.
        @app.middleware("http")
        async def _auth_guard(request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            token = settings.auth_token
            if not token:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": {
                            "code": "auth_required",
                            "message": "server is bound to a non-loopback address but no auth token is configured",
                            "request_id": getattr(request.state, "request_id", None),
                        }
                    },
                )
            header = request.headers.get("authorization", "")
            if header != f"Bearer {token}":
                raise AuthRequiredError("invalid or missing bearer token")
            return await call_next(request)

    app.include_router(health_routes.router)
    app.include_router(transcribe_routes.router)
    app.include_router(admin_routes.router)
    return app
