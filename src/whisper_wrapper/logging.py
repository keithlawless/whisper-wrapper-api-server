from __future__ import annotations

import logging
import sys
from collections.abc import Awaitable, Callable

import structlog
import ulid
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


def configure(level: str = "info") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request_id, binds it to structlog contextvars, logs request/response."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = str(ulid.new())
        request.state.request_id = rid
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=rid,
            method=request.method,
            path=request.url.path,
        )
        log = get_logger("http")
        log.info("request_start")
        try:
            response = await call_next(request)
        except Exception:
            log.exception("request_error")
            raise
        log.info("request_end", status_code=response.status_code)
        response.headers["x-request-id"] = rid
        return response


def install_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestContextMiddleware)
