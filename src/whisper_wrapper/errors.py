from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class WhisperWrapperError(Exception):
    code = "internal_error"
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AudioDecodeError(WhisperWrapperError):
    code = "audio_decode_failed"
    http_status = status.HTTP_400_BAD_REQUEST


class UnsupportedFormatError(WhisperWrapperError):
    code = "unsupported_format"
    http_status = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


class ModelNotAvailableError(WhisperWrapperError):
    code = "model_not_available"
    http_status = status.HTTP_400_BAD_REQUEST


class ModelDownloadError(WhisperWrapperError):
    code = "model_download_failed"
    http_status = status.HTTP_503_SERVICE_UNAVAILABLE


class AudioTooLongError(WhisperWrapperError):
    code = "audio_too_long"
    http_status = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


class AuthRequiredError(WhisperWrapperError):
    code = "auth_required"
    http_status = status.HTTP_401_UNAUTHORIZED


def _envelope(code: str, message: str, request_id: str | None) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(WhisperWrapperError)
    async def _handle_known(request: Request, exc: WhisperWrapperError) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=exc.http_status,
            content=_envelope(exc.code, exc.message, rid),
        )

    @app.exception_handler(Exception)
    async def _handle_unknown(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        # Don't leak internals; structured log captures the traceback elsewhere.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("internal_error", str(exc) or "internal error", rid),
        )
