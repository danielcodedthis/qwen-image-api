"""Shared helpers for routes: getting the service, turning errors into responses."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..service import ImageService, ServiceError


def get_service(request: Request) -> ImageService:
    return request.app.state.service


def validation_message(err: ValidationError | RequestValidationError) -> str:
    """'steps: must be <= 200; prompt: field required' — readable in the UI and by agents."""
    parts = []
    for e in err.errors():
        loc = [str(p) for p in e.get("loc", ()) if p not in ("body", "query", "path")]
        msg = e.get("msg", "invalid").removeprefix("Value error, ")
        parts.append(f"{'.'.join(loc)}: {msg}" if loc else msg)
    return "; ".join(parts) or "Invalid request."


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    # README promises 400 for bad input; FastAPI's default would be 422.
    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"detail": validation_message(exc)})
