"""GET /status (live progress and memory), GET /config (what the UI offers), GET /health."""
from __future__ import annotations

from fastapi import APIRouter, Request

from .deps import get_service

router = APIRouter(tags=["system"])


@router.get("/status")
def get_status(request: Request) -> dict:
    return get_service(request).status()


@router.get("/config")
def get_public_config(request: Request) -> dict:
    return get_service(request).public_config()


@router.get("/health")
def health() -> dict:
    """Cheap liveness check (no GPU query)."""
    return {"ok": True}
