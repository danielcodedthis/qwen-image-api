"""Browse output/: list images, fetch one image, its thumbnail, an edit's reference images and its log."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response

from ..service import InvalidRequest, NotFound
from ..storage import InvalidName, OutputStore
from .deps import get_service

router = APIRouter(prefix="/v1/gallery", tags=["gallery"])

# Generated files never change once written; let the browser keep them.
_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def _store(request: Request) -> OutputStore:
    return get_service(request).store


def _existing(store: OutputStore, stem: str) -> str:
    try:
        if not store.exists(stem):
            raise NotFound(f"No image named '{stem}.png' in output/.")
    except InvalidName as e:
        raise InvalidRequest(str(e)) from None
    return stem


@router.get("")
def list_gallery(request: Request,
                 offset: int = Query(0, ge=0),
                 limit: int = Query(60, ge=1, le=500)) -> dict:
    items, total = _store(request).list_images(offset=offset, limit=limit)
    for it in items:
        base = f"/v1/gallery/{it['id']}"
        it["image_url"] = f"{base}/image"
        it["thumb_url"] = f"{base}/thumb"
    return {"total": total, "offset": offset, "limit": limit, "items": items}


@router.get("/{stem}")
def gallery_item(stem: str, request: Request) -> dict:
    store = _store(request)
    _existing(store, stem)
    record = store.read_log(stem)
    base = f"/v1/gallery/{stem}"
    return {
        "id": stem,
        "file": f"{stem}.png",
        "image_url": f"{base}/image",
        "thumb_url": f"{base}/thumb",
        "source_urls": [f"{base}/source/{n}" for n in range(1, store.source_count(stem) + 1)],
        "log_url": f"{base}/log" if record else None,
        "fields": record.fields if record else {},
        "prompt": record.prompt if record else None,
        "negative_prompt": record.negative_prompt if record else None,
        "log": record.text if record else None,
    }


@router.get("/{stem}/image")
def gallery_image(stem: str, request: Request):
    store = _store(request)
    _existing(store, stem)
    return FileResponse(store.image_path(stem), media_type="image/png", headers=_IMMUTABLE)


@router.get("/{stem}/thumb")
def gallery_thumb(stem: str, request: Request):
    store = _store(request)
    _existing(store, stem)
    return Response(store.thumbnail(stem), media_type="image/jpeg", headers=_IMMUTABLE)


@router.get("/{stem}/source/{n}")
def gallery_source(stem: str, n: int, request: Request):
    """Reference image n (1-based) of an edit."""
    store = _store(request)
    _existing(store, stem)
    path = store.source_path(stem, n) if n >= 1 else None
    if path is None or not path.is_file():
        raise NotFound(f"'{stem}' has no reference image {n}.")
    return FileResponse(path, media_type="image/png", headers=_IMMUTABLE)


@router.get("/{stem}/log", response_class=PlainTextResponse)
def gallery_log(stem: str, request: Request):
    store = _store(request)
    _existing(store, stem)
    record = store.read_log(stem)
    if record is None:
        raise NotFound(f"No log for '{stem}' (output/logs/{stem}.log).")
    return PlainTextResponse(record.text)
