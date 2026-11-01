"""POST /v1/images/generations (text to image) and POST /v1/images/edits (instruction
editing with one or more reference images)."""
from __future__ import annotations

import base64
import json
import time

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from ..schemas import EditRequest, GenerateRequest, ImageRef
from ..service import ImageService, InvalidRequest, JobOutcome
from .deps import get_service, validation_message

router = APIRouter(tags=["images"])


def image_response(outcome: JobOutcome, response_format: str, public_url: str) -> dict:
    if response_format == "url":
        if not outcome.saved:
            raise InvalidRequest("The image could not be saved to output/, so there is no URL. "
                                 "Use response_format b64_json.")
        item = {"url": f"{public_url}/v1/gallery/{outcome.id}/image"}
    else:
        item = {"b64_json": base64.b64encode(outcome.png).decode("ascii")}
    body = {
        "created": int(time.time()),
        "data": [item],
        "id": outcome.id,
        "file": outcome.file,
        "model": outcome.model,
        "seed": outcome.seed,
        "size": outcome.size,
        "generation_seconds": outcome.seconds,
        "steps_ran": outcome.steps_ran,
    }
    if outcome.kind == "edit":
        body["reference_images"] = outcome.details.get("reference_images")
    return body


@router.post("/v1/images/generations")
def generate_image(req: GenerateRequest, request: Request) -> dict:
    service: ImageService = get_service(request)
    outcome = service.generate(
        prompt=req.prompt, size=req.size, model=req.model, steps=req.steps,
        true_cfg_scale=req.true_cfg_scale, seed=req.seed, negative_prompt=req.negative_prompt,
    )
    return image_response(outcome, req.response_format, service.settings.public_url)


# Form fields of a multipart edit request that map 1:1 onto EditRequest.
_FORM_FIELDS = ("prompt", "negative_prompt", "model", "size", "steps", "true_cfg_scale",
                "seed", "n", "response_format")


async def _parse_edit(request: Request, max_bytes: int) -> tuple[EditRequest, list]:
    """Returns the validated request and an ordered list of references, each either
    ("upload", bytes, filename) or ("ref", ImageRef)."""
    ctype = request.headers.get("content-type", "")
    if ctype.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        form = await request.form()
        fields = {k: form.get(k) for k in _FORM_FIELDS if form.get(k) not in (None, "")}
        refs: list = []
        # Uploaded files first, in the order sent (repeat `image`, or OpenAI's `image[]`),
        # then any `source` names (images already in output/).
        for value in [*form.getlist("image"), *form.getlist("image[]")]:
            if isinstance(value, UploadFile):
                refs.append(("upload", await value.read(max_bytes + 1), value.filename or "upload"))
            elif isinstance(value, str) and value:
                refs.append(("ref", ImageRef(data=value)))
        for name in form.getlist("source"):
            if isinstance(name, str) and name:
                refs.append(("ref", ImageRef(source=name)))
        # Satisfy the schema's "at least one image" rule; the real list is `refs`.
        if refs:
            fields["images"] = [{"source": "placeholder"}] * len(refs)
        try:
            req = EditRequest.model_validate(fields)
        except ValidationError as e:
            raise InvalidRequest(validation_message(e)) from None
        return req, refs
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise InvalidRequest("Send the edit as multipart/form-data (with 'image' files) "
                             "or as a JSON body.") from None
    try:
        req = EditRequest.model_validate(payload)
    except ValidationError as e:
        raise InvalidRequest(validation_message(e)) from None
    return req, [("ref", r) for r in req.images]


@router.post("/v1/images/edits")
async def edit_image(request: Request) -> dict:
    service: ImageService = get_service(request)
    req, refs = await _parse_edit(request, service.settings.max_upload_bytes)
    if len(refs) > service.settings.edit_max_images:
        raise InvalidRequest(f"{len(refs)} reference images given; the limit is "
                             f"{service.settings.edit_max_images} (edit.max_images in the config).")

    def run() -> dict:
        sources = []
        for n, ref in enumerate(refs, start=1):
            if ref[0] == "upload":
                sources.append(service.load_source(image_bytes=ref[1], label=f"{ref[2]}"))
            else:
                r: ImageRef = ref[1]
                sources.append(service.load_source(image_b64=r.data, source_name=r.source,
                                                   label=None if r.source else f"image {n}"))
        outcome = service.edit(
            prompt=req.prompt, sources=sources, size=req.size, model=req.model, steps=req.steps,
            true_cfg_scale=req.true_cfg_scale, seed=req.seed, negative_prompt=req.negative_prompt,
        )
        return image_response(outcome, req.response_format, service.settings.public_url)

    return await run_in_threadpool(run)
