"""Embedded MCP server: lets agents generate and edit images and check status.

Served by the same FastAPI app at `POST /mcp` (MCP streamable HTTP, stateless). The
base container answers on http://localhost:8000/mcp, Turbo on http://localhost:8001/mcp.
Tools call the same ImageService as the HTTP routes, so the one-job-at-a-time rule,
validation, logs and output/ files are shared.

Generations take a while, so `generate_image` / `edit_image` send MCP progress
notifications while they wait, and also accept `wait=false` to return a job id at
once (poll `get_status`, then fetch the result with `get_image`).
"""
from __future__ import annotations

import base64
import io
import json
from typing import Annotated, Any

import anyio
from fastapi import FastAPI
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ImageContent, TextContent, ToolAnnotations
from PIL import Image
from pydantic import Field

from .imaging import flatten
from .monitoring import log_buffer
from .service import ImageService, JobOutcome, ServiceError
from .storage import InvalidName, stem_from_name

MCP_PATH = "/mcp"
PROGRESS_INTERVAL = 2.0  # seconds between progress notifications
PREVIEW_MAX_SIDE = 512


def instructions(service: ImageService) -> str:
    s = service.settings
    model = s.model(s.default_model)
    steps = (f"always {model.fixed_steps} steps (saved schedule)" if model.fixed_steps
             else f"{model.total_steps} steps by default")
    return f"""\
Local {model.label} server on one AMD GPU ({steps}).
Qwen-Image-2.1 is a unified generation AND editing model.
- generate_image: text to image. Renders text in images well; put exact words in quotes.
- edit_image: instruction editing ("Change the background to a sunset beach", "Make the
  shirt red", "Remove the person on the left"). Takes 1-{s.edit_max_images} reference images
  (ids of images in output/, or base64), referred to in the prompt as Picture 1, 2, ...
  in the order given. Without a size, the output follows the last image's aspect ratio.
- Only one job runs at a time; a second one fails with "busy". Check get_status first.
- Jobs take from tens of seconds to minutes. With wait=false you get an id at once;
  poll get_status until busy is false, then call get_image(id).
- Results are saved in output/ (image_url is a link to the PNG on this server).
"""


Prompt = Annotated[str, Field(description="What to draw (generate) or what to change (edit).")]
Size = Annotated[str, Field(description="WIDTHxHEIGHT, multiples of 32, e.g. 1024x1024. "
                                        "get_options lists sizes and the pixel cap.")]
Model = Annotated[str | None, Field(description="Model id from get_options. Default: this server's model.")]
Steps = Annotated[int | None, Field(ge=1, le=200, description="Denoising steps. Default: config. "
                                                              "Not allowed for Turbo (fixed 8-step schedule).")]
Cfg = Annotated[float | None, Field(ge=1, le=10, description="True CFG scale; 1.0 = off (recommended "
                                                              "for Qwen-Image-2.1). >1 doubles the time.")]
Seed = Annotated[int | None, Field(ge=0, le=2**63 - 1, description="Fixed seed for reproducibility.")]
Negative = Annotated[str | None, Field(description="Negative prompt; only used when true_cfg_scale > 1.")]
Wait = Annotated[bool, Field(description="true: wait for the image. "
                                         "false: return a job id at once and poll get_status.")]
IncludeImage = Annotated[bool, Field(description="Also return a small JPEG preview "
                                                 "(only useful for clients that can see images).")]


def _preview(png: bytes) -> ImageContent:
    with Image.open(io.BytesIO(png)) as im:
        im = flatten(im)
        im.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return ImageContent(type="image", data=base64.b64encode(buf.getvalue()).decode(), mime_type="image/jpeg")


def build_mcp(service: ImageService) -> MCPServer:
    settings = service.settings
    mcp = MCPServer(name=settings.mcp_name, title=settings.model(settings.default_model).label,
                    instructions=instructions(service), version="1.0.0", log_level="WARNING")
    public = settings.public_url

    def links(stem: str) -> dict:
        return {
            "id": stem,
            "file": f"output/{stem}.png",
            "image_url": f"{public}/v1/gallery/{stem}/image",
            "log_url": f"{public}/v1/gallery/{stem}/log",
        }

    def outcome_json(o: JobOutcome) -> dict:
        out = {"status": "success", **links(o.id), "saved": o.saved, "kind": o.kind, "model": o.model,
               "size": o.size, "seed": o.seed, "seconds": o.seconds, "steps_ran": o.steps_ran}
        if o.kind == "edit":
            out["reference_images"] = o.details.get("reference_images")
        if not o.saved:
            out["warning"] = "The image could not be saved to output/ (see server log)."
        return out

    async def report_while_running(ctx: Context, done: anyio.Event) -> None:
        while not done.is_set():
            p = service.monitor.snapshot()
            if p.active:
                msg = f"{p.phase} step {p.step}/{p.total_steps}" if p.total_steps else p.phase
                try:
                    await ctx.report_progress(p.step, p.total_steps or None, msg)
                except Exception:  # noqa: BLE001 - progress is best effort
                    return
            with anyio.move_on_after(PROGRESS_INTERVAL):
                await done.wait()

    async def run_job(ctx: Context, fn, include_image: bool) -> list:
        """Run a blocking service call in a worker thread, reporting progress meanwhile."""
        done = anyio.Event()
        result: dict[str, Any] = {}

        def call():
            try:
                result["value"] = fn()
            except BaseException as e:  # noqa: BLE001
                result["error"] = e

        async with anyio.create_task_group() as tg:
            tg.start_soon(report_while_running, ctx, done)
            await anyio.to_thread.run_sync(call)
            done.set()

        if "error" in result:
            e = result["error"]
            if isinstance(e, ServiceError):
                raise ToolError(str(e))
            raise ToolError(f"{type(e).__name__}: {e}. See get_status logs or the image's log.")
        value = result["value"]
        if isinstance(value, str):  # wait=false: a job id
            return [TextContent(type="text", text=json.dumps({
                "status": "running", **links(value),
                "next": "Poll get_status until busy is false, then call get_image with this id."}))]
        content: list = [TextContent(type="text", text=json.dumps(outcome_json(value)))]
        if include_image:
            content.append(_preview(value.png))
        return content

    # ---------------------------------------------------------------- tools
    @mcp.tool(title="Generate an image", annotations=ToolAnnotations(
        read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False))
    async def generate_image(ctx: Context, prompt: Prompt, size: Size = "1024x1024", model: Model = None,
                             steps: Steps = None, true_cfg_scale: Cfg = None, seed: Seed = None,
                             negative_prompt: Negative = None, wait: Wait = True,
                             include_image: IncludeImage = False) -> list[TextContent | ImageContent]:
        """Generate an image from a text prompt. Saves it to output/ and returns its id,
        file path and URL."""
        if not prompt.strip():
            raise ToolError("prompt must not be empty.")
        return await run_job(ctx, lambda: service.generate(
            prompt=prompt, size=size, model=model, steps=steps, true_cfg_scale=true_cfg_scale,
            seed=seed, negative_prompt=negative_prompt, wait=wait), include_image)

    @mcp.tool(title="Edit an image", annotations=ToolAnnotations(
        read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False))
    async def edit_image(
        ctx: Context,
        prompt: Annotated[str, Field(description=(
            "The edit instruction, e.g. 'Change the background to a sunset beach'. With several "
            "images, refer to them as Picture 1, Picture 2, ... in the order given."))],
        sources: Annotated[list[str] | None, Field(description=(
            "Ids or file names of images in output/ (from generate_image or list_images), in order."))] = None,
        images_base64: Annotated[list[str] | None, Field(description=(
            "Images as base64 or data URLs, for images not in output/. They come after `sources`."))] = None,
        size: Annotated[str | None, Field(description=(
            "Output WIDTHxHEIGHT (multiples of 32), or omit to follow the last image's aspect ratio."))] = None,
        model: Model = None, steps: Steps = None, true_cfg_scale: Cfg = None, seed: Seed = None,
        negative_prompt: Negative = None, wait: Wait = True, include_image: IncludeImage = False,
    ) -> list[TextContent | ImageContent]:
        """Edit one or more reference images with an instruction (Qwen-Image-2.1 native
        editing): change backgrounds, colours, clothing, text, style; remove or add objects;
        combine people or products from several images."""
        refs = [("source", s) for s in sources or []] + [("data", d) for d in images_base64 or []]
        if not refs:
            raise ToolError("Give at least one image: 'sources' (output/ ids) or 'images_base64'.")
        if len(refs) > settings.edit_max_images:
            raise ToolError(f"{len(refs)} images given; this server allows {settings.edit_max_images}.")
        if not prompt.strip():
            raise ToolError("prompt must not be empty.")

        def job():
            loaded = [service.load_source(source_name=v, label=None) if k == "source"
                      else service.load_source(image_b64=v, label=f"image {i}")
                      for i, (k, v) in enumerate(refs, start=1)]
            return service.edit(prompt=prompt, sources=loaded, size=size, model=model, steps=steps,
                                true_cfg_scale=true_cfg_scale, seed=seed, negative_prompt=negative_prompt,
                                wait=wait)

        return await run_job(ctx, job, include_image)

    @mcp.tool(title="Server status", structured_output=True,
              annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def get_status(log_lines: Annotated[int, Field(ge=0, le=200, description="Recent log lines.")] = 10
                   ) -> dict[str, Any]:
        """Whether a job is running (and its progress), which model is loaded, VRAM/RAM
        use, memory warnings and recent server log lines."""
        s = service.status()
        return {
            "busy": s["busy"], "phase": s["phase"], "job": s["job"],
            "step": s["step"] if s["busy"] else None,
            "total_steps": s["total_steps"] if s["busy"] else None,
            "model_loaded": s["model_loaded"], "loaded_model": s["loaded_model"],
            "memory_mode": s["memory_mode"], "vram": s["gpu"], "ram": s["ram"],
            "warnings": s["warnings"],
            "logs": log_buffer.recent(log_lines) if log_lines else [],
        }

    @mcp.tool(title="Models, sizes and defaults", structured_output=True,
              annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def get_options() -> dict[str, Any]:
        """Models (and whether they're downloaded), allowed sizes, the pixel cap, step
        counts and edit limits."""
        return service.public_config()

    @mcp.tool(title="List generated images", structured_output=True,
              annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def list_images(limit: Annotated[int, Field(ge=1, le=100)] = 20,
                    offset: Annotated[int, Field(ge=0)] = 0) -> dict[str, Any]:
        """Images in output/, newest first, with prompt preview, size and status."""
        items, total = service.store.list_images(offset=offset, limit=limit)
        return {"total": total, "items": [
            {**links(it["id"]), "created": it["created"], "kind": it["kind"], "status": it["status"],
             "size": it["size"], "model": it["model"], "seed": it["seed"],
             "prompt": (it["prompt"] or "")[:160] or None}
            for it in items]}

    @mcp.tool(title="Get one image's details", annotations=ToolAnnotations(
        read_only_hint=True, open_world_hint=False))
    def get_image(id: Annotated[str, Field(description="Image id or file name from output/.")],
                  include_image: IncludeImage = False,
                  include_log: Annotated[bool, Field(description="Include the full per-job log.")] = False,
                  ) -> list[TextContent | ImageContent]:
        """Parameters, prompt and status of one job (also works while it is still running
        or after it failed), plus optionally a preview and the full log."""
        try:
            stem = stem_from_name(id)
        except InvalidName as e:
            raise ToolError(str(e))
        p = service.monitor.snapshot()
        record = service.store.read_log(stem)
        exists = service.store.exists(stem)
        if p.active and p.stem == stem:
            info = {"status": "running", **links(stem), "phase": p.phase,
                    "step": p.step, "total_steps": p.total_steps}
        elif record is None and not exists:
            raise ToolError(f"No image or log with id '{stem}'.")
        else:
            fields = record.fields if record else {}
            info = {"status": fields.get("status", "success" if exists else "unknown"),
                    **links(stem), "exists": exists, "fields": fields,
                    "prompt": record.prompt if record else None}
            if record and include_log:
                info["log"] = record.text
        content: list = [TextContent(type="text", text=json.dumps(info))]
        if include_image and exists:
            content.append(_preview(service.store.image_path(stem).read_bytes()))
        return content

    return mcp


def mount_mcp(app: FastAPI, service: ImageService) -> MCPServer:
    """Add the MCP endpoint to `app` at /mcp. The caller must run
    `mcp.session_manager.run()` in the app's lifespan."""
    mcp = build_mcp(service)
    settings = service.settings
    hosts = ["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*", "[::1]", "[::1]:*",
             *settings.mcp_allowed_hosts]
    origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
               *[f"http://{h}" for h in settings.mcp_allowed_hosts]]
    sub = mcp.streamable_http_app(
        streamable_http_path=MCP_PATH,
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins),
        # base64 images for edit_image: allow the upload limit per image plus base64 overhead.
        max_request_body_size=int(settings.max_upload_bytes * settings.edit_max_images * 1.4) + 1024 * 1024,
    )
    # Reuse the SDK's route (exact path /mcp, all methods) inside the FastAPI router.
    for route in sub.routes:
        app.router.routes.append(route)
    return mcp
