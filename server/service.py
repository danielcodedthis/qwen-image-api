"""The job layer: validates a request against the config, runs it on the engine one at
a time, and records the result (PNG, per-job log, progress, OOM state).

HTTP routes and MCP tools both call this; neither talks to the engine directly.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image

from . import imaging
from .engine import Engine, RenderJob, RenderResult
from .monitoring import Monitor, get_gpu_stats, get_ram_stats, is_oom_error, log_buffer, logger
from .settings import SIZE_MULTIPLE, ModelSettings, Settings
from .storage import InvalidName, OutputStore, stem_from_name

_SIZE_RE = re.compile(r"^\s*(\d{1,5})\s*[xX×]\s*(\d{1,5})\s*$")


# ------------------------------------------------------------------ errors
class ServiceError(Exception):
    status_code = 500


class InvalidRequest(ServiceError):
    status_code = 400


class NotFound(ServiceError):
    status_code = 404


class Busy(ServiceError):
    status_code = 409

    def __init__(self, message: str = "Another image is already being generated. Wait for it to finish."):
        super().__init__(message)


class OutOfMemory(ServiceError):
    status_code = 503


# ------------------------------------------------------------------ results
@dataclass
class JobOutcome:
    id: str
    kind: str
    model: str
    width: int
    height: int
    seed: int
    seconds: float
    png: bytes
    saved: bool
    steps_ran: int
    details: dict = field(default_factory=dict)

    @property
    def file(self) -> str | None:
        return f"{self.id}.png" if self.saved else None

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class SourceImage:
    """A reference image for an edit, plus a note on where it came from (for the log)."""

    image: Image.Image
    label: str


def parse_size(size: str) -> tuple[int, int]:
    m = _SIZE_RE.match(size or "")
    if not m:
        raise InvalidRequest(f"Size '{size}' is not valid: use WIDTHxHEIGHT, for example 1024x1024.")
    return int(m.group(1)), int(m.group(2))


class ImageService:
    def __init__(self, settings: Settings, engine: Engine, store: OutputStore, monitor: Monitor):
        self.settings = settings
        self.engine = engine
        self.store = store
        self.monitor = monitor
        self._lock = threading.Lock()  # one job (or model load) at a time

    # ------------------------------------------------------------ validation
    def resolve_model(self, name: str | None) -> ModelSettings:
        name = name or self.settings.default_model
        if name not in self.settings.models:
            raise InvalidRequest(f"Unknown model '{name}'. Available: {', '.join(self.settings.models)}.")
        ms = self.settings.model(name)
        if not ms.available:
            raise InvalidRequest(
                f"Model folder not found: {ms.path}. Download it into ./models first (see docs/MODEL_SETUP.md)."
            )
        return ms

    def check_size(self, w: int, h: int) -> None:
        if w <= 0 or h <= 0 or w % SIZE_MULTIPLE or h % SIZE_MULTIPLE:
            raise InvalidRequest(
                f"Size {w}x{h} is not valid: width and height must be positive multiples of {SIZE_MULTIPLE}."
            )
        if w * h > self.settings.max_pixels:
            raise InvalidRequest(
                f"Resolution {w}x{h} ({w * h} px) exceeds max_pixels ({self.settings.max_pixels}) in the config."
            )

    def build_job(self, *, prompt: str, width: int, height: int, model: str | None,
                  steps: int | None, true_cfg_scale: float | None, seed: int | None,
                  negative_prompt: str | None) -> tuple[RenderJob, ModelSettings]:
        self.check_size(width, height)
        ms = self.resolve_model(model)
        if steps is not None and ms.fixed_steps and steps != ms.fixed_steps:
            raise InvalidRequest(
                f"{ms.label} always runs the {ms.fixed_steps}-step schedule saved in its checkpoint; "
                "leave steps empty."
            )
        job = RenderJob(
            model=ms.id,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt is not None else ms.negative_prompt,
            width=width,
            height=height,
            steps=steps if steps is not None else ms.total_steps,
            true_cfg_scale=true_cfg_scale if true_cfg_scale is not None else ms.true_cfg_scale,
            seed=seed if seed is not None else int.from_bytes(os.urandom(4), "big"),
        )
        if not 1 <= job.steps <= 200:
            raise InvalidRequest("steps must be between 1 and 200.")
        if not 1 <= job.true_cfg_scale <= 10:
            raise InvalidRequest("true_cfg_scale must be between 1 (off) and 10.")
        return job, ms

    def load_source(self, *, image_bytes: bytes | None = None, image_b64: str | None = None,
                    source_name: str | None = None, label: str | None = None) -> SourceImage:
        """One reference image, from an upload, base64, or a previous output."""
        try:
            if source_name:
                try:
                    stem = stem_from_name(source_name)
                except InvalidName as e:
                    raise InvalidRequest(str(e)) from None
                path = self.store.image_path(stem)
                if not path.is_file():
                    raise NotFound(f"No image named '{stem}.png' in output/.")
                img = imaging.open_image(path.read_bytes(), self.settings.max_upload_bytes)
                return SourceImage(imaging.shrink_reference(img), f"{stem}.png")
            if image_b64 is not None:
                image_bytes = imaging.decode_base64_image(image_b64)
            img = imaging.open_image(image_bytes or b"", self.settings.max_upload_bytes)
            return SourceImage(imaging.shrink_reference(img), label or f"upload ({img.width}x{img.height})")
        except imaging.ImageInputError as e:
            raise InvalidRequest(f"Reference image{f' ({label})' if label else ''}: {e}") from None

    def edit_size(self, last: Image.Image, size: str | None) -> tuple[int, int]:
        if size is None or size.strip().lower() in ("", "auto"):
            w, h = imaging.auto_edit_size(last.width, last.height, self.settings.edit_resolution,
                                          self.settings.max_pixels)
        else:
            w, h = parse_size(size)
        self.check_size(w, h)
        return w, h

    # ------------------------------------------------------------ public entry points
    def generate(self, *, prompt: str, size: str = "1024x1024", model: str | None = None,
                 steps: int | None = None, true_cfg_scale: float | None = None,
                 seed: int | None = None, negative_prompt: str | None = None,
                 wait: bool = True) -> JobOutcome | str:
        w, h = parse_size(size)
        job, ms = self.build_job(prompt=prompt, width=w, height=h, model=model, steps=steps,
                                 true_cfg_scale=true_cfg_scale, seed=seed, negative_prompt=negative_prompt)
        return self._launch(kind="generate", job=job, ms=ms, wait=wait,
                            render=lambda hooks: self.engine.text_to_image(job, hooks))

    def edit(self, *, prompt: str, sources: list[SourceImage], size: str | None = None,
             model: str | None = None, steps: int | None = None, true_cfg_scale: float | None = None,
             seed: int | None = None, negative_prompt: str | None = None,
             wait: bool = True) -> JobOutcome | str:
        if not sources:
            raise InvalidRequest("An edit needs at least one reference image.")
        limit = self.settings.edit_max_images
        if len(sources) > limit:
            raise InvalidRequest(f"{len(sources)} reference images given; the limit is {limit} "
                                 "(edit.max_images in the config, at most 10).")
        images = [s.image for s in sources]
        w, h = self.edit_size(images[-1], size)
        job, ms = self.build_job(prompt=prompt, width=w, height=h, model=model, steps=steps,
                                 true_cfg_scale=true_cfg_scale, seed=seed, negative_prompt=negative_prompt)
        resolution = self.settings.edit_resolution
        extra = {"reference_images": len(images),
                 "sources": "; ".join(f"{i}: {s.label}" for i, s in enumerate(sources, 1)),
                 "edit_resolution": resolution}
        return self._launch(kind="edit", job=job, ms=ms, wait=wait, sources=images, extra_fields=extra,
                            render=lambda hooks: self.engine.edit(job, images, resolution, hooks))

    def preload(self) -> None:
        """Load the default model in the background at startup (startup.lazy_load: false)."""
        if not self._lock.acquire(blocking=False):
            return
        try:
            ms = self.resolve_model(None)
            self.monitor.begin("load", "-", ms.id)
            self.engine.load(ms.id)
        except InvalidRequest as e:
            logger.warning(f"Not preloading: {e}")
        except Exception as e:  # noqa: BLE001
            if is_oom_error(e):
                self.monitor.report_oom(e, "model load (startup)", self.settings.memory_mode)
                self.engine.free_memory()
            else:
                logger.exception("Model preload failed")
        finally:
            self.monitor.end()
            self._lock.release()

    # ------------------------------------------------------------ running
    def _launch(self, *, kind: str, job: RenderJob, ms: ModelSettings, wait: bool, render: Callable,
                sources: list[Image.Image] | None = None,
                extra_fields: dict | None = None) -> JobOutcome | str:
        if not self._lock.acquire(blocking=False):
            raise Busy()
        stem = self.store.new_stem()
        # Mark busy before returning, so a status poll right after a background submit sees it.
        self.monitor.begin(kind, stem, job.model)
        args = dict(kind=kind, job=job, ms=ms, render=render, stem=stem,
                    sources=sources or [], extra_fields=extra_fields or {})
        if wait:
            try:
                return self._run(**args)
            finally:
                self._lock.release()

        def background():
            try:
                self._run(**args)
            except Exception:  # noqa: BLE001 - already logged and written to the job's .log
                pass
            finally:
                self._lock.release()

        threading.Thread(target=background, name=f"job-{stem}", daemon=True).start()
        return stem

    def _run(self, *, kind: str, job: RenderJob, ms: ModelSettings, render: Callable, stem: str,
             sources: list[Image.Image], extra_fields: dict) -> JobOutcome:
        cursor = log_buffer.cursor()
        started = time.time()
        verb = "Editing" if kind == "edit" else "Generating"
        noun = "Edit" if kind == "edit" else "Generation"
        preview = " ".join(job.prompt.split())[:60]
        cfg = f"cfg={job.true_cfg_scale}" if job.true_cfg_scale > 1 else "cfg=off"
        refs = f", {len(sources)} reference image(s)" if sources else ""
        logger.info(f"{verb} [{ms.label}, {ms.fixed_steps or job.steps} steps, seed {job.seed}{refs}]: "
                    f"'{preview}' size={job.width}x{job.height} {cfg}")
        base_fields = {
            "mode": kind,
            "model": f"{ms.id} ({ms.label})",
            "size": f"{job.width}x{job.height}",
            "steps_requested": ms.fixed_steps or job.steps,
            "true_cfg_scale": job.true_cfg_scale,
            "seed": job.seed,
            "memory_mode": self.settings.memory_mode,
            **extra_fields,
        }

        def write_log(status: str, more: dict) -> None:
            p = self.monitor.snapshot()
            fields = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status, **base_fields,
                      "duration_seconds": round(time.time() - started, 1),
                      "peak_vram_gb": p.peak_vram_gb, "peak_ram_gb": p.peak_ram_gb, **more}
            negative = job.negative_prompt if job.true_cfg_scale > 1 else ""
            self.store.write_log(stem, fields, job.prompt, negative, log_buffer.since(cursor))

        if sources:
            self.store.save_sources(stem, sources)

        try:
            try:
                result: RenderResult = render(self.monitor)
            except Exception as e:
                phase = self.monitor.snapshot().phase
                error_text = " ".join(str(e).split())[:500] or type(e).__name__
                if is_oom_error(e):
                    stage = (f"model load ({job.model})" if phase == "loading"
                             else f"{kind} ({job.width}x{job.height})")
                    self.monitor.report_oom(e, stage, self.settings.memory_mode)
                    self.engine.free_memory()
                    write_log("failed", {"error": f"Out of memory: {error_text}"})
                    if phase == "loading":
                        raise OutOfMemory("Out of memory while loading the model. See logs.") from e
                    hint = " or fewer reference images" if kind == "edit" else ""
                    raise OutOfMemory(f"Out of memory at {job.width}x{job.height}. "
                                      f"Try a smaller size{hint} (see server logs for details).") from e
                logger.exception(f"{noun} failed")
                write_log("failed", {"error": error_text})
                raise

            secs = time.time() - started
            p = self.monitor.snapshot()
            logger.info(f"{noun} complete in {secs:.1f}s "
                        f"(peak VRAM {p.peak_vram_gb} GB, peak RAM {p.peak_ram_gb} GB)")
            if result.steps_ran == 0:
                logger.warning("No denoising steps were counted, so the progress bar can't show real progress.")
            elif result.steps_ran != result.steps_planned:
                logger.warning(f"Pipeline ran {result.steps_ran} denoising steps but {result.steps_planned} "
                               "were planned — the model's saved schedule may not be applied as expected.")
            else:
                logger.info(f"Ran {result.steps_ran} denoising steps.")

            self.monitor.set_phase("saving")
            png = imaging.png_bytes(result.image)
            saved = self.store.save_image(stem, png)
            self.monitor.clear_oom()
            width, height = result.image.size
            write_log("success", {"steps_ran": result.steps_ran, **result.details,
                                  "image_file": f"output/{stem}.png", "image_saved": saved})
            self.engine.free_memory()
            return JobOutcome(id=stem, kind=kind, model=job.model, width=width, height=height,
                              seed=job.seed, seconds=round(secs, 1), png=png, saved=saved,
                              steps_ran=result.steps_ran, details=dict(result.details))
        finally:
            self.monitor.end()

    # ------------------------------------------------------------ read-only views
    def status(self) -> dict:
        gpu, ram = get_gpu_stats(), get_ram_stats()
        p = self.monitor.snapshot()
        busy = p.active
        return {
            "busy": busy,
            "generating": busy,  # kept for older clients
            "phase": p.phase,
            "job": {
                "id": p.stem, "kind": p.kind, "model": p.model,
                "elapsed_seconds": round(time.time() - p.started_at, 1) if p.started_at else None,
            } if busy else None,
            "model_loaded": self.engine.is_loaded(),
            "loaded_model": self.engine.loaded_model(),
            "run_on_ram": self.settings.run_on_ram,
            "memory_mode": self.settings.memory_mode,
            "step": p.step,
            "total_steps": p.total_steps,
            "gpu": gpu,
            "ram": ram,
            "warnings": self.monitor.active_warnings(gpu, ram),
            "logs": log_buffer.recent(),
        }

    def public_config(self) -> dict:
        return {
            "available_sizes": self.settings.available_sizes,
            "max_pixels": self.settings.max_pixels,
            "size_multiple": SIZE_MULTIPLE,
            "default_model": self.settings.default_model,
            "edit": {
                "max_images": self.settings.edit_max_images,
                "resolution": self.settings.edit_resolution,
                "max_upload_mb": self.settings.raw["edit"]["max_upload_mb"],
            },
            "mcp": {"enabled": self.settings.mcp_enabled, "path": "/mcp", "name": self.settings.mcp_name},
            "models": [
                {
                    "id": m.id,
                    "label": m.label,
                    "path": m.path,
                    "available": m.available,
                    "total_steps": m.steps,
                    "fixed_steps": m.fixed_steps,
                    "true_cfg_scale": m.true_cfg_scale,
                    "negative_prompt": m.negative_prompt,
                }
                for m in self.settings.models.values()
            ],
        }
