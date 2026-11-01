"""A fake engine with the same interface as server.engine.qwen.QwenEngine.

No torch, no model files, no GPU. It produces deterministic images (a colour
gradient derived from the seed and prompt), reports progress through the same hooks,
and can be told to fail or to pause mid-run so tests can exercise every server path.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

from PIL import Image

from server.engine import JobHooks, RenderJob, RenderResult


@dataclass
class Call:
    kind: str
    job: RenderJob
    images: list[Image.Image] = field(default_factory=list)
    resolution: int | None = None


@dataclass
class MockEngine:
    loaded: str | None = None
    calls: list[Call] = field(default_factory=list)
    loads: list[str] = field(default_factory=list)
    # models whose checkpoint has a saved schedule: id -> steps it always runs
    fixed_steps: dict[str, int] = field(default_factory=lambda: {"turbo": 8})
    # failure injection
    fail_on_load: BaseException | None = None
    fail_on_render: BaseException | None = None
    # pause support: set `hold` to make a job wait at its first step until `release` is set
    hold: bool = False
    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    freed: int = 0
    # seconds per fake denoising step (0 in tests; >0 in mock_server.py to watch progress)
    step_delay: float = 0.0
    # produce RGBA output (like a transparent generation)
    rgba: bool = False

    # ---- Engine interface ----
    def loaded_model(self) -> str | None:
        return self.loaded

    def is_loaded(self) -> bool:
        return self.loaded is not None

    def load(self, model: str) -> None:
        if self.fail_on_load is not None:
            raise self.fail_on_load
        if self.loaded != model:
            self.loads.append(model)
            self.loaded = model

    def free_memory(self) -> None:
        self.freed += 1

    def text_to_image(self, job: RenderJob, hooks: JobHooks) -> RenderResult:
        self.calls.append(Call("generate", job))
        return self._run(job, hooks)

    def edit(self, job: RenderJob, images: list[Image.Image], resolution: int,
             hooks: JobHooks) -> RenderResult:
        self.calls.append(Call("edit", job, list(images), resolution))
        result = self._run(job, hooks)
        # paste the last reference in the corner, so tests can tell it was used
        ref = images[-1].convert("RGB").resize((max(1, job.width // 4), max(1, job.height // 4)))
        result.image.paste(ref, (0, 0))
        result.details["reference_images"] = len(images)
        return result

    # ---- helpers ----
    def _run(self, job: RenderJob, hooks: JobHooks) -> RenderResult:
        hooks.set_phase("loading")
        self.load(job.model)
        steps = self.fixed_steps.get(job.model, job.steps)
        hooks.set_phase("denoising", total_steps=steps)
        for i in range(steps):
            if i == 0:
                self.started.set()
                if self.hold:
                    assert self.release.wait(10), "test never released the mock job"
            if self.fail_on_render is not None and i == steps // 2:
                raise self.fail_on_render
            if self.step_delay:
                time.sleep(self.step_delay)
            hooks.on_step()
        details = {"cfg": "off" if job.true_cfg_scale <= 1 else job.true_cfg_scale}
        return RenderResult(image=render_for(job, self.rgba), steps_ran=steps, steps_planned=steps,
                            details=details)


def render_for(job: RenderJob, rgba: bool = False) -> Image.Image:
    """Deterministic picture for a job: same seed + prompt -> same pixels."""
    digest = hashlib.sha256(f"{job.seed}|{job.prompt}".encode()).digest()
    w, h = job.width, job.height
    ramp = Image.linear_gradient("L").resize((w, h))
    img = Image.merge("RGB", (ramp, Image.new("L", (w, h), digest[0]), Image.new("L", (w, h), digest[1])))
    if rgba:
        img.putalpha(Image.linear_gradient("L").rotate(90).resize((w, h)))
    return img
