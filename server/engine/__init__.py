"""Model engines: the only part of the app that touches torch / diffusers.

`Engine` is the interface the service layer talks to. `QwenEngine` (engine/qwen.py)
is the real one; tests swap in a fake that satisfies the same interface, so the API,
MCP server and storage can be tested without a model, a GPU or torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image


@dataclass(frozen=True)
class RenderJob:
    """Everything an engine needs to produce one image (already validated)."""

    model: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int              # num_inference_steps (ignored by a checkpoint with a saved schedule)
    true_cfg_scale: float   # 1.0 = guidance off
    seed: int


@dataclass
class RenderResult:
    image: Image.Image
    steps_ran: int
    steps_planned: int
    details: dict[str, Any] = field(default_factory=dict)  # extra fields for the per-job .log


class JobHooks(Protocol):
    """How an engine reports progress (implemented by monitoring.Monitor)."""

    def set_phase(self, phase: str, total_steps: int | None = None) -> None: ...
    def on_step(self) -> None: ...


class Engine(Protocol):
    def loaded_model(self) -> str | None: ...
    def is_loaded(self) -> bool: ...
    def load(self, model: str) -> None: ...
    def text_to_image(self, job: RenderJob, hooks: JobHooks) -> RenderResult: ...
    def edit(self, job: RenderJob, images: list[Image.Image], resolution: int,
             hooks: JobHooks) -> RenderResult: ...
    def free_memory(self) -> None: ...
