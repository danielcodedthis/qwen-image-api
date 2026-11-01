"""Request bodies. Types and ranges are checked here; anything that depends on the
config (sizes vs max_pixels, model names, image limits) is checked in service.py."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_SEED = 2**63 - 1


class _Common(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    prompt: str = Field(..., max_length=100_000)
    # Only used when true_cfg_scale > 1. None = the model's configured negative prompt.
    negative_prompt: Optional[str] = Field(None, max_length=20_000)
    # None = default_model from the config.
    model: Optional[str] = None
    # None = the model's total_steps. Not allowed for a checkpoint with a saved schedule (Turbo).
    steps: Optional[int] = Field(None, ge=1, le=200)
    # 1.0 = guidance off (how Qwen-Image-2.1 is meant to run); >1 runs the transformer twice per step.
    true_cfg_scale: Optional[float] = Field(None, ge=1, le=10)
    # None = random. The seed that was used is returned in the response.
    seed: Optional[int] = Field(None, ge=0, le=MAX_SEED)
    # OpenAI compatibility: one image per request.
    n: int = Field(1, ge=1, le=1)
    # "b64_json" returns the PNG inline; "url" returns a link to the saved file.
    response_format: Literal["b64_json", "url"] = "b64_json"

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v


class GenerateRequest(_Common):
    size: str = "1024x1024"


class ImageRef(BaseModel):
    """One reference image: base64 / data URL (`data`) or an image in output/ (`source`)."""

    model_config = ConfigDict(extra="forbid")

    data: Optional[str] = None
    source: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one(self):
        if bool(self.data) == bool(self.source):
            raise ValueError("each entry in 'images' needs exactly one of 'data' or 'source'")
        return self


class EditRequest(_Common):
    """JSON form of POST /v1/images/edits. Reference images, in order (Picture 1, 2, ...):

    - `images`: list of {"data": base64} / {"source": "<id in output/>"}, or
    - shortcuts for one image: `image` (base64 / data URL) or `source` (id).
    """

    # None or "auto": the last reference image's aspect ratio at about edit.resolution².
    size: Optional[str] = None
    images: list[ImageRef] = Field(default_factory=list, max_length=10)
    image: Optional[str] = None
    source: Optional[str] = None

    @model_validator(mode="after")
    def _collect(self):
        refs = list(self.images)
        if self.image:
            refs.append(ImageRef(data=self.image))
        if self.source:
            refs.append(ImageRef(source=self.source))
        if not refs:
            raise ValueError("give at least one reference image: 'images', 'image' (base64) or 'source'")
        if self.images and (self.image or self.source):
            raise ValueError("use either 'images' or the 'image'/'source' shortcut, not both")
        self.images = refs
        self.image = self.source = None
        return self
