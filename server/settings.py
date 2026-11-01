"""Configuration: built-in defaults, merged with config.yaml, validated once at startup.

`config.yaml` (or `config.turbo.yaml`, mounted at the same place in the Turbo
container) is optional. Every key has a default here, so an older config without the
newer sections (`edit`, `server`, `mcp`) keeps working.

Paths come from environment variables so tests (and non-Docker runs) can point the
app at temporary folders:

    CONFIG_PATH   default /app/config.yaml
    OUTPUT_DIR    default /app/output
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "/app/config.yaml"
DEFAULT_OUTPUT_DIR = "/app/output"

# QwenImage21Pipeline floors width and height to multiples of 32 (VAE factor 16 x 2),
# so the API requires them up front instead of silently changing the size.
SIZE_MULTIPLE = 32

DEFAULTS: dict[str, Any] = {
    "memory": {
        # True = keep the whole pipeline in system RAM and run on CPU (no GPU use).
        "run_on_ram": False,
        # Int8 weight-only quantization of the transformer (torchao).
        "quantize_transformer": True,
        # Run the VAE in float32 (can remove fine banding; more VRAM in the decode).
        "vae_fp32": False,
        # Keep the VAE in system RAM and encode/decode on the CPU in float32.
        "vae_on_cpu": False,
        # Placement budgets for device_map="balanced", applied at load time.
        "gpu_max_memory": "22GiB",
        "cpu_max_memory": "80GiB",
    },
    "generation": {
        "total_steps": 40,
        "max_pixels": 1024 * 1024,
        "true_cfg_scale": 1.0,  # 1.0 = off
        "negative_prompt": "",
    },
    "edit": {
        # Most reference images one edit may use. The model supports 10; every image
        # adds tokens (VRAM and time), so raise this gradually.
        "max_images": 4,
        # Reference images are resized to about this side length squared (keeping their
        # aspect ratio). An edit without a size also comes out at about this area, with
        # the last reference image's aspect ratio. 1024 is the pipeline's default.
        "resolution": 1024,
        # Uploads larger than this are rejected before decoding.
        "max_upload_mb": 25,
    },
    "ui": {
        "available_sizes": ["512x512", "768x768", "1024x1024"],
    },
    "startup": {
        "lazy_load": False,
    },
    "monitoring": {
        "vram_warn_percent": 92,
        "ram_warn_percent": 90,
    },
    "server": {
        # Base URL that image links in API/MCP responses are built from.
        "public_url": "http://localhost:8000",
    },
    "mcp": {
        "enabled": True,
        # Server name shown to MCP clients (the Turbo container uses "qwen-image-turbo").
        "name": "qwen-image",
        # Host headers the MCP endpoint accepts, on top of localhost / 127.0.0.1.
        "allowed_hosts": [],
    },
    # Models selectable from the web UI. Per-model total_steps / true_cfg_scale /
    # negative_prompt override the values in "generation".
    "default_model": "base",
    "models": {
        "base": {"label": "Qwen-Image-2.1", "path": "/app/models/Qwen-Image-2.1"},
        "turbo": {
            "label": "Qwen-Image-2.1 Turbo",
            "path": "/app/models/Qwen-Image-2.1-Turbo",
            "total_steps": 8,
            "true_cfg_scale": 1.0,
        },
    },
}


class ConfigError(ValueError):
    """config.yaml has a value the app can't run with."""


def saved_schedule_steps(model_path: str) -> int | None:
    """Length of the sampling schedule saved with the checkpoint (`sample_sigmas` in
    model_index.json), or None. When it is set the pipeline always runs that many steps
    and ignores num_inference_steps (Qwen-Image-2.1-Turbo: 8)."""
    try:
        with open(os.path.join(model_path, "model_index.json"), encoding="utf-8") as f:
            sigmas = json.load(f).get("sample_sigmas")
    except (OSError, ValueError):
        return None
    return len(sigmas) if isinstance(sigmas, list) and sigmas else None


@dataclass(frozen=True)
class ModelSettings:
    """One model's resolved settings: the `generation` defaults plus its own overrides."""

    id: str
    label: str
    path: str
    total_steps: int
    true_cfg_scale: float
    negative_prompt: str
    # Steps fixed by the checkpoint's saved schedule (Turbo), None when steps are free.
    fixed_steps: int | None = None

    @property
    def available(self) -> bool:
        return os.path.isdir(self.path)

    @property
    def steps(self) -> int:
        """The number of steps that will actually run."""
        return self.fixed_steps or self.total_steps


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    output_dir: Path
    config_path: Path | None
    models: dict[str, ModelSettings] = field(default_factory=dict)

    # --- convenience accessors ---
    @property
    def run_on_ram(self) -> bool:
        return bool(self.raw["memory"]["run_on_ram"])

    @property
    def memory_mode(self) -> str:
        """Short name for /status and logs: "cpu" (run_on_ram) or "balanced"."""
        return "cpu" if self.run_on_ram else "balanced"

    @property
    def memory(self) -> dict[str, Any]:
        return self.raw["memory"]

    @property
    def max_pixels(self) -> int:
        return int(self.raw["generation"]["max_pixels"])

    @property
    def available_sizes(self) -> list[str]:
        return list(self.raw["ui"]["available_sizes"])

    @property
    def lazy_load(self) -> bool:
        return bool(self.raw["startup"]["lazy_load"])

    @property
    def vram_warn_percent(self) -> float:
        return float(self.raw["monitoring"]["vram_warn_percent"])

    @property
    def ram_warn_percent(self) -> float:
        return float(self.raw["monitoring"]["ram_warn_percent"])

    @property
    def default_model(self) -> str:
        return self.raw["default_model"]

    @property
    def edit_max_images(self) -> int:
        return int(self.raw["edit"]["max_images"])

    @property
    def edit_resolution(self) -> int:
        return int(self.raw["edit"]["resolution"])

    @property
    def max_upload_bytes(self) -> int:
        return int(float(self.raw["edit"]["max_upload_mb"]) * 1024 * 1024)

    @property
    def public_url(self) -> str:
        return str(self.raw["server"]["public_url"]).rstrip("/")

    @property
    def mcp_enabled(self) -> bool:
        return bool(self.raw["mcp"]["enabled"])

    @property
    def mcp_name(self) -> str:
        return str(self.raw["mcp"]["name"])

    @property
    def mcp_allowed_hosts(self) -> list[str]:
        return [str(h) for h in self.raw["mcp"]["allowed_hosts"] or []]

    def model(self, name: str) -> ModelSettings:
        return self.models[name]


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_model(name: str, entry: dict, gen: dict) -> ModelSettings:
    if not isinstance(entry, dict) or not entry.get("path"):
        raise ConfigError(f"config.yaml: models.{name} needs a 'path'.")
    path = str(entry["path"])
    return ModelSettings(
        id=name,
        label=str(entry.get("label", name)),
        path=path,
        total_steps=int(entry.get("total_steps", gen["total_steps"])),
        true_cfg_scale=float(entry.get("true_cfg_scale", gen["true_cfg_scale"])),
        negative_prompt=str(entry.get("negative_prompt", gen["negative_prompt"]) or ""),
        fixed_steps=saved_schedule_steps(path),
    )


def _validate(raw: dict) -> None:
    gen, edit = raw["generation"], raw["edit"]
    if int(gen["max_pixels"]) <= 0:
        raise ConfigError("config.yaml: generation.max_pixels must be positive.")
    if int(gen["total_steps"]) < 1:
        raise ConfigError("config.yaml: generation.total_steps must be at least 1.")
    if float(gen["true_cfg_scale"]) < 1:
        raise ConfigError("config.yaml: generation.true_cfg_scale must be at least 1.0 (1.0 = off).")
    if not 1 <= int(edit["max_images"]) <= 10:
        raise ConfigError("config.yaml: edit.max_images must be between 1 and 10 (the model's limit).")
    if int(edit["resolution"]) < 256 or int(edit["resolution"]) % SIZE_MULTIPLE:
        raise ConfigError(f"config.yaml: edit.resolution must be a multiple of {SIZE_MULTIPLE}, at least 256.")
    if float(edit["max_upload_mb"]) <= 0:
        raise ConfigError("config.yaml: edit.max_upload_mb must be positive.")
    if not isinstance(raw["models"], dict) or not raw["models"]:
        raise ConfigError("config.yaml: 'models' must list at least one model.")


def load_settings(config_path: str | os.PathLike | None = None,
                  output_dir: str | os.PathLike | None = None) -> Settings:
    """Read config.yaml (if present), merge it over DEFAULTS and validate it."""
    path = Path(config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))
    out = Path(output_dir or os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

    user_config: dict = {}
    found: Path | None = None
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"{path}: the top level must be a mapping.")
        user_config = loaded or {}
        found = path

    # Models merge with the defaults (as before): config.turbo.yaml relies on this to
    # disable `base` with just `enabled: false`.
    raw = deep_merge(DEFAULTS, user_config)
    _validate(raw)

    gen = raw["generation"]
    models = {
        name: _resolve_model(name, entry, gen)
        for name, entry in raw["models"].items()
        if not (isinstance(entry, dict) and entry.get("enabled", True) is False)
    }
    if not models:
        raise ConfigError("config.yaml: every model is disabled; enable at least one under 'models'.")
    raw["models"] = {k: v for k, v in raw["models"].items() if k in models}
    if raw["default_model"] not in models:
        raw["default_model"] = next(iter(models))

    return Settings(raw=raw, output_dir=out, config_path=found, models=models)
