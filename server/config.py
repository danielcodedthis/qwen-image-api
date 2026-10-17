import os
import yaml

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config.yaml")

DEFAULTS = {
    "memory": {
        # True = keep the whole pipeline in system RAM and run on CPU (no GPU use).
        "run_on_ram": False,
        # Int8 weight-only quantization of the transformer.
        "quantize_transformer": True,
        # Decode images through the VAE in float32 instead of bf16. Can remove fine
        # banding/stripes; costs extra VRAM during the final decode step.
        "vae_fp32": False,
        # Keep the VAE in system RAM and decode the final image on the CPU (float32).
        # Slower for the last step only, but uses almost no VRAM for it. Use this when
        # the decode runs out of GPU memory at large sizes.
        "vae_on_cpu": False,
        "gpu_max_memory": "22GiB",
        "cpu_max_memory": "80GiB",
    },
    "generation": {
        "total_steps": 40,
        "max_pixels": 1024 * 1024,
        "true_cfg_scale": 1.0,  # 1.0 = off
        "negative_prompt": "",
    },
    "ui": {
        "available_sizes": ["512x512", "768x768", "1024x1024"],
    },
    "startup": {
        "lazy_load": False,
    },
    # Models selectable from the web UI. Per-model total_steps / true_cfg_scale /
    # negative_prompt override the values in "generation".
    "default_model": "base",
    "models": {
        "base": {
            "label": "Qwen-Image-2.1",
            "path": "/app/models/Qwen-Image-2.1",
        },
        "turbo": {
            "label": "Qwen-Image-2.1 Turbo",
            "path": "/app/models/Qwen-Image-2.1-Turbo",
            "total_steps": 8,
            "true_cfg_scale": 1.0,
        },
    },
    "monitoring": {
        # Log + show a warning in the UI when usage crosses these percentages.
        "vram_warn_percent": 92,
        "ram_warn_percent": 90,
    },
}


def _deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config():
    config = DEFAULTS
    try:
        with open(CONFIG_PATH) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(DEFAULTS, user_config)
    except FileNotFoundError:
        # Fall back to defaults — lets the app run standalone (e.g. outside
        # docker compose, or if the config mount is ever omitted) without crashing.
        pass

    # A model with `enabled: false` is dropped, so a container can serve just one.
    enabled = {k: v for k, v in config["models"].items() if v.get("enabled", True)}
    if not enabled:
        raise ValueError("config.yaml: every model is disabled; enable at least one under 'models'.")
    config = dict(config, models=enabled)
    if config["default_model"] not in enabled:
        config["default_model"] = next(iter(enabled))
    return config


config = load_config()


def get_model_settings(name):
    """Resolved settings for one model: generation defaults + that model's overrides."""
    gen = config["generation"]
    m = config["models"][name]
    return {
        "id": name,
        "label": m.get("label", name),
        "path": m["path"],
        "total_steps": int(m.get("total_steps", gen["total_steps"])),
        "true_cfg_scale": float(m.get("true_cfg_scale", gen["true_cfg_scale"])),
        "negative_prompt": m.get("negative_prompt", gen["negative_prompt"]) or "",
    }
