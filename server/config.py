import os
import yaml

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config.yaml")

DEFAULTS = {
    "memory": {
        # True = keep the whole pipeline in system RAM and run on CPU (no GPU use).
        "run_on_ram": False,
        # Int8 weight-only quantization of the transformer.
        "quantize_transformer": True,
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
    return config


config = load_config()
