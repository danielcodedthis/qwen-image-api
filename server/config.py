import os
import yaml

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config.yaml")

DEFAULTS = {
    "memory": {
        "gpu_max_memory": "22GiB",
        "cpu_max_memory": "40GiB",
    },
    "generation": {
        "total_steps": 40,
        "max_pixels": 1024 * 1024,
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