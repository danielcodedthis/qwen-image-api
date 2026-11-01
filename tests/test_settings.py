"""Config loading: defaults, merging, per-model overrides, saved schedules, validation."""
import json

import pytest

from conftest import ROOT
from server.settings import DEFAULTS, ConfigError, load_settings, saved_schedule_steps


def test_missing_config_file_uses_defaults(tmp_path):
    s = load_settings(tmp_path / "nope.yaml", tmp_path / "out")
    assert s.config_path is None
    assert s.run_on_ram is False and s.memory_mode == "balanced"
    assert s.max_pixels == 1024 * 1024
    assert set(s.models) == {"base", "turbo"}
    assert s.default_model == "base"
    assert s.output_dir == tmp_path / "out"
    assert s.mcp_name == "qwen-image" and s.public_url == "http://localhost:8000"


def test_env_vars_choose_paths(tmp_env):
    s = load_settings()  # CONFIG_PATH / OUTPUT_DIR set by the tmp_env fixture
    assert s.config_path == tmp_env["config"]
    assert s.output_dir == tmp_env["output"]


@pytest.mark.parametrize("name, model, port, mcp_name", [
    ("config.yaml", "base", 8000, "qwen-image"),
    ("config.turbo.yaml", "turbo", 8001, "qwen-image-turbo"),
])
def test_repo_configs_load(tmp_path, name, model, port, mcp_name):
    """Both shipped configs must stay valid, and each serves exactly one model."""
    s = load_settings(ROOT / name, tmp_path)
    assert list(s.models) == [model] and s.default_model == model
    assert s.public_url == f"http://localhost:{port}" and s.mcp_name == mcp_name
    for size in s.available_sizes:
        w, h = map(int, size.split("x"))
        assert w % 32 == 0 and h % 32 == 0, size


def test_partial_config_merges_over_defaults(write_config, tmp_env):
    write_config(replace={"generation": {"total_steps": 12}})
    s = load_settings(tmp_env["config"], tmp_env["output"])
    assert s.model("base").total_steps == 12
    assert s.model("base").true_cfg_scale == 1.0           # default kept
    assert s.edit_max_images == DEFAULTS["edit"]["max_images"]  # new section defaulted
    assert s.mcp_enabled is True


def test_disabling_a_default_model_with_just_enabled_false(write_config, tmp_env):
    # config.turbo.yaml relies on model entries merging with the defaults.
    write_config(replace={"models": {"base": {"enabled": False}}, "default_model": "turbo"})
    s = load_settings(tmp_env["config"], tmp_env["output"])
    assert list(s.models) == ["turbo"] and s.model("turbo").total_steps == 8


def test_per_model_overrides(tmp_env):
    s = load_settings(tmp_env["config"], tmp_env["output"])
    assert (s.model("base").total_steps, s.model("base").negative_prompt) == (6, "blurry")


def test_saved_schedule_is_read_from_model_index(tmp_env):
    s = load_settings(tmp_env["config"], tmp_env["output"])
    turbo = s.model("turbo")
    assert turbo.total_steps == 6 and turbo.fixed_steps == 8 and turbo.steps == 8
    assert s.model("base").fixed_steps is None and s.model("base").steps == 6


def test_saved_schedule_edge_cases(tmp_path):
    assert saved_schedule_steps(str(tmp_path / "absent")) is None
    (tmp_path / "model_index.json").write_text("{not json")
    assert saved_schedule_steps(str(tmp_path)) is None
    (tmp_path / "model_index.json").write_text(json.dumps({"sample_sigmas": None}))
    assert saved_schedule_steps(str(tmp_path)) is None


def test_disabled_models_are_dropped_and_default_falls_back(write_config, tmp_env):
    write_config({"models": {"base": {"enabled": False}}})
    s = load_settings(tmp_env["config"], tmp_env["output"])
    assert "base" not in s.models and s.default_model == "turbo"


@pytest.mark.parametrize("overrides, message", [
    ({"generation": {"max_pixels": 0}}, "max_pixels"),
    ({"generation": {"total_steps": 0}}, "total_steps"),
    ({"generation": {"true_cfg_scale": 0.5}}, "true_cfg_scale"),
    ({"edit": {"max_images": 0}}, "max_images"),
    ({"edit": {"max_images": 11}}, "max_images"),
    ({"edit": {"resolution": 1000}}, "resolution"),
    ({"edit": {"max_upload_mb": 0}}, "max_upload_mb"),
])
def test_invalid_values_are_rejected(write_config, tmp_env, overrides, message):
    write_config(overrides)
    with pytest.raises(ConfigError, match=message):
        load_settings(tmp_env["config"], tmp_env["output"])


def test_all_models_disabled_is_an_error(write_config, tmp_env):
    write_config({"models": {name: {"enabled": False} for name in ("base", "turbo", "missing")}})
    with pytest.raises(ConfigError, match="every model is disabled"):
        load_settings(tmp_env["config"], tmp_env["output"])


def test_non_mapping_config_is_an_error(tmp_env):
    tmp_env["config"].write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(tmp_env["config"], tmp_env["output"])


def test_model_availability_follows_folder(tmp_env):
    s = load_settings(tmp_env["config"], tmp_env["output"])
    assert s.model("base").available is True
    assert s.model("missing").available is False
