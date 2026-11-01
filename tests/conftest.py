"""Shared fixtures: a temporary config + output folder, a mock engine, and a
TestClient running the real FastAPI app (lifespan included, so /mcp works).

Nothing here touches the real config.yaml, output/, models/, torch or a GPU.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_engine import MockEngine  # noqa: E402

from server.app import create_app  # noqa: E402
from server.settings import load_settings  # noqa: E402


def base_config(models_dir: Path) -> dict:
    return {
        "memory": {"run_on_ram": False, "quantize_transformer": False, "vae_fp32": False,
                   "vae_on_cpu": True, "gpu_max_memory": "14GiB", "cpu_max_memory": "80GiB"},
        "generation": {"total_steps": 6, "true_cfg_scale": 1.0, "negative_prompt": "blurry",
                       "max_pixels": 1024 * 1024},
        "edit": {"max_images": 3, "resolution": 256, "max_upload_mb": 1},
        "ui": {"available_sizes": ["256x256", "512x512", "1024x1024", "1280x1280"]},
        "startup": {"lazy_load": True},
        "server": {"public_url": "http://testserver"},
        "mcp": {"enabled": True, "name": "qwen-image-test", "allowed_hosts": ["testserver"]},
        "default_model": "base",
        "models": {
            "base": {"label": "Test Base", "path": str(models_dir / "base")},
            "turbo": {"label": "Test Turbo", "path": str(models_dir / "turbo"), "total_steps": 6},
            "missing": {"label": "Not downloaded", "path": str(models_dir / "missing")},
        },
    }


# The Turbo checkpoint's model_index.json carries its 8-step schedule.
TURBO_SIGMAS = [1.0, 0.978, 0.954, 0.927, 0.895, 0.845, 0.705, 0.415]


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """Temporary data/config dirs. Also points CONFIG_PATH/OUTPUT_DIR there, so even an
    accidental load_settings() without arguments can't read real state."""
    models = tmp_path / "models"
    (models / "base").mkdir(parents=True)
    (models / "turbo").mkdir(parents=True)
    (models / "turbo" / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImage21Pipeline", "sample_sigmas": TURBO_SIGMAS}))
    output = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base_config(models), sort_keys=False))
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setenv("OUTPUT_DIR", str(output))
    return {"root": tmp_path, "models": models, "output": output, "config": config_path}


@pytest.fixture
def write_config(tmp_env):
    """Rewrite config.yaml with overrides merged into the base test config."""
    from server.settings import deep_merge

    def _write(overrides: dict | None = None, replace: dict | None = None) -> Path:
        cfg = replace if replace is not None else deep_merge(base_config(tmp_env["models"]), overrides or {})
        tmp_env["config"].write_text(yaml.safe_dump(cfg, sort_keys=False))
        return tmp_env["config"]
    return _write


@pytest.fixture
def mock_engine():
    return MockEngine()


@pytest.fixture
def make_client(tmp_env, mock_engine):
    """Build a TestClient for the real app with the mock engine (after config tweaks)."""
    clients = []

    def _make(engine=None) -> TestClient:
        settings = load_settings(tmp_env["config"], tmp_env["output"])
        app = create_app(settings, engine=engine or mock_engine)
        client = TestClient(app)
        client.__enter__()  # run the lifespan (MCP session manager, preload)
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def prompts() -> dict:
    return json.loads((FIXTURES / "prompts.json").read_text(encoding="utf-8"))


@pytest.fixture
def fixture_bytes():
    return lambda name: (FIXTURES / name).read_bytes()


@pytest.fixture
def fixture_b64(fixture_bytes):
    return lambda name: base64.b64encode(fixture_bytes(name)).decode()


# ------------------------------------------------------------------ MCP helper
class MCPClient:
    """Minimal MCP streamable-HTTP client over TestClient (stateless server)."""

    HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
               "MCP-Protocol-Version": "2025-06-18"}

    def __init__(self, http: TestClient):
        self.http = http
        self._id = 0

    def rpc(self, method: str, params: dict | None = None, *, headers: dict | None = None):
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        resp = self.http.post("/mcp", json=body, headers={**self.HEADERS, **(headers or {})})
        return resp, self._parse(resp)

    @staticmethod
    def _parse(resp):
        if resp.status_code != 200:
            return None
        ctype = resp.headers.get("content-type", "")
        if ctype.startswith("application/json"):
            return resp.json()
        messages = [json.loads(line[5:].strip()) for line in resp.text.splitlines()
                    if line.startswith("data:") and line[5:].strip()]
        final = [m for m in messages if "result" in m or "error" in m]
        return final[-1] if final else None

    def initialize(self):
        return self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                       "clientInfo": {"name": "pytest", "version": "1"}})

    def call(self, name: str, arguments: dict | None = None) -> dict:
        resp, msg = self.rpc("tools/call", {"name": name, "arguments": arguments or {}})
        assert resp.status_code == 200, resp.text
        assert msg is not None and "result" in msg, msg
        return msg["result"]


@pytest.fixture
def mcp(client):
    m = MCPClient(client)
    resp, _ = m.initialize()
    assert resp.status_code == 200, resp.text
    return m
