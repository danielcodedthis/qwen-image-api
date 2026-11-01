"""tools/mcp_stdio_bridge.py (for stdio-only MCP hosts such as Bionic), end to end:
a real uvicorn server with the mock engine, and the bridge as a subprocess."""
import json
import socket
import subprocess
import sys
import threading
import time

import pytest
import uvicorn

from conftest import ROOT
from server.app import create_app
from server.settings import load_settings

BRIDGE = ROOT / "tools" / "mcp_stdio_bridge.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(tmp_env, mock_engine, write_config):
    port = free_port()
    write_config({"mcp": {"allowed_hosts": []}})  # real Host header: 127.0.0.1:<port>
    app = create_app(load_settings(tmp_env["config"], tmp_env["output"]), engine=mock_engine)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(5)


def run_bridge(url: str, messages: list[dict], timeout: float = 20) -> list[dict]:
    stdin = "".join(json.dumps(m) + "\n" for m in messages)
    proc = subprocess.run([sys.executable, str(BRIDGE), url], input=stdin, capture_output=True,
                          text=True, timeout=timeout, cwd="/")
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "bionic-like", "version": "1"}}}


def test_bridge_initialize_list_and_call(live_server, tmp_env):
    out = run_bridge(live_server, [INIT])
    assert out[0]["id"] == 1 and out[0]["result"]["serverInfo"]["name"] == "qwen-image-test"

    out = run_bridge(live_server, [
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "generate_image", "arguments": {"prompt": "via bridge", "size": "256x256"}}},
    ])
    by_id = {m["id"]: m for m in out if "id" in m}
    assert {t["name"] for t in by_id[2]["result"]["tools"]} >= {"generate_image", "edit_image", "get_status"}
    result = json.loads(by_id[3]["result"]["content"][0]["text"])
    assert result["status"] == "success"
    assert (tmp_env["output"] / f"{result['id']}.png").is_file()


def test_bridge_reports_unreachable_api():
    out = run_bridge(f"http://127.0.0.1:{free_port()}/mcp", [{"jsonrpc": "2.0", "id": 7, "method": "ping"}])
    assert out[0]["id"] == 7 and "Can't reach the Qwen Image API" in out[0]["error"]["message"]


def test_bridge_rejects_bad_url():
    proc = subprocess.run([sys.executable, str(BRIDGE), "ftp://x"], input="", capture_output=True, text=True)
    assert proc.returncode == 2 and "not an http(s) URL" in proc.stderr
