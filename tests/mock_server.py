"""Run the real API with the mock engine: UI and MCP work without a GPU or model.

    python tests/mock_server.py                  # API + MCP on :8102, UI on :3102
    python tests/mock_server.py --step-delay 0.3 --api-port 8202 --ui-port 3202

Images go to a temporary output folder (printed at start) unless --output is given.
Open the UI with ?api=<api-port> if you change the API port.
"""
import argparse
import json
import functools
import http.server
import sys
import tempfile
import threading
from pathlib import Path

import uvicorn
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

from conftest import TURBO_SIGMAS, base_config  # noqa: E402
from mock_engine import MockEngine  # noqa: E402

from server.app import create_app  # noqa: E402
from server.settings import deep_merge, load_settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-port", type=int, default=8102)
    ap.add_argument("--ui-port", type=int, default=3102)
    ap.add_argument("--step-delay", type=float, default=0.15, help="seconds per fake step")
    ap.add_argument("--output", type=Path, default=None, help="output folder (default: a temp dir)")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="qwen-mock-"))
    (work / "models" / "base").mkdir(parents=True)
    (work / "models" / "turbo").mkdir(parents=True)
    (work / "models" / "turbo" / "model_index.json").write_text(json.dumps({"sample_sigmas": TURBO_SIGMAS}))
    output = args.output or work / "output"
    cfg = deep_merge(base_config(work / "models"), {
        "generation": {"total_steps": 20, "max_pixels": 1280 * 1280},
        "edit": {"max_images": 4, "resolution": 1024, "max_upload_mb": 25},
        "ui": {"available_sizes": ["512x512", "768x768", "1024x1024", "768x1024", "1024x768", "1280x1280"]},
        "server": {"public_url": f"http://localhost:{args.api_port}"},
        "mcp": {"name": "qwen-image-mock", "allowed_hosts": []},
    })
    (work / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    app = create_app(load_settings(work / "config.yaml", output), engine=MockEngine(step_delay=args.step_delay))

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "web"))
    ui = http.server.ThreadingHTTPServer(("0.0.0.0", args.ui_port), handler)
    threading.Thread(target=ui.serve_forever, daemon=True).start()
    print(f"mock output folder: {output}")
    print(f"UI:  http://localhost:{args.ui_port}/?api={args.api_port}")
    print(f"MCP: http://localhost:{args.api_port}/mcp")
    uvicorn.run(app, host="0.0.0.0", port=args.api_port, log_level="warning")


if __name__ == "__main__":
    main()
