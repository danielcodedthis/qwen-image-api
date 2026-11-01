#!/usr/bin/env python3
"""stdio <-> HTTP bridge for the Qwen Image MCP server.

Some MCP hosts (e.g. LM Studio Bionic's "Custom MCP") only launch servers as a
command speaking MCP over stdin/stdout. The Qwen Image API serves MCP over HTTP at
http://localhost:8000/mcp (Turbo: http://localhost:8001/mcp). This script sits in
between: it reads JSON-RPC messages from stdin, POSTs each one to the HTTP endpoint,
and writes every message of the reply (including progress notifications) to stdout.

    python3 tools/mcp_stdio_bridge.py [URL]       # default URL: http://localhost:8000/mcp
    QWEN_MCP_URL=http://localhost:8001/mcp python3 tools/mcp_stdio_bridge.py   # Turbo

Standard library only: no venv, pip or npm needed. Requests run concurrently, so a
`ping` or `get_status` still answers while a long generate_image call is waiting.
Diagnostics go to stderr (stdout carries the protocol).
"""
from __future__ import annotations

import http.client
import json
import os
import sys
import threading
from urllib.parse import urlsplit

DEFAULT_URL = "http://localhost:8000/mcp"

_out_lock = threading.Lock()
_protocol_version: str | None = None


def log(msg: str) -> None:
    print(f"[qwen-mcp-bridge] {msg}", file=sys.stderr, flush=True)


def emit(message: dict) -> None:
    line = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    with _out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def error_reply(msg_id, text: str, code: int = -32000) -> None:
    if msg_id is not None:
        emit({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": text}})


def _connection(url):
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    # No read timeout: a generation can take many minutes on a slow memory mode.
    return cls(url.hostname, url.port or (443 if url.scheme == "https" else 80), timeout=None)


def _remember_version(message: dict) -> None:
    global _protocol_version
    result = message.get("result")
    if isinstance(result, dict) and isinstance(result.get("protocolVersion"), str):
        _protocol_version = result["protocolVersion"]


def forward(url, raw: str) -> None:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        log("ignoring a line that is not JSON")
        return
    msg_id = message.get("id") if isinstance(message, dict) else None
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if _protocol_version:
        headers["MCP-Protocol-Version"] = _protocol_version
    conn = _connection(url)
    try:
        conn.request("POST", url.path or "/", body=raw.encode("utf-8"), headers=headers)
        resp = conn.getresponse()
        ctype = resp.getheader("Content-Type", "")
        if resp.status == 202 or msg_id is None and resp.status < 300:
            resp.read()
            return
        if resp.status >= 400:
            body = resp.read().decode("utf-8", "replace").strip()
            error_reply(msg_id, f"Qwen Image API answered HTTP {resp.status}: {body[:300]}")
            return
        if ctype.startswith("application/json"):
            reply = json.loads(resp.read())
            for m in reply if isinstance(reply, list) else [reply]:
                _remember_version(m)
                emit(m)
            return
        # text/event-stream: one JSON-RPC message per `data:` event
        data_lines: list[str] = []
        while True:
            line = resp.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip("\r\n")
            if text.startswith("data:"):
                data_lines.append(text[5:].lstrip())
            elif text == "" and data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                try:
                    m = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                _remember_version(m)
                emit(m)
    except (OSError, http.client.HTTPException) as e:
        error_reply(msg_id, f"Can't reach the Qwen Image API at {url.geturl()} ({e}). "
                            "Is the qwen-image-api container running?")
    finally:
        conn.close()


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QWEN_MCP_URL", DEFAULT_URL)
    url = urlsplit(target)
    if url.scheme not in ("http", "https") or not url.hostname:
        log(f"not an http(s) URL: {target}")
        sys.exit(2)
    log(f"forwarding stdio to {target}")
    workers: list[threading.Thread] = []
    for raw in sys.stdin:
        raw = raw.strip()
        if raw:
            t = threading.Thread(target=forward, args=(url, raw), daemon=True)
            t.start()
            workers.append(t)
            workers = [w for w in workers if w.is_alive()]
    for t in workers:  # stdin closed: let in-flight replies finish
        t.join()


if __name__ == "__main__":
    main()
