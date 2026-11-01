"""Failure handling: busy (409), out of memory (503), crashes (500), unsaveable output."""
import os
import threading

import pytest

GEN = "/v1/images/generations"


def test_second_job_while_busy_gets_409(client, mock_engine):
    mock_engine.hold = True
    results = {}
    t = threading.Thread(target=lambda: results.setdefault(
        "first", client.post(GEN, json={"prompt": "slow", "size": "256x256"})))
    t.start()
    assert mock_engine.started.wait(5)

    busy = client.post(GEN, json={"prompt": "second", "size": "256x256"})
    assert busy.status_code == 409 and "already being generated" in busy.json()["detail"]
    edit = client.post("/v1/images/edits", json={"prompt": "x", "source": "whatever"})
    assert edit.status_code in (404, 409)  # source lookup may fail first; it must never run

    status = client.get("/status").json()
    assert status["busy"] is True and status["generating"] is True
    assert status["job"]["kind"] == "generate" and status["phase"] == "denoising"

    mock_engine.release.set()
    t.join(5)
    assert results["first"].status_code == 200
    assert client.get("/status").json()["busy"] is False
    assert client.post(GEN, json={"prompt": "third", "size": "256x256"}).status_code == 200


def test_out_of_memory_during_generation(client, mock_engine, tmp_env):
    mock_engine.fail_on_render = RuntimeError("HIP out of memory. Tried to allocate 4.00 GiB")
    r = client.post(GEN, json={"prompt": "x", "size": "1024x1024", "seed": 3})
    assert r.status_code == 503
    assert "Out of memory at 1024x1024" in r.json()["detail"]

    status = client.get("/status").json()
    assert status["busy"] is False
    assert any("Out of memory" in w for w in status["warnings"])
    assert any("OUT OF MEMORY" in line for line in status["logs"])

    logs = list((tmp_env["output"] / "logs").glob("*.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert "status: failed" in text and "error: Out of memory" in text
    assert not list(tmp_env["output"].glob("*.png"))

    # a later success clears the OOM banner
    mock_engine.fail_on_render = None
    assert client.post(GEN, json={"prompt": "x", "size": "256x256"}).status_code == 200
    assert not any("Out of memory" in w for w in client.get("/status").json()["warnings"])


def test_out_of_memory_while_loading_model(client, mock_engine):
    mock_engine.fail_on_load = MemoryError()
    r = client.post(GEN, json={"prompt": "x", "size": "256x256"})
    assert r.status_code == 503 and "loading the model" in r.json()["detail"]


def test_oom_detected_through_exception_chain(client, mock_engine):
    try:
        raise RuntimeError("hipErrorOutOfMemory")
    except RuntimeError as inner:
        wrapped = ValueError("decode failed")
        wrapped.__cause__ = inner
    mock_engine.fail_on_render = wrapped
    assert client.post(GEN, json={"prompt": "x", "size": "256x256"}).status_code == 503


def test_engine_crash_is_500_and_logged(make_client, mock_engine, tmp_env):
    client = make_client()
    client_no_raise = type(client)(client.app, raise_server_exceptions=False)
    mock_engine.fail_on_render = RuntimeError("something exploded")
    r = client_no_raise.post(GEN, json={"prompt": "x", "size": "256x256"})
    assert r.status_code == 500
    text = next((tmp_env["output"] / "logs").glob("*.log")).read_text()
    assert "status: failed" in text and "error: something exploded" in text
    assert client.get("/status").json()["busy"] is False  # lock released
    mock_engine.fail_on_render = None
    assert client.post(GEN, json={"prompt": "x", "size": "256x256"}).status_code == 200


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file permissions")
def test_unwritable_output_still_returns_image(client, tmp_env):
    out = tmp_env["output"]
    out.chmod(0o555)
    try:
        r = client.post(GEN, json={"prompt": "x", "size": "256x256"})
        assert r.status_code == 200 and r.json()["file"] is None and r.json()["data"][0]["b64_json"]
        r = client.post(GEN, json={"prompt": "x", "size": "256x256", "response_format": "url"})
        assert r.status_code == 400 and "could not be saved" in r.json()["detail"]
    finally:
        out.chmod(0o755)


def test_save_failure_is_non_fatal(client, tmp_env, monkeypatch):
    """Same as above but works as root: make the write itself fail."""
    from server import storage

    def boom(self, data):
        raise PermissionError("[Errno 13] Permission denied")
    monkeypatch.setattr(storage.Path, "write_bytes", boom)
    r = client.post(GEN, json={"prompt": "x", "size": "256x256"})
    assert r.status_code == 200
    body = r.json()
    assert body["file"] is None and body["data"][0]["b64_json"]
    assert any("Returning it without saving" in line for line in client.get("/status").json()["logs"])
