"""GET /status, /config, /health, startup behaviour and the per-job log capture."""
import time

from server.monitoring import LogBuffer, logger


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_config_endpoint(client):
    c = client.get("/config").json()
    assert c["available_sizes"] == ["256x256", "512x512", "1024x1024", "1280x1280"]
    assert c["max_pixels"] == 1024 * 1024 and c["size_multiple"] == 32
    assert c["default_model"] == "base"
    assert c["edit"] == {"max_images": 3, "resolution": 256, "max_upload_mb": 1}
    assert c["mcp"] == {"enabled": True, "path": "/mcp", "name": "qwen-image-test"}
    models = {m["id"]: m for m in c["models"]}
    assert models["base"]["available"] and not models["missing"]["available"]
    assert models["base"]["total_steps"] == 6 and models["base"]["fixed_steps"] is None
    assert models["turbo"]["total_steps"] == 8 and models["turbo"]["fixed_steps"] == 8
    assert models["base"]["negative_prompt"] == "blurry" and models["base"]["true_cfg_scale"] == 1.0


def test_status_idle(client):
    s = client.get("/status").json()
    for key in ("busy", "generating", "phase", "job", "model_loaded", "loaded_model", "memory_mode", "run_on_ram",
                "step", "total_steps", "gpu", "ram", "warnings", "logs"):
        assert key in s, key
    assert s["busy"] is False and s["phase"] == "idle" and s["job"] is None
    assert s["model_loaded"] is False and s["loaded_model"] is None
    assert s["memory_mode"] == "balanced" and s["run_on_ram"] is False
    assert "error" in s["gpu"] or "total_gb" in s["gpu"]  # no rocm-smi in CI is fine


def test_status_after_generation(client):
    client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256"})
    s = client.get("/status").json()
    assert s["model_loaded"] is True and s["loaded_model"] == "base"
    assert s["step"] == 6 and s["total_steps"] == 6
    assert any("Ran 6 denoising steps." in line for line in s["logs"])


def test_ram_warning_threshold(write_config, make_client):
    write_config({"monitoring": {"ram_warn_percent": 0.001}})
    s = make_client().get("/status").json()
    assert any("System RAM at" in w for w in s["warnings"])


def test_eager_load_at_startup(write_config, make_client, mock_engine):
    write_config({"startup": {"lazy_load": False}})
    client = make_client()
    for _ in range(50):
        if mock_engine.is_loaded() and not client.get("/status").json()["busy"]:
            break
        time.sleep(0.02)
    assert mock_engine.loads == ["base"]


def test_cors_headers_for_the_ui(client):
    r = client.get("/status", headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_log_buffer_cursor_survives_wraparound():
    """Regression: the old cursor was len(buffer), so once 200 lines had been logged the
    per-job .log files silently lost their 'server logs' section."""
    buf = LogBuffer(maxlen=5)
    logger.addHandler(buf)
    try:
        for i in range(10):
            logger.info(f"before {i}")
        cursor = buf.cursor()
        logger.info("during 1")
        logger.info("during 2")
        lines = buf.since(cursor)
        assert [line.split(" ", 2)[2] for line in lines] == ["during 1", "during 2"]
        assert len(buf.recent(50)) == 5
    finally:
        logger.removeHandler(buf)
