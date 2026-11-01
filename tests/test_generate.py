"""POST /v1/images/generations end to end against the mock engine."""
import base64
import io
import json

import pytest
from PIL import Image


def decode(resp_json) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(resp_json["data"][0]["b64_json"])))


def test_generate_returns_png_and_saves_it(client, mock_engine, tmp_env, prompts):
    r = client.post("/v1/images/generations", json={"prompt": prompts["plain"], "size": "512x512", "seed": 42})
    assert r.status_code == 200, r.text
    body = r.json()
    img = decode(body)
    assert img.format == "PNG" and img.size == (512, 512)
    assert body["seed"] == 42 and body["model"] == "base" and body["size"] == "512x512"
    assert body["steps_ran"] == 6
    assert body["file"] == f"{body['id']}.png"
    assert (tmp_env["output"] / body["file"]).read_bytes() == base64.b64decode(body["data"][0]["b64_json"])
    job = mock_engine.calls[-1].job
    assert (job.width, job.height, job.steps, job.true_cfg_scale, job.seed) == (512, 512, 6, 1.0, 42)


def test_generation_log_file(client, tmp_env, prompts):
    r = client.post("/v1/images/generations", json={"prompt": prompts["multiline"], "size": "256x256",
                                                    "seed": 7, "true_cfg_scale": 2.0,
                                                    "negative_prompt": "low quality"})
    stem = r.json()["id"]
    text = (tmp_env["output"] / "logs" / f"{stem}.log").read_text()
    assert "status: success" in text and "mode: generate" in text and "seed: 7" in text
    assert "true_cfg_scale: 2.0" in text and "steps_ran: 6" in text
    assert f"image_file: output/{stem}.png" in text
    assert "server logs during this generation:" in text and "Ran 6 denoising steps." in text
    detail = client.get(f"/v1/gallery/{stem}").json()
    assert detail["prompt"] == prompts["multiline"]
    assert detail["negative_prompt"] == "low quality"


def test_negative_prompt_not_logged_when_cfg_off(client, tmp_env):
    r = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256",
                                                    "negative_prompt": "unused"})
    assert client.get(f"/v1/gallery/{r.json()['id']}").json()["negative_prompt"] == ""


def test_same_seed_same_image(client, prompts):
    req = {"prompt": prompts["plain"], "size": "256x256", "seed": 1234}
    a = client.post("/v1/images/generations", json=req).json()
    b = client.post("/v1/images/generations", json=req).json()
    assert a["data"][0]["b64_json"] == b["data"][0]["b64_json"]
    assert a["id"] != b["id"]


def test_random_seed_is_reported(client, mock_engine):
    body = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256"}).json()
    assert isinstance(body["seed"], int) and body["seed"] == mock_engine.calls[-1].job.seed


def test_overrides(client, mock_engine):
    r = client.post("/v1/images/generations", json={
        "prompt": "x", "size": "256x256", "steps": 3, "true_cfg_scale": 1.5, "negative_prompt": "ugly"})
    assert r.status_code == 200
    job = mock_engine.calls[-1].job
    assert (job.steps, job.true_cfg_scale, job.negative_prompt) == (3, 1.5, "ugly")


def test_turbo_runs_its_saved_schedule(client, mock_engine, tmp_env):
    r = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256", "model": "turbo"})
    assert r.status_code == 200 and r.json()["steps_ran"] == 8
    log = (tmp_env["output"] / "logs" / f"{r.json()['id']}.log").read_text()
    assert "steps_requested: 8" in log and "Ran 8 denoising steps." in log
    assert mock_engine.loads == ["turbo"]


def test_turbo_rejects_a_different_step_count(client, mock_engine):
    r = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256", "model": "turbo", "steps": 4})
    assert r.status_code == 400 and "8-step schedule" in r.json()["detail"]
    ok = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256", "model": "turbo", "steps": 8})
    assert ok.status_code == 200


@pytest.mark.parametrize("key", ["unicode", "layout"])
def test_unicode_and_long_prompts_pass_through(client, mock_engine, prompts, key):
    r = client.post("/v1/images/generations", json={"prompt": prompts[key], "size": "1024x1024"})
    assert r.status_code == 200
    assert mock_engine.calls[-1].job.prompt == prompts[key]
    if key == "layout":
        json.loads(mock_engine.calls[-1].job.prompt)


def test_transparent_output_is_kept(client, mock_engine):
    mock_engine.rgba = True
    img = decode(client.post("/v1/images/generations", json={"prompt": "sticker", "size": "256x256"}).json())
    assert img.mode == "RGBA"


def test_response_format_url(client):
    body = client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256",
                                                       "response_format": "url"}).json()
    url = body["data"][0]["url"]
    assert url == f"http://testserver/v1/gallery/{body['id']}/image"
    img = client.get(url.replace("http://testserver", ""))
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_size_accepts_uppercase_and_spaces(client):
    assert client.post("/v1/images/generations", json={"prompt": "x", "size": " 256 X 256 "}).status_code == 200


def test_memory_is_freed_after_each_job(client, mock_engine):
    client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256"})
    assert mock_engine.freed >= 1
