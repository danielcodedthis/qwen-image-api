"""POST /v1/images/edits end to end against the mock engine: multipart uploads (one or
several `image` files), JSON with base64 / output ids, ordering, limits and errors."""
import base64
import io

import pytest
from PIL import Image

EDIT = "/v1/images/edits"


def png_of(body) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"])))


def upload(fixture_bytes, *names):
    return [("image", (n, fixture_bytes(n), "image/png")) for n in names]


def test_single_upload_auto_size(client, mock_engine, fixture_bytes, prompts, tmp_env):
    r = client.post(EDIT, data={"prompt": prompts["edit"], "seed": "5"},
                    files=upload(fixture_bytes, "landscape_200x150.png"))
    assert r.status_code == 200, r.text
    body = r.json()
    # auto size: the reference's 4:3 at about resolution² (256² in the test config), multiples of 32
    assert body["size"] == "288x224" and png_of(body).size == (288, 224)
    assert body["seed"] == 5 and body["reference_images"] == 1
    call = mock_engine.calls[-1]
    assert call.kind == "edit" and call.resolution == 256
    assert [im.size for im in call.images] == [(200, 150)]   # references passed as-is; the pipeline resizes
    stem = body["id"]
    assert (tmp_env["output"] / "sources" / f"{stem}_1.png").is_file()
    log = (tmp_env["output"] / "logs" / f"{stem}.log").read_text()
    assert "mode: edit" in log and "reference_images: 1" in log and "sources: 1: landscape_200x150.png" in log


def test_several_uploads_keep_their_order(client, mock_engine, fixture_bytes, tmp_env):
    r = client.post(EDIT, data={"prompt": "Put the circle from Picture 1 on the background of Picture 2"},
                    files=upload(fixture_bytes, "landscape_200x150.png", "square_64.png"))
    assert r.status_code == 200, r.text
    call = mock_engine.calls[-1]
    assert [im.size for im in call.images] == [(200, 150), (64, 64)]
    assert r.json()["size"] == "256x256"   # follows the LAST image (square)
    stem = r.json()["id"]
    assert Image.open(tmp_env["output"] / "sources" / f"{stem}_2.png").size == (64, 64)
    d = client.get(f"/v1/gallery/{stem}").json()
    assert d["source_urls"] == [f"/v1/gallery/{stem}/source/1", f"/v1/gallery/{stem}/source/2"]


def test_explicit_size_and_settings(client, mock_engine, fixture_bytes):
    r = client.post(EDIT, data={"prompt": "x", "size": "512x256", "steps": "9", "true_cfg_scale": "1.5"},
                    files=upload(fixture_bytes, "landscape_200x150.png"))
    assert r.status_code == 200, r.text
    job = mock_engine.calls[-1].job
    assert (job.width, job.height, job.steps, job.true_cfg_scale) == (512, 256, 9, 1.5)


def test_json_images_list_mixing_data_and_sources(client, mock_engine, fixture_b64):
    gen = client.post("/v1/images/generations", json={"prompt": "first", "size": "512x512"}).json()
    raw = fixture_b64("landscape_200x150.png")
    r = client.post(EDIT, json={"prompt": "combine", "images": [
        {"source": gen["id"]}, {"data": "data:image/png;base64," + raw}]})
    assert r.status_code == 200, r.text
    assert [im.size for im in mock_engine.calls[-1].images] == [(512, 512), (200, 150)]
    log = client.get(f"/v1/gallery/{r.json()['id']}").json()["fields"]
    assert log["sources"] == f"1: {gen['file']}; 2: image 2"


def test_json_shortcuts(client, mock_engine, fixture_b64):
    gen = client.post("/v1/images/generations", json={"prompt": "first", "size": "512x512"}).json()
    for body in ({"prompt": "x", "image": fixture_b64("square_64.png")},
                 {"prompt": "x", "source": gen["file"]},
                 {"prompt": "x", "source": f"output/{gen['file']}"}):
        r = client.post(EDIT, json=body)
        assert r.status_code == 200, (body, r.text)
        assert len(mock_engine.calls[-1].images) == 1


def test_transparency_is_kept(client, mock_engine, fixture_bytes):
    r = client.post(EDIT, data={"prompt": "Change the blue box to red"},
                    files=upload(fixture_bytes, "transparent_48x32.png"))
    assert r.status_code == 200
    ref = mock_engine.calls[-1].images[0]
    assert ref.mode == "RGBA" and ref.getpixel((0, 0))[3] == 0


def test_exif_rotation_is_applied(client, mock_engine, fixture_bytes):
    r = client.post(EDIT, data={"prompt": "x"},
                    files=[("image", ("r.jpg", fixture_bytes("exif_rotate90_40x20.jpg"), "image/jpeg"))])
    assert r.status_code == 200
    assert mock_engine.calls[-1].images[0].size == (20, 40)
    w, h = map(int, r.json()["size"].split("x"))
    assert h > w


def test_too_many_images(client, mock_engine, fixture_bytes, fixture_b64):
    r = client.post(EDIT, data={"prompt": "x"}, files=upload(fixture_bytes, *["square_64.png"] * 4))
    assert r.status_code == 400 and "limit is 3" in r.json()["detail"]
    r = client.post(EDIT, json={"prompt": "x", "images": [{"data": fixture_b64("square_64.png")}] * 4})
    assert r.status_code == 400 and "limit is 3" in r.json()["detail"]
    assert mock_engine.calls == []


def test_large_reference_is_shrunk(client, mock_engine, write_config, make_client):
    write_config({"edit": {"max_upload_mb": 20}})
    c = make_client()
    buf = io.BytesIO()
    Image.new("RGB", (3000, 1500), "teal").save(buf, format="PNG")
    r = c.post(EDIT, data={"prompt": "x"}, files=[("image", ("big.png", buf.getvalue(), "image/png"))])
    assert r.status_code == 200
    assert mock_engine.calls[-1].images[0].size == (2048, 1024)


@pytest.mark.parametrize("data, files, fragment", [
    ({"prompt": "x"}, [], "at least one reference image"),
    ({}, ["square_64.png"], "prompt"),
    ({"prompt": "x", "size": "300x300"}, ["square_64.png"], "multiples of 32"),
    ({"prompt": "x", "size": "1280x1280"}, ["square_64.png"], "max_pixels"),
    ({"prompt": "x"}, ["not_an_image.png"], "could not read the image"),
    ({"prompt": "x", "true_cfg_scale": "0.2"}, ["square_64.png"], "true_cfg_scale"),
])
def test_multipart_errors(client, mock_engine, fixture_bytes, data, files, fragment):
    r = client.post(EDIT, data=data, files=upload(fixture_bytes, *files) or None)
    assert r.status_code == 400, r.text
    assert fragment in r.json()["detail"]
    assert mock_engine.calls == []


def test_json_errors(client, mock_engine, fixture_b64):
    cases = [
        ({"prompt": "x"}, "at least one reference image"),
        ({"prompt": "x", "images": [{}]}, "exactly one of 'data' or 'source'"),
        ({"prompt": "x", "images": [{"data": "a", "source": "b"}]}, "exactly one"),
        ({"prompt": "x", "images": [{"source": "a"}], "source": "b"}, "not both"),
        ({"prompt": "x", "image": "%%%not base64%%%"}, "base64"),
        ({"prompt": "x", "image": base64.b64encode(b"hello").decode()}, "could not read"),
        ({"prompt": "x", "source": "../config"}, "not a valid image name"),
        ({"prompt": "x", "source": ".thumbs/x"}, "not a valid image name"),
    ]
    for body, fragment in cases:
        r = client.post(EDIT, json=body)
        assert r.status_code == 400, (body, r.text)
        assert fragment in r.json()["detail"], (body, r.json())
    r = client.post(EDIT, json={"prompt": "x", "source": "1700000000_deadbeef"})
    assert r.status_code == 404
    assert mock_engine.calls == []


def test_upload_size_limit(client, mock_engine):
    big = b"\x89PNG" + b"0" * (1024 * 1024 + 10)  # test config: max_upload_mb = 1
    r = client.post(EDIT, data={"prompt": "x"}, files={"image": ("big.png", big, "image/png")})
    assert r.status_code == 400 and "MB limit" in r.json()["detail"]


def test_unsupported_content_type(client):
    r = client.post(EDIT, content=b"prompt=x", headers={"content-type": "text/plain"})
    assert r.status_code == 400 and "multipart/form-data" in r.json()["detail"]


def test_turbo_edits_in_eight_steps(client, mock_engine, fixture_bytes):
    r = client.post(EDIT, data={"prompt": "x", "model": "turbo"}, files=upload(fixture_bytes, "square_64.png"))
    assert r.status_code == 200 and r.json()["steps_ran"] == 8
