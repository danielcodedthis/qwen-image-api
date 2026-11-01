"""Gallery endpoints over output/: listing, detail with log, image/thumb/source files."""
import io
import os
import time

import pytest
from PIL import Image


def make(client, prompt="x", size="256x256"):
    return client.post("/v1/images/generations", json={"prompt": prompt, "size": size}).json()


def test_empty_gallery(client):
    assert client.get("/v1/gallery").json() == {"total": 0, "offset": 0, "limit": 60, "items": []}


def test_list_newest_first_with_pagination(client, tmp_env):
    ids = []
    for i in range(3):
        ids.append(make(client, prompt=f"prompt {i}")["id"])
        # mtime resolution: make the order unambiguous
        p = tmp_env["output"] / f"{ids[-1]}.png"
        os.utime(p, (time.time() + i, time.time() + i))
    page = client.get("/v1/gallery", params={"limit": 2}).json()
    assert page["total"] == 3
    assert [it["id"] for it in page["items"]] == [ids[2], ids[1]]
    item = page["items"][0]
    assert item["prompt"] == "prompt 2" and item["kind"] == "generate" and item["status"] == "success"
    assert item["size"] == "256x256" and item["has_log"] and item["sources"] == 0
    assert item["image_url"] == f"/v1/gallery/{ids[2]}/image"
    rest = client.get("/v1/gallery", params={"offset": 2, "limit": 2}).json()
    assert [it["id"] for it in rest["items"]] == [ids[0]]


def test_detail_image_thumb_log(client, tmp_env):
    gen = make(client, size="512x512")
    stem = gen["id"]
    d = client.get(f"/v1/gallery/{stem}").json()
    assert d["fields"]["status"] == "success" and d["fields"]["size"] == "512x512"
    assert d["prompt"] == "x" and d["log"].startswith("time: ")
    assert d["source_urls"] == [] and d["log_url"] == f"/v1/gallery/{stem}/log"

    img = client.get(d["image_url"])
    assert img.status_code == 200 and img.content == (tmp_env["output"] / f"{stem}.png").read_bytes()
    assert "immutable" in img.headers["cache-control"]

    thumb = client.get(d["thumb_url"])
    assert thumb.headers["content-type"] == "image/jpeg"
    t = Image.open(io.BytesIO(thumb.content))
    assert max(t.size) <= 384
    assert (tmp_env["output"] / ".thumbs" / f"{stem}.jpg").is_file()
    assert client.get(d["thumb_url"]).content == thumb.content  # served from cache

    log = client.get(d["log_url"])
    assert log.headers["content-type"].startswith("text/plain") and "status: success" in log.text


def test_png_without_log_is_listed(client, tmp_env):
    tmp_env["output"].mkdir(exist_ok=True)
    Image.new("RGB", (32, 32), "red").save(tmp_env["output"] / "from curl.png")
    items = client.get("/v1/gallery").json()["items"]
    assert [it["id"] for it in items] == ["from curl"]
    assert items[0]["has_log"] is False and items[0]["kind"] is None
    d = client.get("/v1/gallery/from curl").json()
    assert d["log"] is None and d["fields"] == {}
    assert client.get("/v1/gallery/from curl/log").status_code == 404
    assert client.get("/v1/gallery/from curl/image").status_code == 200


def test_failed_job_has_log_but_no_image(client, mock_engine):
    mock_engine.fail_on_render = MemoryError()
    client.post("/v1/images/generations", json={"prompt": "x", "size": "256x256"})
    assert client.get("/v1/gallery").json()["total"] == 0


def test_edit_shows_source(client, fixture_bytes):
    r = client.post("/v1/images/edits", data={"prompt": "x"},
                    files={"image": ("s.png", fixture_bytes("square_64.png"), "image/png")}).json()
    d = client.get(f"/v1/gallery/{r['id']}").json()
    assert d["fields"]["mode"] == "edit" and len(d["source_urls"]) == 1
    src = client.get(d["source_urls"][0])
    assert src.status_code == 200 and Image.open(io.BytesIO(src.content)).size == (64, 64)
    item = client.get("/v1/gallery").json()["items"][0]
    assert item["kind"] == "edit" and item["sources"] == 1
    assert client.get(f"/v1/gallery/{r['id']}/source/2").status_code == 404
    assert client.get(f"/v1/gallery/{r['id']}/source/0").status_code == 404


@pytest.mark.parametrize("path, code", [
    ("/v1/gallery/nope", 404),
    ("/v1/gallery/nope/image", 404),
    ("/v1/gallery/nope/thumb", 404),
    ("/v1/gallery/.thumbs", 400),
    # %2F is decoded before routing, so these don't even match a gallery route
    ("/v1/gallery/..%2Fconfig", 404),
    ("/v1/gallery/%2Fetc%2Fpasswd/image", 404),
    ("/v1/gallery/..%5Cconfig/image", 400),     # backslash reaches the name check
])
def test_bad_names(client, path, code):
    assert client.get(path).status_code == code


def test_non_edit_has_no_source(client):
    stem = make(client)["id"]
    assert client.get(f"/v1/gallery/{stem}/source/1").status_code == 404


def test_hidden_and_non_png_files_are_not_listed(client, tmp_env):
    out = tmp_env["output"]
    out.mkdir(exist_ok=True)
    (out / "notes.txt").write_text("hi")
    Image.new("RGB", (8, 8)).save(out / ".hidden.png")
    assert client.get("/v1/gallery").json()["total"] == 0
