"""Request validation: sizes, pixel caps, ranges, bad input. All must be 400s that
never reach the engine."""
import pytest

GEN = "/v1/images/generations"


@pytest.mark.parametrize("size, fragment", [
    ("1000x1000", "multiples of 32"),
    ("0x512", "multiples of 32"),
    ("512x8", "multiples of 32"),
    ("528x512", "multiples of 32"),     # a multiple of 16 is not enough for Qwen-Image-2.1
    ("big", "WIDTHxHEIGHT"),            # used to silently become 1024x1024
    ("512", "WIDTHxHEIGHT"),
    ("512x512x3", "WIDTHxHEIGHT"),
    ("-512x512", "WIDTHxHEIGHT"),
    ("", "WIDTHxHEIGHT"),
    ("1280x1280", "max_pixels"),        # test config caps at 1024*1024
    ("2048x544", "max_pixels"),
])
def test_bad_sizes(client, mock_engine, size, fragment):
    r = client.post(GEN, json={"prompt": "x", "size": size})
    assert r.status_code == 400, r.text
    assert fragment in r.json()["detail"]
    assert mock_engine.calls == []


def test_exactly_at_the_pixel_cap_is_allowed(client):
    # 2048 x 512 = 1,048,576 = the test config's max_pixels
    assert client.post(GEN, json={"prompt": "x", "size": "2048x512"}).status_code == 200


def test_one_over_the_pixel_cap(write_config, make_client):
    write_config({"generation": {"max_pixels": 512 * 512 - 1}})
    c = make_client()
    r = c.post(GEN, json={"prompt": "x", "size": "512x512"})
    assert r.status_code == 400 and "max_pixels" in r.json()["detail"]


@pytest.mark.parametrize("body, fragment", [
    ({}, "prompt"),
    ({"prompt": ""}, "prompt"),
    ({"prompt": "   \n "}, "prompt must not be empty"),
    ({"prompt": "x", "steps": 0}, "steps"),
    ({"prompt": "x", "steps": 201}, "steps"),
    ({"prompt": "x", "true_cfg_scale": 0.5}, "true_cfg_scale"),
    ({"prompt": "x", "true_cfg_scale": 11}, "true_cfg_scale"),
    ({"prompt": "x", "seed": -1}, "seed"),
    ({"prompt": "x", "seed": 2**63}, "seed"),
    ({"prompt": "x", "steps": "many"}, "steps"),
    ({"prompt": "x", "n": 2}, "n"),
    ({"prompt": "x", "response_format": "png"}, "response_format"),
    ({"prompt": 5}, "prompt"),
])
def test_bad_fields_are_400_with_readable_detail(client, mock_engine, body, fragment):
    r = client.post(GEN, json=body)
    assert r.status_code == 400, r.text
    assert isinstance(r.json()["detail"], str) and fragment in r.json()["detail"]
    assert mock_engine.calls == []


def test_unknown_model(client):
    r = client.post(GEN, json={"prompt": "x", "size": "256x256", "model": "nope"})
    assert r.status_code == 400 and "Unknown model 'nope'" in r.json()["detail"]


def test_model_folder_missing(client):
    r = client.post(GEN, json={"prompt": "x", "size": "256x256", "model": "missing"})
    assert r.status_code == 400 and "Model folder not found" in r.json()["detail"]


def test_malformed_json_body(client):
    r = client.post(GEN, content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_unknown_fields_are_ignored(client):
    assert client.post(GEN, json={"prompt": "x", "size": "256x256", "quality": "hd"}).status_code == 200


def test_seed_at_upper_bound_is_ok(client, mock_engine):
    r = client.post(GEN, json={"prompt": "x", "size": "256x256", "seed": 2**63 - 1})
    assert r.status_code == 200 and mock_engine.calls[-1].job.seed == 2**63 - 1
