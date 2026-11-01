"""The embedded MCP server at /mcp (streamable HTTP), driven over JSON-RPC."""
import base64
import json
import time

from conftest import MCPClient


def text_json(result) -> dict:
    return json.loads(next(c["text"] for c in result["content"] if c["type"] == "text"))


def test_initialize_and_list_tools(mcp):
    _, msg = mcp.initialize()
    info = msg["result"]
    assert "unified generation AND editing" in info["instructions"]
    assert info["serverInfo"]["name"] == "qwen-image-test"
    _, msg = mcp.rpc("tools/list")
    tools = {t["name"]: t for t in msg["result"]["tools"]}
    assert set(tools) == {"generate_image", "edit_image", "get_status", "get_options",
                          "list_images", "get_image"}
    gen = tools["generate_image"]["inputSchema"]
    assert gen["required"] == ["prompt"]
    assert gen["properties"]["steps"]["anyOf"][0]["maximum"] == 200
    assert tools["get_status"]["annotations"]["readOnlyHint"] is True


def test_generate_image_tool(mcp, mock_engine, tmp_env):
    result = mcp.call("generate_image", {"prompt": "a fox", "size": "256x256", "seed": 9})
    assert result["isError"] is False
    out = text_json(result)
    assert out["status"] == "success" and out["seed"] == 9 and out["size"] == "256x256"
    assert out["file"] == f"output/{out['id']}.png"
    assert out["image_url"] == f"http://testserver/v1/gallery/{out['id']}/image"
    assert (tmp_env["output"] / f"{out['id']}.png").is_file()
    assert all(c["type"] == "text" for c in result["content"])  # no image unless asked


def test_include_image_returns_small_jpeg(mcp):
    result = mcp.call("generate_image", {"prompt": "a fox", "size": "1024x1024", "include_image": True})
    images = [c for c in result["content"] if c["type"] == "image"]
    assert len(images) == 1 and images[0]["mimeType"] == "image/jpeg"
    from PIL import Image
    import io
    assert max(Image.open(io.BytesIO(base64.b64decode(images[0]["data"]))).size) == 512


def test_edit_image_tool_from_previous_output(mcp, mock_engine):
    first = text_json(mcp.call("generate_image", {"prompt": "a fox", "size": "512x512"}))
    result = mcp.call("edit_image", {"prompt": "Make the fox wear a red scarf", "sources": [first["id"]]})
    out = text_json(result)
    assert out["kind"] == "edit" and out["reference_images"] == 1 and out["size"] == "256x256"
    assert mock_engine.calls[-1].kind == "edit"


def test_edit_image_tool_mixed_sources_and_base64(mcp, mock_engine, fixture_b64):
    first = text_json(mcp.call("generate_image", {"prompt": "a fox", "size": "512x512"}))
    out = text_json(mcp.call("edit_image", {"prompt": "Put Picture 1 into Picture 2",
                                            "sources": [first["id"]],
                                            "images_base64": [fixture_b64("landscape_200x150.png")],
                                            "size": "256x256"}))
    assert out["status"] == "success" and out["reference_images"] == 2
    assert [im.size for im in mock_engine.calls[-1].images] == [(512, 512), (200, 150)]


def test_edit_image_tool_limit(mcp, fixture_b64):
    result = mcp.call("edit_image", {"prompt": "x", "images_base64": [fixture_b64("square_64.png")] * 4})
    assert result["isError"] is True and "allows 3" in result["content"][0]["text"]


def test_tool_errors_are_reported_not_raised(mcp, mock_engine):
    cases = [
        ("generate_image", {"prompt": "x", "size": "528x528"}, "multiples of 32"),
        ("generate_image", {"prompt": "x", "model": "turbo", "steps": 4}, "8-step schedule"),
        ("generate_image", {"prompt": "x", "model": "nope"}, "Unknown model"),
        ("generate_image", {"prompt": "  "}, "prompt must not be empty"),
        ("edit_image", {"prompt": "x"}, "at least one image"),
        ("edit_image", {"prompt": "x", "sources": ["nothing_here"]}, "No image named"),
        ("get_image", {"id": "../etc"}, "not a valid image name"),
        ("get_image", {"id": "1700000000_00000000"}, "No image or log"),
    ]
    for tool, args, fragment in cases:
        result = mcp.call(tool, args)
        assert result["isError"] is True, (tool, args)
        assert fragment in result["content"][0]["text"], (tool, result)
    assert mock_engine.calls == []


def test_schema_validation_rejects_out_of_range(mcp):
    result = mcp.call("generate_image", {"prompt": "x", "steps": 500})
    assert result["isError"] is True


def test_oom_is_a_tool_error(mcp, mock_engine):
    mock_engine.fail_on_render = RuntimeError("out of memory")
    result = mcp.call("generate_image", {"prompt": "x", "size": "256x256"})
    assert result["isError"] is True and "Out of memory" in result["content"][0]["text"]


def test_background_job_then_poll(mcp, mock_engine):
    mock_engine.hold = True
    started = text_json(mcp.call("generate_image", {"prompt": "slow", "size": "256x256", "wait": False}))
    assert started["status"] == "running"
    job_id = started["id"]
    assert mock_engine.started.wait(5)

    status = mcp.call("get_status")["structuredContent"]
    assert status["busy"] is True and status["job"]["id"] == job_id
    running = text_json(mcp.call("get_image", {"id": job_id}))
    assert running["status"] == "running"
    busy = mcp.call("generate_image", {"prompt": "other", "size": "256x256"})
    assert busy["isError"] is True and "already being generated" in busy["content"][0]["text"]

    mock_engine.release.set()
    for _ in range(100):
        if not mcp.call("get_status")["structuredContent"]["busy"]:
            break
        time.sleep(0.02)
    done = text_json(mcp.call("get_image", {"id": job_id, "include_log": True}))
    assert done["status"] == "success" and done["exists"] is True
    assert done["prompt"] == "slow" and "status: success" in done["log"]


def test_status_options_and_listing(mcp):
    mcp.call("generate_image", {"prompt": "listed", "size": "256x256"})
    status = mcp.call("get_status", {"log_lines": 3})["structuredContent"]
    assert status["busy"] is False and status["loaded_model"] == "base" and len(status["logs"]) == 3
    options = mcp.call("get_options")["structuredContent"]
    assert options["default_model"] == "base" and "256x256" in options["available_sizes"]
    assert options["edit"]["max_images"] == 3
    listing = mcp.call("list_images", {"limit": 5})["structuredContent"]
    assert listing["total"] == 1 and listing["items"][0]["prompt"] == "listed"


def test_dns_rebinding_protection(client):
    m = MCPClient(client)
    resp, _ = m.rpc("tools/list", headers={"Host": "attacker.example"})
    assert resp.status_code == 421
    resp, _ = m.rpc("tools/list", headers={"Origin": "http://attacker.example"})
    assert resp.status_code == 403


def test_mcp_can_be_disabled(write_config, make_client):
    write_config({"mcp": {"enabled": False}})
    c = make_client()
    assert c.post("/mcp", json={}, headers=MCPClient.HEADERS).status_code in (404, 405)
