---
name: qwen-image-api-mcp
description: Generate and edit images on the local Qwen Image servers (Qwen-Image-2.1 on :8000, 2.1-Turbo on :8001) through their MCP tools (generate_image, edit_image, get_status, get_options, list_images, get_image), and connect an MCP client to them (Claude Code, LM Studio Bionic via the stdio bridge, any streamable-HTTP client). Use when asked to create or edit images with Qwen, check those servers, or set up the connection.
---

# Using the Qwen Image MCP servers

Two containers on the user's machine (one AMD R9700 GPU), each with an MCP endpoint:

| Server | URL | Model | Steps |
|---|---|---|---|
| `qwen-image` | `http://localhost:8000/mcp` | Qwen-Image-2.1 | config (20 by default) |
| `qwen-image-turbo` | `http://localhost:8001/mcp` | Qwen-Image-2.1-Turbo | always 8 (faster) |

Images land in the repo's `output/` (Turbo: `output-turbo/`); each job also writes
`output/logs/<id>.log`. Qwen-Image-2.1 is a unified **generation and editing** model.

## 1. Connect

Check the API is up: `curl -s http://localhost:8000/health` (Turbo: 8001) → `{"ok":true}`.
If not: `cd <repo> && docker compose up -d qwen-image-api` (or `qwen-image-turbo-api`).
Both share the GPU: only one should be generating; `docker compose stop` the other if
VRAM is tight.

**Claude Code**

```bash
claude mcp add --transport http qwen-image http://localhost:8000/mcp
claude mcp add --transport http qwen-image-turbo http://localhost:8001/mcp
```

**LM Studio Bionic** (registers servers as a command, Settings → Connected Apps → Custom MCP).
Use the bundled stdlib-only bridge (no venv, pip or npm), one entry per container:

| Field | Base | Turbo |
|---|---|---|
| Name | `qwen-image` | `qwen-image-turbo` |
| Command | `python3` (or the path from `command -v python3`) | same |
| Args item 1 | absolute path to `<repo>/tools/mcp_stdio_bridge.py` | same |
| Args item 2 | (none; default `http://localhost:8000/mcp`) | `http://localhost:8001/mcp` |
| Env | empty | empty |

Args must be **separate items** (one combined item → `-32000: Connection closed`). Close
classic LM Studio first. An agent can't click Bionic's UI: print these exact values for
the user, with the resolved path (`realpath tools/mcp_stdio_bridge.py`).

Verify the bridge before the user opens Bionic:

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"check","version":"1"}}}' \
  | python3 <repo>/tools/mcp_stdio_bridge.py http://localhost:8001/mcp
```

Expect a JSON line with `"serverInfo":{"name":"qwen-image-turbo"`. "Can't reach the Qwen
Image API" means that container isn't running.

**Other clients**: a "streamable HTTP" server with the URL above, e.g.
`{"mcpServers": {"qwen-image": {"type": "http", "url": "http://localhost:8000/mcp"}}}`.

**From another machine/container**: only `localhost`/`127.0.0.1` Host headers are
accepted (else `421`). The user adds the host to that container's config
`mcp.allowed_hosts` (e.g. `"192.168.1.20:*"`) and restarts it. No auth: don't suggest
exposing the ports beyond a trusted LAN.

**Without the GPU**: `python tests/mock_server.py` serves the same API/MCP with a fake
model at `http://localhost:8102/mcp`.

## 2. Workflow

1. `get_status` first. If `busy`, another job is running (UI, curl, another agent):
   wait and poll; don't start a second one.
2. `get_options` once: sizes, `max_pixels`, steps (Turbo: fixed 8), `edit.max_images`.
3. `generate_image` / `edit_image`.
4. Report the result's `file` and `image_url`, plus the `seed`.

Prefer Turbo for quick drafts and iterations; base for final quality (more steps).

### generate_image

```json
{"prompt": "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement",
 "size": "1024x1024", "seed": 42}
```

- `size`: `WIDTHxHEIGHT`, multiples of **32**, ≤ `max_pixels` (usually 1280×1280).
- Qwen renders text well: put the exact words in quotes.
- Leave `steps`, `true_cfg_scale`, `negative_prompt` unset unless asked. CFG should stay
  1.0 (off) for Qwen-Image-2.1; >1 doubles the time and high values give worse images.
  `steps` on Turbo is an error (fixed 8-step schedule).
- Transparent PNG: start with "This is an RGBA image with transparency." and end with
  "The image has alpha channel and the background is transparent."
- `include_image: true` adds a 512 px JPEG preview (only for clients that can see images).

### edit_image

Real instruction editing. The prompt says **what to change**; everything else is kept.

```json
{"prompt": "Change the background to a sunset beach", "sources": ["1761234567_ab12cd34"]}
{"prompt": "Put the mug from Picture 1 on the wooden table in Picture 2",
 "sources": ["1761234567_ab12cd34", "1761234599_cd56ef78"]}
```

- Good instructions: change background/lighting/season/style; change colour or material
  of an object; replace or add text ("Replace the sign text with \"OPEN\""); remove an
  object; combine people/products from several images; keep identity ("the same woman").
- Images: `sources` (ids/file names in `output/`, preferred) and/or `images_base64`
  (base64 or data URLs; they come after `sources`). 1 to `edit.max_images` (default 4)
  in total; refer to them as **Picture 1, Picture 2, …** in that order. Transparency
  in the inputs is kept.
- `size`: omit to follow the **last** image's aspect ratio at about 1 MP.
- Iterate: edit the previous result's id with the next instruction.

### Long jobs

Base takes roughly tens of seconds to a few minutes; Turbo much less.
- Default `wait: true`: returns when done, with progress notifications every ~2 s. If the
  client times out, the job **keeps running**: poll `get_status`, then `list_images`.
- `wait: false`: returns `{"status": "running", "id": ...}`. Poll `get_status` every
  10–20 s until `busy` is false, then `get_image(id)` (`status` `success` or `failed`
  with `fields.error`).

### Inspecting results

- `list_images(limit, offset)`: newest first, with prompt preview, size, kind, status.
- `get_image(id, include_log=true)`: parameters, prompt, the full per-job log (timing,
  peak VRAM/RAM, references used, server lines). Use it to explain a failure.

## 3. Errors

| Tool error text | What to do |
|---|---|
| "Another image is already being generated" | Wait for `get_status.busy == false`, retry once. |
| "must be positive multiples of 32" / "exceeds max_pixels" | Use a size from `get_options.available_sizes`. |
| "always runs the 8-step schedule" | Turbo: drop `steps`. |
| "N images given; this server allows M" | Use fewer reference images. |
| "Out of memory ..." | Retry smaller, or with fewer images. Don't loop; tell the user (memory settings are theirs, see `docs/PITFALLS.md`). |
| "Reference image ...: could not read the image" | Not PNG/JPEG/WebP, or bad base64. |
| "No image named ..." | Wrong id; get it from `list_images` (note: base and Turbo have separate output folders). |
| Bridge: "Can't reach the Qwen Image API" | Container down: `docker compose up -d <service>` in the repo. |

Don't retry a failing job more than once with the same arguments.

## 4. Etiquette

- One GPU shared by both containers and the user's UIs. Don't queue many jobs; ask before
  generating more than a few images.
- Don't change configs or restart containers unless the user asks.
- `output/` and `output-turbo/` are the user's galleries: don't delete files there.
- An id from one server doesn't exist on the other (separate output folders).
