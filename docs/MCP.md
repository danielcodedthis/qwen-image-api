# MCP server

Each API container serves the [Model Context Protocol](https://modelcontextprotocol.io):

| Container | MCP URL | Server name |
|---|---|---|
| `qwen-image-api` (Qwen-Image-2.1) | `http://localhost:8000/mcp` | `qwen-image` |
| `qwen-image-turbo-api` (2.1-Turbo) | `http://localhost:8001/mcp` | `qwen-image-turbo` |

An agent can generate and edit images and check the server's state. MCP is part of each
API (no extra service or port) and uses the same job lock, validation, logs and output
folder as the HTTP API and the web UI. Connect to whichever container is running; both
can be registered at once (they share the GPU, so only use one at a time).

- Transport: streamable HTTP, stateless.
- SDK: the official Python `mcp` package, 2.x (`mcp.server.mcpserver.MCPServer`).
- Code: `server/mcp_server.py`. Tests: `tests/test_mcp.py`, `tests/test_mcp_bridge.py`.

## Tools

| Tool | What it does |
|---|---|
| `generate_image` | Text to image. Saves to `output/`, returns id, file, URL, seed. |
| `edit_image` | Instruction editing with 1–4 reference images (ids in `output/`, or base64). |
| `get_status` | Busy?, phase and step, loaded model, VRAM/RAM, warnings, recent log lines. |
| `get_options` | Models (downloaded?, real step counts), sizes, pixel cap, edit limits. |
| `list_images` | Images in `output/`, newest first, with prompt preview and status. |
| `get_image` | One job's parameters, prompt and status (also while running or after failure); optional preview and full log. |

### `generate_image`

| Argument | Default | Notes |
|---|---|---|
| `prompt` | required | Put exact text to render in quotes. |
| `size` | `1024x1024` | `WIDTHxHEIGHT`, multiples of **32**, at most `max_pixels`. |
| `model` | the container's model | See `get_options`. |
| `steps` | config | Not allowed on Turbo: it always runs its saved 8-step schedule. |
| `true_cfg_scale` | config (1.0 = off) | Qwen-Image-2.1 is meant to run without guidance; >1 doubles the time. |
| `negative_prompt` | config | Only used when `true_cfg_scale` > 1. |
| `seed` | random | Same seed + settings → same image. |
| `wait` | `true` | `false` returns `{"status": "running", "id": ...}` at once. |
| `include_image` | `false` | Adds a ≤512 px JPEG preview (only for clients that can see images). |

Transparent PNGs: start the prompt with "This is an RGBA image with transparency." and
end it with "The image has alpha channel and the background is transparent."

Result (text content, JSON):

```json
{"status": "success", "id": "1761234567_ab12cd34", "file": "output/1761234567_ab12cd34.png",
 "image_url": "http://localhost:8000/v1/gallery/1761234567_ab12cd34/image",
 "log_url": "http://localhost:8000/v1/gallery/1761234567_ab12cd34/log",
 "saved": true, "kind": "generate", "model": "base", "size": "1024x1024",
 "seed": 42, "seconds": 48.2, "steps_ran": 20}
```

### `edit_image`

Qwen-Image-2.1 edits natively: the prompt is an **instruction**, and what it doesn't
mention is kept.

| Argument | Notes |
|---|---|
| `prompt` | "Change the background to a sunset beach", "Make the jacket red", "Replace the sign text with \"OPEN\"", "Put the mug from Picture 1 on the table in Picture 2". |
| `sources` | Ids or file names of images in `output/`, in order. |
| `images_base64` | Images as base64 or data URLs; they come **after** `sources`. |
| `size` | Omit to follow the last image's aspect ratio at about 1 MP (`edit.resolution`). |

Together `sources` + `images_base64` must give 1 to `edit.max_images` (default 4)
images, numbered Picture 1, 2, … in that order. Other arguments as for `generate_image`.
Every reference image adds to the sequence, so more images cost VRAM and time.

### Long jobs

- **`wait: true`** (default): the call returns when the image is done. Meanwhile the
  server sends MCP progress notifications every 2 s (`denoising step 12/20`). If the
  client gives up first, the job keeps running; find it with `get_status` / `list_images`.
- **`wait: false`**: returns a job id at once. Poll `get_status` until `busy` is false,
  then `get_image(id)` for the result (or the error, if it failed).

Only one job runs at a time per container. A second one fails with "Another image is
already being generated" (the HTTP API answers 409); check `get_status` first.

## Connecting

The endpoint only accepts requests whose `Host` header is `localhost`, `127.0.0.1` or
`[::1]` (DNS-rebinding protection; others get `421 Invalid Host header`). To reach it from
another machine or container, add the host to the config (`config.yaml` and/or
`config.turbo.yaml`) and restart that container:

```yaml
mcp:
  allowed_hosts: ["192.168.1.20:*", "host.docker.internal:*"]
```

There is no authentication: anyone who can reach ports 8000/8001 can use the API and MCP.
Keep them on localhost or a trusted LAN.

### Claude Code

```bash
claude mcp add --transport http qwen-image http://localhost:8000/mcp
claude mcp add --transport http qwen-image-turbo http://localhost:8001/mcp
```

### Clients that take a URL (streamable HTTP)

```json
{ "mcpServers": {
    "qwen-image":       { "type": "http", "url": "http://localhost:8000/mcp" },
    "qwen-image-turbo": { "type": "http", "url": "http://localhost:8001/mcp" } } }
```

### LM Studio Bionic (and other command-only hosts)

Bionic registers MCP servers as a command (stdio), under **Settings → Connected Apps →
Custom MCP**. Use the bundled bridge, which needs only the system Python (standard
library; no venv, pip or npm). One entry per container:

| Field | Base | Turbo |
|---|---|---|
| Name | `qwen-image` | `qwen-image-turbo` |
| Command | `python3` (or the absolute path from `command -v python3`) | same |
| Args item 1 | absolute path to `tools/mcp_stdio_bridge.py`, e.g. `/home/daniel/qwen-image-api/tools/mcp_stdio_bridge.py` | same |
| Args item 2 | (leave out: default `http://localhost:8000/mcp`) | `http://localhost:8001/mcp` |
| Env | empty | empty |

Enter the args as **separate items**: a single combined item fails with
`-32000: Connection closed`. Close classic LM Studio first; it can't run alongside Bionic.

Check the bridge before opening Bionic (the API container must be up):

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"check","version":"1"}}}' \
  | python3 tools/mcp_stdio_bridge.py http://localhost:8001/mcp
# -> {"jsonrpc":"2.0","id":1,"result":{... "serverInfo":{"name":"qwen-image-turbo",...}}}
```

Then in a Bionic chat: "Check the image server's status", then "Generate a 1024x1024
image of a red fox in snow", then "Change the background of that image to a beach".
Keep Bionic's tool-approval prompts on. A green "connected" only means the bridge started
and listed tools; the API container must also be running.

### Testing without the GPU

`python tests/mock_server.py` runs the real API (UI on :3102, API and MCP on :8102) with
a fake model, so you can set up and try a client (URL `http://localhost:8102/mcp`)
without loading Qwen-Image.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `421 Invalid Host header` | The client reaches the API under a name other than localhost. Add it to `mcp.allowed_hosts`. |
| `403 Invalid Origin header` | A browser-based client on another origin. Add its host to `mcp.allowed_hosts`. |
| `406 Not Acceptable` | The client accepts `application/json` but not `text/event-stream`, so it isn't a streamable-HTTP client. Use the bridge. |
| Bridge: `Can't reach the Qwen Image API` | That container isn't running, or the wrong port: `docker compose ps`, `curl localhost:8000/health` (or 8001). |
| `-32000: Connection closed` in Bionic | Args entered as one item, or a wrong path to the bridge. |
| "Another image is already being generated" | One job at a time. Wait (`get_status`) and retry. |
| "always runs the 8-step schedule" | Turbo: leave `steps` empty. |
| "Out of memory ..." | Smaller size, fewer reference images; see `docs/PITFALLS.md`. |
| Tool call times out but the image appears later | The client's timeout is shorter than the job. Use `wait: false` and poll. |
| `/mcp` returns 404 | `mcp.enabled: false` in that container's config, or an old image: rebuild. |
