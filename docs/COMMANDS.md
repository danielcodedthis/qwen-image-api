# Command reference

Services in `docker-compose.yml`: `qwen-image-api` (`:8000`), `qwen-image-ui`
(`:3000`), `qwen-image-turbo-api` (`:8001`), `qwen-image-turbo-ui` (`:3001`).

## Docker — build & run

```bash
# Rebuild — required after ANY change to server/, requirements.txt, Dockerfile or Dockerfile.turbo.
# `restart` alone does NOT pick up code changes (COPY bakes files in at build time).
docker compose build

# Bring up everything (after a full `down`, always omit the service name
# so every service, including both UIs, comes back, not just one)
docker compose up -d --build

# Bring up one service only (only safe if siblings are already running)
docker compose up -d --build qwen-image-api
docker compose up -d --build qwen-image-turbo-api

# Force the Turbo image to pull the latest diffusers (the pip layer is cached
# otherwise, so a plain rebuild keeps the old diffusers). Slow: re-downloads.
docker compose build --no-cache qwen-image-turbo-api

# Pin Turbo's diffusers to a specific commit instead of main
docker compose build --build-arg DIFFUSERS_REF=<commit-sha> qwen-image-turbo-api
```

**Config and UI edits don't need a rebuild:**

```bash
nano config.yaml && docker compose restart qwen-image-api          # base settings
nano config.turbo.yaml && docker compose restart qwen-image-turbo-api   # Turbo settings
chmod -R a+rX web                                                  # UI: just refresh the browser
```

## Docker — stop, restart, remove

```bash
# Full teardown — use after a crash/OOM to guarantee a clean GPU/allocator state
docker compose down

# Restart existing container(s) — no rebuild, less reliable than down+up for
# clearing stuck VRAM allocations after a crash
docker compose restart

docker compose stop
docker compose start
```

**One GPU, two models.** A loaded model stays in VRAM until its container
stops. Before using the other model:

```bash
docker compose stop qwen-image-api            # then use Turbo on :3001
docker compose stop qwen-image-turbo-api      # then use the base model on :3000
docker compose start qwen-image-turbo-api     # bring one back
```

## Docker — status & inspection

```bash
docker compose ps
docker stats qwen-image-api                 # live CPU/RAM/network
docker exec -it qwen-image-api bash         # shell in
docker exec qwen-image-api grep -A3 "CORSMiddleware" /app/server/app.py
docker inspect qwen-image-api               # mounts, env, devices
```

## Docker — logs

```bash
docker compose logs -f qwen-image-api       # follow live
docker compose logs -f qwen-image-turbo-api
docker logs -f qwen-image-api               # same, without compose context
docker logs --tail 100 qwen-image-turbo-api
docker logs -f -t qwen-image-api            # with timestamps

# After a Turbo rebuild: library versions, and the check that the 8-step
# schedule really ran (look for "Ran 8 denoising steps.")
docker logs qwen-image-turbo-api 2>&1 | grep -E "Versions:|Ran [0-9]+ denoising|WARNING|ERROR"
```

The app's own `/status` endpoint returns recent log lines, live GPU/RAM stats,
generation progress and active warnings:

```bash
curl http://localhost:8000/status           # base: busy/phase/progress, VRAM/RAM, warnings, log lines
curl http://localhost:8001/status           # Turbo
curl http://localhost:8000/config           # sizes, models (real step counts), edit limits
curl http://localhost:8000/health           # liveness, no GPU query
curl "http://localhost:8000/v1/gallery?limit=5"     # newest images with prompt previews
curl http://localhost:8000/v1/gallery/<id>/log      # one job's log
curl http://localhost:8001/config           # sizes, models, per-model steps/guidance
```

## Requests

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox sitting in a snowy forest", "size": "1024x1024"}' \
  -o response.json

# Turbo
curl -X POST http://localhost:8001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox sitting in a snowy forest", "size": "1024x1024"}' \
  -o response.json

# Decode the PNG from the response
python3 -c "import json,base64;d=json.load(open('response.json'));open('out.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))"
```

### Edits

```bash
# one image (multipart); repeat -F image=@... for Picture 2, 3, ...
curl -X POST http://localhost:8000/v1/images/edits \
  -F image=@photo.jpg -F prompt="Change the background to a sunset beach" -o edit.json

# images already in output/, by id, in order (JSON)
curl -X POST http://localhost:8000/v1/images/edits -H "Content-Type: application/json" \
  -d '{"prompt": "Put the mug from Picture 1 on the table in Picture 2",
       "images": [{"source": "1761234567_ab12cd34"}, {"source": "1761234599_cd56ef78"}]}' -o edit.json
```

A `400` means a bad size (not a multiple of 32), model, setting or image; a `404` an
unknown source image; a `409` that another job is running; a `503` out of memory.

## MCP

```bash
# through the stdio bridge (what Bionic runs): initialize + status (Turbo: add http://localhost:8001/mcp)
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"check","version":"1"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_status","arguments":{}}}' \
  | python3 tools/mcp_stdio_bridge.py

claude mcp add --transport http qwen-image http://localhost:8000/mcp        # Claude Code
claude mcp add --transport http qwen-image-turbo http://localhost:8001/mcp
```

See `docs/MCP.md`.

## Tests (no GPU needed)

```bash
pip install -r requirements-dev.txt
pytest -q                                   # whole suite
pytest -q tests/test_edit.py -k several     # a subset
python tests/mock_server.py                 # real API + UI + MCP with a fake model (UI :3102, API :8102)
python tests/fixtures/make_fixtures.py      # regenerate the synthetic test images
```

## GPU (ROCm)

```bash
rocm-smi --showmeminfo vram                      # quick VRAM summary
watch -n 1 rocm-smi --showmeminfo vram           # live view
rocm-smi                                          # full status: temp, power, clocks

# Sanity check the GPU is visible at all, outside this project's containers
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  rocm/pytorch:latest rocm-smi
```

## System RAM

```bash
free -h            # model weights are memory-mapped: look at buff/cache, not just "used"
watch -n 1 free -h
ps aux --sort=-%mem | head -15
sudo dmesg | grep -i "killed process"            # confirm OOM-killer activity
sudo journalctl -k -b | grep -i oom
```

## Files and permissions

```bash
mkdir -p models output output-turbo cache/miopen   # create these yourself, as your user
ls -ld output-turbo                                # should be owned by you, not root
sudo chown -R "$USER":"$USER" output-turbo         # fix a root-owned mount folder
chmod -R a+rX web                                  # nginx 403 "Permission denied" fix
docker logs qwen-image-turbo-ui 2>&1 | tail -5     # nginx errors
```

## Model downloads

```bash
huggingface-cli download Qwen/Qwen-Image-2.1 --local-dir ./models/Qwen-Image-2.1
huggingface-cli download Qwen/Qwen-Image-2.1-Turbo --local-dir ./models/Qwen-Image-2.1-Turbo
# (newer huggingface_hub: `hf download ...` takes the same arguments)
```

See `docs/MODEL_SETUP.md` for details.

## Debugging sequence (this project, in order)

```bash
docker compose logs --tail 200 qwen-image-api    # 1. what happened (OOM entries show VRAM/RAM at failure)
docker compose down && docker compose up -d --build   # 2. clean restart
watch -n 1 rocm-smi --showmeminfo vram           # 3a. watch VRAM
watch -n 1 free -h                                # 3b. watch RAM
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox sitting in a snowy forest", "size": "1024x1024"}' \
  -o response.json                                # 4. test request
```

## Slow `docker build`

If the build output starts with `buildx Docker CLI plugin not found: falling
back to the classic builder`, install the buildx plugin (Arch:
`sudo pacman -S docker-buildx`). The classic builder is slow at committing
layers on top of the large `rocm/pytorch` image. Also check free disk space:

```bash
df -h / /var/lib/docker
docker system df
```

## Git

```bash
git status                          # check before every commit
git add .
git commit -m "describe what changed"
git push

git log --oneline                   # quick history
git diff                            # unstaged changes
git diff --staged                   # staged changes not yet committed
```

First-time setup on a new machine:

```bash
sudo pacman -S git github-cli
gh auth login                       # browser-based device flow
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
