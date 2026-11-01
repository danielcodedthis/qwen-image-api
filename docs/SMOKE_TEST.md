# Manual smoke test on the real model

The automated tests (`pytest`) never load the model. This checklist is for after a
rebuild or a change to `server/engine/`, on the real machine and GPU. Do it once per
container (base on :8000, Turbo on :8001; replace the port below). Stop at the first
failure: the job's log in `output/logs/` and `docker logs <container>` say why.

Only one model should be loaded at a time: `docker compose stop qwen-image-turbo-api`
while testing the base container, and the reverse.

## 0. Start clean

```bash
docker compose down
docker compose up -d --build
docker logs -f qwen-image-api        # Ctrl+C once you see "Application startup complete"
curl -s localhost:8000/health        # {"ok":true}
curl -s localhost:8000/config | python3 -m json.tool | head -40
```

Check: the model is `"available": true`; on Turbo `"fixed_steps": 8`.

## 1. Text to image (API)

```bash
curl -s -X POST localhost:8000/v1/images/generations -H "Content-Type: application/json" \
  -d '{"prompt":"a red fox sitting in a snowy forest","size":"1024x1024","seed":42}' -o gen.json
python3 -c "import json,base64;d=json.load(open('gen.json'));open('gen.png','wb').write(base64.b64decode(d['data'][0]['b64_json']));print(d['id'],d['seed'],d['steps_ran'],d['generation_seconds'])"
```

Check:
- `docker logs qwen-image-api | tail -20` shows `Versions: ...`, `Pipeline loaded and
  ready.`, `Ran N denoising steps.` (Turbo: 8) and no warning about a different count.
- `gen.png` looks right; `output/<id>.png` and `output/logs/<id>.log` exist.
- Same request again with the same seed gives the same image.

## 2. Edit with one image (API)

```bash
curl -s -X POST localhost:8000/v1/images/edits \
  -F image=@gen.png -F prompt="Change the background to a sunset beach" -F seed=7 -o edit.json
python3 -c "import json,base64;d=json.load(open('edit.json'));open('edit.png','wb').write(base64.b64decode(d['data'][0]['b64_json']));print(d.get('size'),d.get('reference_images'),d.get('steps_ran'),d.get('detail'))"
```

Check:
- `edit.png` is the same fox on a beach at sunset (an edit, not a new picture).
- No `Input type (BFloat16) and bias type (float)` error: that would mean the VAE wrapper
  isn't applied to encode (it matters with `vae_on_cpu: true` or `vae_fp32: true`).
- Peak VRAM in `output/logs/<id>.log` (`peak_vram_gb`). Record it in `CLAUDE.md`.
- `output/sources/<id>_1.png` is the reference.

## 3. Edit with several images

```bash
ID=$(python3 -c "import json;print(json.load(open('gen.json'))['id'])")
curl -s -X POST localhost:8000/v1/images/edits -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Put the fox from Picture 1 into the scene of Picture 2\",\"images\":[{\"source\":\"$ID\"},{\"source\":\"$(python3 -c "import json;print(json.load(open('edit.json'))['id'])")\"}]}" \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('id'),d.get('size'),d.get('reference_images'),d.get('detail'))"
```

Then try 4 images (the default `edit.max_images`) once, and record its peak VRAM. If it
runs out of memory, lower `edit.max_images` or `edit.resolution` in the config.

Transparency: generate with "This is an RGBA image with transparency. A cute cartoon
dragon sticker. The image has alpha channel and the background is transparent." The
PNG should have a transparent background; then edit it ("Make the dragon blue") and check
the background stays transparent.

## 4. Web UI

Open http://localhost:3000 (Turbo: http://localhost:3001; hard refresh: Ctrl+Shift+R).
The title says "Qwen Image" or "Qwen Image Turbo".

- **Create**: generate 1024x1024. Progress overlay shows steps and time left (Turbo:
  "Step x of 8"); the result shows seed and time.
- **Edit this** on the result → Edit with it as Picture 1. Instruction, **Edit image**.
  **Compare** flips between reference and result.
- Drop two photos onto the Edit panel: they show as Picture 1 and Picture 2; remove one.
- **Gallery**: new images have thumbnails; open one → full image, facts, reference
  images, and the log text below. Refresh: it stays on the same image. **Reuse settings**
  brings prompt, seed and references back.
- Older images (from before this update) are listed too, with "No log file".
- Run a job with curl and try **Generate** at the same time: the button is disabled
  with "Another job is running".

## 5. MCP

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"1"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_status","arguments":{}}}' \
  | python3 tools/mcp_stdio_bridge.py http://localhost:8000/mcp
```

Check: an `initialize` result with `"name":"qwen-image"` (Turbo: `qwen-image-turbo`), then
a status result. Then from your agent (Bionic or Claude Code, see `docs/MCP.md`): check
status, generate an image, then edit it by id.

## 6. Failure paths

- A size that isn't a multiple of 32 (`1000x1000`): clean `400`.
- Turbo with `"steps": 4`: `400` explaining the fixed 8-step schedule.
- Start a long job, then a second: `409`.
- (Optional) a size or image count you know runs out of VRAM: `503`, an `OUT OF MEMORY`
  block in the log with VRAM/RAM numbers, a red banner in the UI, and the next small job
  clears it.

Record anything new in `docs/PITFALLS.md`.
