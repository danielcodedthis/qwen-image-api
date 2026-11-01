---
name: qwen-image-api-internals
description: How the qwen-image-api repo works inside (two FastAPI containers around diffusers' QwenImage21Pipeline on ROCm: base Qwen-Image-2.1 and 2.1-Turbo; web UI; gallery; native multi-image editing; MCP; tests), so an agent can pinpoint a bug or make a change. Use when debugging or modifying qwen-image-api, its editing, memory options, logs, UI or test suite.
---

# qwen-image-api internals

Two FastAPI containers running the same `server/` package around diffusers'
`QwenImage21Pipeline` on AMD ROCm, Radeon AI PRO R9700 (gfx1201, 32 GB VRAM), 96 GB RAM,
Arch Linux, Docker:

| Container | Model | API (+ `/mcp`) | UI | Config | Output |
|---|---|---|---|---|---|
| `qwen-image-api` | Qwen-Image-2.1 | :8000 | :3000 | `config.yaml` | `output/` |
| `qwen-image-turbo-api` | Qwen-Image-2.1-Turbo | :8001 | :3001 | `config.turbo.yaml` | `output-turbo/` |

Read `CLAUDE.md` (hard constraints, known-good config) and `docs/PITFALLS.md` (solved
problems) before debugging. Never fix a failing generation by guessing: read the job's log.

## Map: where things live

| File | Owns | Imports torch? |
|---|---|---|
| `server/app.py` | `create_app(settings=None, engine=None)`: wiring, CORS, error handlers, routers, MCP mount, lifespan (MCP session manager, optional background preload). | no |
| `server/__main__.py` | `python -m server` = container CMD (uvicorn on 8000 inside). | no |
| `server/settings.py` | `DEFAULTS`, `load_settings()` → frozen `Settings` + `ModelSettings`. Reads each model's `model_index.json` `sample_sigmas` → `fixed_steps` (Turbo: 8). Env: `CONFIG_PATH`, `OUTPUT_DIR`. | no |
| `server/service.py` | `ImageService`: validation against config, the **one-job lock**, `generate()` / `edit()` / `preload()`, `_run()` (log lines, OOM mapping, PNG save, per-job log), `status()`, `public_config()`. Errors: `InvalidRequest` 400, `NotFound` 404, `Busy` 409, `OutOfMemory` 503. | no |
| `server/engine/__init__.py` | The `Engine` protocol, `RenderJob`, `RenderResult`, `JobHooks`. | no |
| `server/engine/qwen.py` | `QwenEngine`: load (bf16, torchao int8 transformer, `device_map="balanced"` + `max_memory`, VAE slicing), `place_vae` (vae_fp32 / vae_on_cpu for encode AND decode), `text_to_image`, `edit`. | **yes** |
| `server/imaging.py` | Upload decoding (base64/data URL, EXIF, **alpha kept**, size/bomb limits), `auto_edit_size` (= the pipeline's own rule), `shrink_reference`, PNG and thumbnail encoding. | no |
| `server/storage.py` | `OutputStore`: `output/<id>.png`, `output/logs/<id>.log`, `output/sources/<id>_<n>.png`, `output/.thumbs/`, `check_stem`, `parse_log`, `list_images`. | no |
| `server/monitoring.py` | `logger`, `LogBuffer` (seq-numbered ring buffer), `Monitor` (progress, phases, peaks, pressure warnings, last OOM, Qwen-specific OOM hints), `get_gpu_stats` (rocm-smi, 0.5 s cache), `get_ram_stats`, `is_oom_error`. | lazily |
| `server/routes/{images,gallery,system,deps}.py` | HTTP endpoints; `deps.py` maps `ServiceError` → status code and validation errors → 400. | no |
| `server/mcp_server.py` | MCP tools on `MCPServer` (SDK **2.x**), `/mcp`, stateless streamable HTTP, DNS-rebinding host allowlist; name from `mcp.name`. | no |
| `web/` | `index.html`, `css/app.css`, `js/*.js` ES modules. Same files for both UIs. | – |
| `tools/mcp_stdio_bridge.py` | stdlib-only stdio↔HTTP relay for command-only MCP hosts (Bionic). | – |
| `tests/` | pytest suite, `mock_engine.py`, `mock_server.py`, `fixtures/`. | optional |

## Life of a request

1. Route parses input (`schemas.py`; bad input → **400** with a readable `detail`).
2. `ImageService.generate/edit` checks config-dependent things: size multiple of **32**
   and ≤ `max_pixels`; model exists and its folder exists; `steps` not given for a
   fixed-schedule model; fills defaults (model's `total_steps`, `true_cfg_scale`,
   negative prompt; random 32-bit seed). Edits also load each reference
   (`load_source`: upload / base64 / `output/` id; EXIF transpose, keep alpha, shrink to
   ≤2048 px), check the count ≤ `edit.max_images`, and pick the output size
   (explicit, or `auto_edit_size` from the **last** reference). **Nothing reaches the
   engine before this passes.**
3. `_launch` takes the lock **non-blocking** (busy → 409), makes the stem
   `<unix time>_<8 hex>`, `monitor.begin(...)`. `wait=False` (MCP) runs `_run` in a thread.
4. `_run`: log cursor, "Generating/Editing [...]" line, saves edit references to
   `output/sources/<id>_<n>.png`, calls the engine with `self.monitor` as hooks.
   Phases: `loading → denoising (step 0 = prompt/vision encoding) → decoding → saving`.
5. Success: steps check (`Ran N denoising steps.` / warning), PNG (RGBA kept), save
   (non-fatal), clear last OOM, per-job log, `engine.free_memory()`.
   Failure: OOM (type or message markers through `__cause__/__context__`) → `report_oom`
   + free memory + log + **503**; else `logger.exception`, log, re-raise (500).

## The engine

`QwenEngine._call` builds `pipe(prompt, width, height, num_inference_steps, generator=
torch.Generator("cpu").manual_seed(seed), callback_on_step_end=...)`; adds
`true_cfg_scale` + `negative_prompt` (`" "` when empty) only when CFG > 1. Edit adds
`image=[...]` (the references, in order) and `output_resolution=edit.resolution`.

Pipeline facts (diffusers `pipeline_qwenimage21.py`):
- `image` = list of PIL images (the model supports up to 10). Each is converted to RGBA,
  resized to about `output_resolution`² (aspect kept), read by the Qwen3-VL text encoder
  as vision context ("Picture N") **and** VAE-encoded into latent tokens prepended to the
  noise. More images → longer sequence → more VRAM/time. KV cache (`use_kv_cache=True`,
  default) reuses text/reference context across steps.
- Without height/width the output follows the last image's aspect at
  `output_resolution`². We compute that ourselves (`imaging.auto_edit_size`, tested for
  parity against `calculate_dimensions`) so we can validate it and log it.
- Width/height are floored to multiples of 32 → the API requires multiples of 32.
- `sample_sigmas` in the checkpoint config overrides `num_inference_steps` (Turbo: 8).
  `planned` steps = `len(pipe.config.sample_sigmas)` if set.
- The VAE (`AutoencoderKLQwenImage21`) takes 4-channel RGBA. `_encode_vae_image` sends
  it the image in the transformer's device/dtype (GPU bf16) and the pipeline concatenates
  the result with GPU latents. `place_vae(pipe, device)` therefore converts encode/decode
  inputs to the VAE's device/dtype and encode's output back to the caller's
  (`vae_on_cpu` also removes accelerate hooks first). See PITFALLS.

## Per-job log format (`output/logs/<id>.log`)

```
time / status (success|failed) / mode (generate|edit) / model / size
steps_requested / true_cfg_scale / seed / memory_mode
reference_images / sources ("1: photo.png; 2: 1761..._ab12.png") / edit_resolution   (edits)
duration_seconds / peak_vram_gb / peak_ram_gb
steps_ran / cfg / image_file / image_saved          (success)   error: ...  (failure)

prompt:
<prompt>

negative_prompt:
<only when CFG > 1, else (none)>

server logs during this generation:
<every log line emitted during the job>
```

Images from before this update have no log; the gallery and `get_image` handle that.

## Common changes: where to make them

| Change | Where | Also update |
|---|---|---|
| New config key | `settings.DEFAULTS` + property + `_validate` | both config files, README, `tests/test_settings.py` |
| New request field | `schemas.py` → `service.py` (`build_job`/`edit`) → `RenderJob` → `engine/qwen.py: _call` | `routes/images.py` `_FORM_FIELDS`, MCP tool args, mock engine, tests |
| New endpoint | a router in `server/routes/`, `include_router` in `app.py` | README API table, tests |
| New MCP tool | `mcp_server.py: build_mcp` (structured results: `-> dict[str, Any]` + `structured_output=True`) | `docs/MCP.md`, `skills/qwen-image-api-mcp`, `tests/test_mcp.py` |
| UI control | `web/index.html` (`data-modes="create edit"` decides visibility) + `web/js/*.js` | – (refresh) |
| Different behaviour per container | config keys (each container has its own file) | both configs |

## UI (`web/js/`)

`main.js` (wiring, views, title "Qwen Image" / "Qwen Image Turbo") · `router.js`
(`#create|#edit|#gallery/<id>`) · `events.js` (bus) · `api.js` (`:3001` → `:8001`, else
`:8000`, or `?api=PORT`; `blobToDataUrl`) · `status.js` (1 s `/status` poll) ·
`controls.js` (models with real step counts, steps field disabled for fixed schedules,
CFG, seed, negative prompt hint "ignored while CFG is off", sizes, `autoEditSize` mirror;
`localStorage` keys `qwen.*`) · `jobs.js` (run, progress, ETA, "another job is running")
· `result.js` (stage, history, Compare with the last reference, Edit this) · `create.js`
(prompt, "Transparent background" helper) · `edit.js` (ordered references, add/remove,
drop/paste many; sends one JSON request: `{source}` for output/ images, `{data}` for the
rest) · `gallery.js` (grid, detail with references and log, Reuse/Edit/Download).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                       # no GPU, model or torch
python tests/mock_server.py     # real app + mock engine; UI :3102, API/MCP :8102
```

- `tests/conftest.py`: `tmp_env` (temp config, output, fake model dirs; the fake Turbo
  folder has a `model_index.json` with the real 8 sigmas), `write_config`, `make_client`,
  `client`, `mcp` helper, fixture loaders.
- `tests/mock_engine.py`: deterministic images; `fixed_steps` per model; `rgba`;
  `fail_on_load`, `fail_on_render`; `hold`/`started`/`release` to pause a job (409 tests).
- `tests/test_engine_wiring.py` (needs CPU torch + diffusers, else skipped): drives
  `QwenEngine` with a stand-in pipeline, and runs `place_vae` through the real
  `_encode_vae_image` with a tiny random `AutoencoderKLQwenImage21`.
- No automated test uses the real weights: after engine changes run `docs/SMOKE_TEST.md`.

## Debugging playbook

1. **Which job?** `output/logs/<id>.log` (Gallery → image → log; `GET /v1/gallery/<id>`).
2. **Where did it stop?** `GET /status` → `phase`.
3. **OOM?** `OUT OF MEMORY during <stage>` with VRAM/RAM; for edits fewer images or lower
   `edit.resolution` first. Exit 137 = host RAM. See PITFALLS.
4. **Edit fails instantly in the VAE** (`BFloat16` vs `float`, or cuda vs cpu tensors) →
   `place_vae` not applied to encode.
5. **"Pipeline ran X steps but Y were planned"** → saved schedule not applied (old
   diffusers in the Turbo image: rebuild with `--no-cache`, PITFALLS).
6. **Edit ignores an image** → check `sources:` in the log (order and count), and refer
   to images as "Picture N" in the prompt.
7. After a crash: `docker compose down` then `up -d --build`.

## Constraints to remember

- `server/` is baked into both images: code edits need `docker compose up -d --build
  qwen-image-api` and/or `qwen-image-turbo-api`. Config → restart. `web/` → refresh
  (`chmod -R a+rX web` if nginx 403s).
- One job at a time per container; one model loaded at a time on the GPU overall.
- Keep: torchao int8 via `quant_mapping`, `device_map="balanced"` budgets, VAE slicing on,
  tiling off, CFG 1.0. Never `expandable_segments:True`, never bitsandbytes, never
  pre-quantized community checkpoints (all in PITFALLS).
- MCP SDK is 2.x (`MCPServer`); pinned `mcp>=2.3,<3`.
