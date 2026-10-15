# Project context for AI assistants

This file is read automatically by Claude Code when working in this repo.
Read `docs/PITFALLS.md` before debugging anything that looks like a repeat
of a previously-solved issue — most obvious-looking bugs here have already
been hit once.

## What this is

Two FastAPI services wrapping the `diffusers` `QwenImage21Pipeline`, running on
AMD ROCm against a Radeon AI PRO R9700 (gfx1201, RDNA4, 32GB VRAM), on Arch
Linux, **96GB system RAM** (upgraded from 32GB):

- `qwen-image-api` (`:8000`, UI `qwen-image-ui` on `:3000`): Qwen-Image-2.1 (base).
- `qwen-image-turbo-api` (`:8001`, UI `qwen-image-turbo-ui` on `:3001`):
  Qwen-Image-2.1-Turbo only.

Both APIs run the same code in `server/` but are separate containers with their
own image (`Dockerfile` / `Dockerfile.turbo`), config (`config.yaml` /
`config.turbo.yaml`), output folder (`output/` / `output-turbo/`) and port.
Both UIs are nginx containers serving the same `web/index.html`; the page picks
its API from the port it is opened on (`:3001` → `:8001`, else `:8000`, or
`?api=PORT`).

## Hard constraints of this environment

- **VRAM (32GB) is genuinely tight** relative to the ~30GB unquantized model
  (7B transformer + 8B Qwen3-VL text encoder + VAE). Almost every past VRAM bug
  traces back to this ceiling. Don't assume "just add more headroom" is free.
- **System RAM is 96GB now** and no longer the binding constraint. Several old
  pitfalls (sequential CPU offload, exit 137) were written when it was 32GB;
  they still explain *why* some choices were made, but RAM is no longer what
  limits us. The text encoder lives in RAM by design.
- **gfx1201 (RDNA4) is a new ROCm target.** Some CUDA-first tooling
  (`bitsandbytes`, certain PyTorch allocator flags) either doesn't support it
  or behaves differently than on NVIDIA. Don't assume a CUDA-era fix applies
  as-is — check for a ROCm-specific angle first.
- **One GPU, two containers.** A loaded model stays in VRAM until its container
  stops (there is no idle unload). Don't have both models loaded at once;
  `docker compose stop` the other API before using one.
- **Editing `server/*.py` requires a rebuild, not a restart.** The Dockerfiles
  `COPY` these files in at build time; they are not bind-mounted. Running
  `docker compose restart` after an edit silently keeps serving the old code.
  Rebuild with `docker compose up -d --build qwen-image-api` (or
  `qwen-image-turbo-api`). `config.yaml` / `config.turbo.yaml` and `web/` *are*
  mounted: a config edit needs only a restart, a UI edit needs only a refresh.
- **`docker compose down` tears down every service in the compose file;
  `docker compose up -d --build <service>` only brings up the named
  service(s).** If you `down` then `up --build` one service, sibling services
  (the UIs) stay down until explicitly brought back up. Prefer
  `docker compose up -d --build` (no service name) after a full `down`.
- **Docker creates missing bind-mount folders as root.** `output-turbo/` must be
  created by the user (`mkdir -p`) before first start, or the container (which
  runs as `UID:GID`) can't write into it.

## Current known-good configuration

**Base (`config.yaml`):**
- Transformer: int8 weight-only quantized via `torchao`
  (`Int8WeightOnlyConfig()` through diffusers' `PipelineQuantizationConfig`
  with an explicit `quant_mapping`, not the deprecated string-based
  `quant_kwargs` API). Toggle: `memory.quantize_transformer`.
- Guidance off (`true_cfg_scale: 1.0`). Qwen-Image-2.1 is designed to be sampled
  without guidance; see PITFALLS for what high values did.

**Turbo (`config.turbo.yaml`):**
- `total_steps: 8`, `true_cfg_scale: 1.0`, `quantize_transformer: false` (plain
  bf16; the 7B transformer fits).
- Built from `Dockerfile.turbo`: diffusers from git main (`ARG DIFFUSERS_REF`),
  `transformers>=5.17.0`. Last known-good versions: torch 2.13.0+rocm7.14.0,
  diffusers 0.41.0.dev0, transformers 5.20.0.
- Verify after any rebuild: the log must show the `Versions:` line and, after a
  generation, `Ran 8 denoising steps.` A warning saying the pipeline ran a
  different number of steps means the saved schedule is not being applied as
  expected.

**Both:**
- Text encoder: placed via `device_map="balanced"` + `max_memory` budget —
  spills to CPU RAM automatically when the GPU budget is set below what
  everything would need. This is intentional: the text encoder only runs
  once per request (not once per denoising step), so CPU placement costs
  little in wall-clock time but frees significant VRAM for the VAE decode step.
- VAE: `enable_slicing()` on, `enable_tiling()` off (see PITFALLS — tiling
  caused visible artifacts once headroom made it unnecessary).
- Allocator: plain default, or
  `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512`.
  **Never** `expandable_segments:True` — see PITFALLS, it's CUDA-tuned and
  fails silently/oddly on this ROCm setup.

## Code layout notes

- `server/config.py` holds defaults and `get_model_settings(name)`, which merges
  the `generation` section with a model's own overrides. A model with
  `enabled: false` is dropped, so a container can serve exactly one model.
- `server/pipeline.py` loads and unloads by model name (`get_pipe(name)`).
  Multi-model switching works in-process, but we deliberately run one model per
  container.
- `server/main.py` allows one generation at a time (a second request gets `409`),
  because a model swap unloads the pipeline. Saving the PNG is non-fatal: a
  disk/permission error is logged and the image is still returned.
- `server/monitoring.py` owns logging, GPU/RAM stats, memory-pressure warnings
  and OOM detection (`is_oom_error`, `report_oom`). `/status` exposes the
  results to the UI.
- `web/index.html` is a single static file. Its `localStorage` keys are
  `qwen.prompt`, `qwen.neg`, `qwen.size`, `qwen.model`.

## When something OOMs or crashes

1. Look at the log first: the app now writes an `OUT OF MEMORY` entry with VRAM
   and RAM at failure plus the peak for that run, and `/status` keeps the last
   unresolved OOM in `warnings`.
2. Check `exit code 137` vs a Python `torch.OutOfMemoryError` traceback —
   these are different failure modes (host RAM/OOM-killer vs VRAM exhaustion
   inside PyTorch) and point to different fixes. See PITFALLS for both.
3. Always `docker compose down` (not just `restart`) before retrying after a
   crash — PyTorch's caching allocator can leave stale reserved VRAM behind
   that a plain restart doesn't clear.
4. Check `/status` or `rocm-smi --showmeminfo vram` for actual live numbers
   before guessing at a fix.

## Don't re-attempt these (already tried, didn't work)

- Pre-quantized community checkpoint repos (e.g. `Rin247/Qwen-Image-2.1-INT8`)
  loaded via plain `from_pretrained` — the `weight_scale` tensors get silently
  discarded because stock `diffusers` doesn't recognize that quantization
  format, and forcing `dtype=torch.bfloat16` upcasts everything back to full
  size in memory anyway. Use on-the-fly `torchao` quantization of the
  original model instead (already implemented in `server/pipeline.py`).
- `bitsandbytes` for quantization on this GPU — PyPI wheels are CUDA-only;
  ROCm support needs a from-source build targeting `gfx1201` specifically,
  which is alpha-quality upstream. `torchao` was the more portable choice.
- **True CFG at 3–4 on Qwen-Image-2.1.** That value comes from the original
  Qwen-Image. 2.1 is meant to run without guidance (`true_cfg_scale: 1.0`);
  strong guidance gave bad images. If guidance is ever tried again, start low
  (about 1.5–2) and expect roughly 2x the time per step.
- **Running Turbo at a step count other than 8, or in an image with an old
  cached diffusers layer.** Turbo needs the 8-step schedule saved in its
  checkpoint and a recent diffusers/transformers. Use `Dockerfile.turbo`, and
  rebuild with `--no-cache` if the diffusers layer might be stale.
- **Judging Turbo quality inside the two-model setup.** Quality problems
  appeared there and disappeared in the dedicated container. When testing, use
  `qwen-image-turbo-api` with a single enabled model.
- **`run_on_ram: true` for everyday use.** It works but takes many minutes per
  image. Treat it as a fallback.
