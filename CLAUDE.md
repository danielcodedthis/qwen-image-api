# Project context for AI assistants

This file is read automatically by Claude Code when working in this repo.
Read `docs/PITFALLS.md` before debugging anything that looks like a repeat
of a previously-solved issue — most obvious-looking bugs here have already
been hit once.

## What this is

A FastAPI service (`qwen-image-api`) wrapping the `diffusers` `QwenImage21Pipeline`
(Qwen-Image-2.1), running on AMD ROCm against a Radeon AI PRO R9700 (gfx1201,
RDNA4, 32GB VRAM), on Arch Linux, 32GB system RAM. A minimal static web UI
(`qwen-image-ui`, nginx) sits in front of it for manual prompting.

## Hard constraints of this environment

- **VRAM (32GB) and system RAM (32GB) are both genuinely tight** relative to
  the ~30GB unquantized model. Almost every past bug in this repo traces back
  to one of these two ceilings. Don't assume "just add more headroom
  somewhere" is free — it usually isn't.
- **gfx1201 (RDNA4) is a new ROCm target.** Some CUDA-first tooling
  (`bitsandbytes`, certain PyTorch allocator flags) either doesn't support it
  or behaves differently than on NVIDIA. Don't assume a CUDA-era fix applies
  as-is — check for a ROCm-specific angle first.
- **Editing `server/*.py` requires a rebuild, not a restart.** The Dockerfile
  `COPY`s these files in at build time; they are not bind-mounted. Running
  `docker compose restart` after an edit silently keeps serving the old code.
  Always: `docker compose up -d --build qwen-image-api`.
- **`docker compose down` tears down every service in the compose file;
  `docker compose up -d --build <service>` only brings up the named
  service(s).** If you `down` then `up --build` one service, sibling services
  (e.g. `qwen-image-ui`) stay down until explicitly brought back up. Prefer
  `docker compose up -d --build` (no service name) after a full `down` unless
  you deliberately want only one service back.

## Current known-good configuration

- Transformer: int8 weight-only quantized via `torchao`
  (`Int8WeightOnlyConfig()` through diffusers' `PipelineQuantizationConfig`
  with an explicit `quant_mapping`, not the deprecated string-based
  `quant_kwargs` API).
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

## When something OOMs or crashes

1. Check `exit code 137` vs a Python `torch.OutOfMemoryError` traceback —
   these are different failure modes (host RAM/OOM-killer vs VRAM exhaustion
   inside PyTorch) and point to different fixes. See PITFALLS for both.
2. Always `docker compose down` (not just `restart`) before retrying after a
   crash — PyTorch's caching allocator can leave stale reserved VRAM behind
   that a plain restart doesn't clear.
3. Check `/status` (the monitoring endpoint) or `rocm-smi --showmeminfo vram`
   for actual live numbers before guessing at a fix.

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
