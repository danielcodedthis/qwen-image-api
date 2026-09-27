# Pitfalls & fixes

A chronological record of real issues hit while building this out. Written so
neither a future AI assistant nor future-you has to re-diagnose these from
scratch. Each entry: symptom → root cause → fix.

## Sequential CPU offload + tight system RAM = OOM (exit 137)

**Symptom:** Container silently killed (`exit code 137`), no Python traceback,
right after `Uvicorn running on...` or partway through generation.

**Root cause:** `enable_sequential_cpu_offload()` keeps the *entire* model
resident in system RAM, streaming layers to GPU one at a time. With a ~30GB
model and only 32GB system RAM, this leaves almost nothing for the OS and
other processes — the kernel OOM-killer terminates the process. `exit 137` =
SIGKILL, almost always the OOM-killer, not an application-level error.

**Fix:** Don't rely on CPU offload as the primary VRAM-saving strategy on a
system this RAM-constrained. Quantize the model instead (see below) so it
fits on GPU directly, or use `device_map` with an explicit `max_memory` split
so only what doesn't fit spills to RAM — not the whole model.

## `pipe.enable_vae_slicing()` / `enable_vae_tiling()` → AttributeError

**Symptom:** `AttributeError: 'QwenImage21Pipeline' object has no attribute
'enable_vae_slicing'`.

**Root cause:** Newer `diffusers` deprecated these pipeline-level convenience
wrappers in favor of calling the methods directly on the VAE component.
`QwenImage21Pipeline`, being newly added, never had the old wrappers at all.

**Fix:** `pipe.vae.enable_slicing()` / `pipe.vae.enable_tiling()` — note the
added `.vae.`.

## `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` breaks on ROCm

**Symptom:** Repeated allocator warnings —
`expandable_segments: memory mapping failed with OOM ... (free: 12.9GB, ...)`
— reporting failure to map a tiny (~20MB) chunk *despite* many GB of VRAM
genuinely being free. Looks like an OOM but the free-memory numbers don't
support that conclusion.

**Root cause:** `expandable_segments` is a CUDA-driver-specific virtual memory
trick. It doesn't reliably work the same way through ROCm/HIP, and fails in
a way that presents as a phantom OOM rather than a clean error.

**Fix:** Removed. Use
`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512`
instead, or omit the env var entirely.

## Pre-quantized community checkpoint loads "successfully" but is actually broken

**Symptom:** Loading a community INT8 checkpoint (`Rin247/Qwen-Image-2.1-INT8`)
logs long lists of `weight_scale` tensors as "not used when initializing" —
then still OOMs identically to the unquantized model.

**Root cause:** That checkpoint's quantization format isn't understood by
stock `diffusers.from_pretrained` — the descaling (`weight_scale`) tensors get
silently discarded, and the raw INT8 values get loaded as if they were normal
weights. Worse, passing `dtype=torch.bfloat16` upcasts every tensor from 1
byte back to 2 bytes in memory during load, completely erasing the intended
memory savings. The load "succeeds" with no error, but is silently wrong on
two counts at once.

**Fix:** Abandoned this checkpoint. Switched to on-the-fly quantization of
the *original* bf16 model via `diffusers`' `PipelineQuantizationConfig` +
`torchao`, which is the officially supported path and actually reduces
in-memory footprint.

## `PipelineQuantizationConfig(quant_backend=..., quant_kwargs=...)` → ValueError

**Symptom:** `ValueError: The signatures of the __init__ methods of the
quantization config classes in diffusers and transformers don't match. ...
provide a quant_mapping instead.`

**Root cause:** The `transformer` (a `diffusers` component) and `text_encoder`
(a `transformers` component) have differently-shaped `TorchAoConfig`
constructors, so one shared `quant_kwargs` dict can't apply to both.

**Fix:** Use `quant_mapping` — a dict naming each component with its own
config object built from the matching package's `TorchAoConfig`:

```python
from diffusers import TorchAoConfig as DiffusersTorchAoConfig
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

quant_mapping = {
    "transformer": DiffusersTorchAoConfig(quant_type=Int8WeightOnlyConfig()),
    "text_encoder": TransformersTorchAoConfig(quant_type=Int8WeightOnlyConfig()),
}
```

Also note: `quant_type` must be an actual `AOBaseConfig` instance
(`Int8WeightOnlyConfig()`), not the older string shorthand
(`"int8_weight_only"`) — that string API was replaced.

## VRAM OOM specifically during VAE decode, not during denoising

**Symptom:** Denoising completes (`100%|... 40/40`), then
`torch.OutOfMemoryError` inside `vae.decode()` / `conv2d`, with very little
free VRAM reported at that point despite plenty being free earlier in the run.

**Root cause:** The unquantized `text_encoder` (~16GB) stays resident in VRAM
for the entire request even though it only does work once, at the very start.
By the time the VAE needs a burst of memory for decode activations, that
headroom is already spent.

**Fix:** Use `device_map="balanced"` with an explicit `max_memory` budget on
the GPU (e.g. `{0: "22GiB", "cpu": "40GiB"}`) at pipeline load time — this
lets `accelerate` automatically place the text encoder on CPU RAM permanently
(not shuffled back and forth like `enable_model_cpu_offload()` does), freeing
VRAM headroom for decode. Cost is bounded since text encoding happens once
per request, not once per step.

## `enable_vae_tiling()` causes visible purple/seam artifacts

**Symptom:** Generated images show faint purple-tinted lines or blotches,
especially in flat/textured regions (e.g. knit fabric).

**Root cause:** VAE tiling decodes the image in patches and stitches them
back together to save VRAM; the seams between tiles can introduce visible
color-channel discontinuities.

**Fix:** Once the `device_map` VRAM headroom fix above was in place, tiling
was no longer necessary — disabling `enable_vae_tiling()` (keeping
`enable_slicing()`) removed the artifacts entirely and meaningfully improved
image quality. Only re-enable tiling if a future resolution/config change
causes a genuine VAE-decode OOM that can't be solved by freeing VRAM another
way.

## Driver/GPU resets on first run after a `device_map` config change

**Symptom:** Container failed to start cleanly, GPU appeared to
reset/recover, twice, before a subsequent attempt started and ran flawlessly.

**Likely cause:** Stale allocator/VRAM state left over from a previous
crash, not fully cleared by a plain `restart`. Not confirmed as a recurring
issue — worth re-investigating via `sudo dmesg -T | grep -i "amdgpu.*reset"`
if it happens again outside of a fresh config change.

**Mitigation:** Always `docker compose down` (full teardown) rather than
`restart` before testing a new resource-budget config, to guarantee a clean
GPU context.

## `docker compose down` + targeted `up --build <service>` leaves siblings down

**Symptom:** After `docker compose down` then
`docker compose up -d --build qwen-image-api`, the `qwen-image-ui` container
did not come back.

**Root cause:** Naming a service in `up` scopes the command to only that
service. `down` with no name tears down everything; `up` with a name brings
back only that one thing.

**Fix:** After a full `down`, bring everything back with no service name:
`docker compose up -d --build` (rebuilds only what changed, starts every
service defined in the compose file).

## `git commit` fails with "Author identity unknown"

**Symptom:** `fatal: unable to auto-detect email address ...` on first
commit on a fresh Arch install.

**Root cause:** `gh auth login` authenticates you to GitHub's API; it's
unrelated to git's own local commit-author identity, which is unset by
default on a fresh install.

**Fix:**
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
Use an email registered to your GitHub account (or its `noreply` alias from
`github.com/settings/emails`) if you want commits attributed to your profile.
