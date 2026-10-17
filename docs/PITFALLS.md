# Pitfalls & fixes

A chronological record of real issues hit while building this out. Written so
neither a future AI assistant nor future-you has to re-diagnose these from
scratch. Each entry: symptom → root cause → fix.

## Sequential CPU offload + tight system RAM = OOM (exit 137)

*(Written when the machine had 32GB of RAM. It now has 96GB, so RAM is no
longer the binding constraint, but this is still why sequential offload was
dropped.)*

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
(`"int8_weight_only"`) — that string API was replaced. (Currently only the
transformer is quantized; the `text_encoder` entry is commented out in
`server/pipeline.py`.)

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

---

*Everything below was added after the RAM upgrade to 96GB, the web UI redesign
and the introduction of Qwen-Image-2.1-Turbo.*

## True CFG at 4.0 gave bad images on Qwen-Image-2.1

**Symptom:** After adding `true_cfg_scale` and a negative prompt, images looked
"really bad" compared to the earlier results.

**Root cause:** The default of 4.0 was taken from the original Qwen-Image.
Qwen-Image-2.1 is designed to be sampled without guidance (upstream default
`true_cfg_scale` 1.0, recommended 40 steps). Strong guidance is off-distribution
for it. Above 1.0 it also runs the transformer twice per step.

**Fix:** Default is now `true_cfg_scale: 1.0` (off). The negative prompt is only
used when guidance is above 1. If guidance is ever tried, start around 1.5–2.
For hands and feet, more steps are a better lever than guidance.

## Turbo quality degraded (mushy detail, "undead" faces) — fixed with a dedicated container

**Symptom:** Turbo images were good at first (or only at 512×512) and then looked
soft, over-processed, or lifeless at other sizes. The log showed
`Generating [Qwen-Image-2.1 Turbo, 15 steps]`, and later 6 steps was tried.

**Likely causes (not isolated, two things changed together):**
1. Turbo is distilled for **8 steps**, and its sampling schedule is stored
   inside the checkpoint. Other step counts are not what it was trained for.
2. Using the saved schedule needs a recent `diffusers` (source build, newer than
   the last tagged release at the time of the Turbo release notes) and
   `transformers>=5.17`. The original image had an older diffusers cached in
   its Docker layer, so the pipeline may have been using a generic schedule.

**Fix:** Turbo moved to its own container (`qwen-image-turbo-api`, built from
`Dockerfile.turbo`) with a fresh diffusers and 8 steps, bf16, no guidance.
Verified versions: torch 2.13.0+rocm7.14.0, diffusers 0.41.0.dev0,
transformers 5.20.0. After the move the log shows `Ran 8 denoising steps.` and
results were good.

**How to check in future:** the log prints a `Versions:` line at load and, after
each generation, either `Ran N denoising steps.` or a warning that the number of
steps run differs from the number requested.

## The Docker pip layer pins whatever diffusers it first installed

**Symptom:** A plain `docker compose build` reports `Using cache` for the
`pip install` step, so `diffusers` installed from git main never updates, even
if main has moved on.

**Root cause:** Docker caches a layer as long as the `RUN` line text is
unchanged; `git+https://github.com/huggingface/diffusers` is just text.

**Fix:** `docker compose build --no-cache qwen-image-turbo-api` to refresh it
(slow), or pin a commit with `--build-arg DIFFUSERS_REF=<sha>` (Turbo's
Dockerfile exposes that argument) for repeatable builds.

## UI: the "Done in Xs" message disappeared after a second

**Symptom:** The total generation time never showed in the web UI.

**Root cause:** The 1.5s status poll cleared the status line whenever nothing
was generating, overwriting the result message. The message order wasn't the
problem.

**Fix:** The result/timing text has its own element that only the Generate
handler writes. Polling no longer touches it.

## UI: a stale "Model not loaded" warning stayed on screen

**Symptom:** The status line said the model wasn't loaded while it was already
generating.

**Root cause:** The poll set the warning before generation began and then
skipped clearing it while `generating` was true.

**Fix:** The status line was replaced by header chips driven directly from
`/status` (API / model / mode), so there is no text left to go stale. Also, the
server keeps the last run's final step in `/status` after generation ends, so the
UI only trusts `step` while `generating` is true.

## `run_on_ram` is very slow, and the RAM bar looks nearly empty

**Symptom:** With `memory.run_on_ram: true`, a 512×512 image took minutes per
step, and the UI RAM bar showed only a few GB used while `free -h` showed ~68GB
in `buff/cache`.

**Root cause:** Running the whole transformer on the CPU is simply slow. The
model weights are memory-mapped from the safetensors files, so Linux counts
them as reclaimable page cache. The UI's "used" figure is `MemTotal -
MemAvailable`, which leaves them out.

**Fix / guidance:** Treat `run_on_ram` as a fallback, not a normal mode. Look at
`buff/cache` in `free -h` for the real footprint. Plain bf16
(`quantize_transformer: false`) was faster than int8 there.

## `PermissionError` saving the PNG — `output-turbo/` owned by root

**Symptom:** Generation completes (`Ran 8 denoising steps.`), then
`PermissionError: [Errno 13] Permission denied: '/app/output/...png'`, and the
request fails.

**Root cause:** `./output-turbo` didn't exist, so Docker created it as root. The
container runs as `UID:GID`, so it can't write there.

**Fix:** `mkdir -p output-turbo` before first start, or
`sudo chown -R "$USER":"$USER" output-turbo`. The server also no longer discards
the image over this: a failed save is logged as a warning and the image is still
returned to the browser.

## nginx `403 Forbidden` on the UI

**Symptom:** `http://localhost:3001/` returns 403. The nginx log says
`open() "/usr/share/nginx/html/index.html" failed (13: Permission denied)`.

**Root cause:** `web/index.html` had mode `-rw-------` (owner-only), and nginx in
the container runs as a different user.

**Fix:** `chmod 644 web/index.html`. No restart is needed, the folder is
mounted live. (On an SELinux system, add `:z` to the volume line instead.)

## `docker compose up --build` seems to hang after "Using cache"

**Symptom:** All the expensive steps say `Using cache`, but the build sits on the
last steps (`EXPOSE` / `CMD`) for a very long time.

**Likely cause (not confirmed):** The output began with `buildx Docker CLI plugin
not found: falling back to the classic builder`. The classic builder starts a
container and commits a layer per changed step, which is slow on top of the
huge `rocm/pytorch` image. Disk space and the storage driver are other
suspects.

**Fix:** Install the buildx plugin (Arch: `sudo pacman -S docker-buildx`), check
`df -h / /var/lib/docker` and `docker system df`.

## `vae_fp32` makes the final decode run out of VRAM at 1024x1024

**Symptom:** `CUDA out of memory. Tried to allocate 10.12 GiB` at 1024x1024 with about
8.8 GiB free, right after the denoising steps. The same size had worked before
(peak about 22 GB).

**Likely root cause:** `memory.vae_fp32: true` decodes the image in float32, which
doubles the decode activations compared with bf16. A single decode allocation of
that size no longer fits next to the transformer. (Inferred from the size and
timing of the allocation, not from a trace of the decode call.)

**Fix:** Set `vae_fp32: false` again, or set `memory.vae_on_cpu: true` to decode
the image on the CPU from system RAM. That only slows the last step, and uses almost
no VRAM for it. `run_on_ram: true` also avoids the problem, but makes the whole run
very slow.

## `vae_on_cpu`: "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)"

**Symptom:** With `memory.vae_on_cpu: true`, denoising finishes and then the request
fails inside `vae.decode` with `RuntimeError: Input type (torch.cuda.FloatTensor) and
weight type (torch.FloatTensor) should be the same`. The traceback passes through
`accelerate/hooks.py`.

**Root cause:** Loading with `device_map` attaches accelerate hooks to each component.
The VAE's hook moves every input to the GPU before the forward pass. Moving only the VAE
weights to the CPU leaves the hook in place, so CPU weights receive CUDA inputs.

**Fix:** Remove the hooks (`accelerate.hooks.remove_hook_from_module(pipe.vae,
recurse=True)`) before `pipe.vae.to("cpu")`.
