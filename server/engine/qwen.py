"""The real engine: Qwen-Image-2.1 / 2.1-Turbo through diffusers' QwenImage21Pipeline.

Loading (unchanged from the original server): bf16, optional torchao int8 weight-only
quantization of the transformer, `device_map="balanced"` with `max_memory` budgets so
the text encoder spills to system RAM, VAE slicing on and tiling off (see PITFALLS).

Editing needs no extra model: Qwen-Image-2.1 is a unified generation + editing model.
`pipe(prompt, image=[...])` conditions on up to 10 reference images, which the Qwen3-VL
text encoder reads as vision context and the VAE encodes into latent tokens.

Progress: diffusers' `callback_on_step_end`, once per denoising step.
"""
from __future__ import annotations

import gc
import threading

import torch
from PIL import Image

from ..monitoring import logger
from ..settings import Settings
from . import JobHooks, RenderJob, RenderResult


def place_vae(pipe, device: str | None, dtype=torch.float32) -> None:
    """Run the VAE in `dtype` (and on `device`, if given), for BOTH directions.

    The pipeline hands the VAE tensors in the transformer's device/dtype (GPU, bf16):
    decode at the end of every job, encode for every edit reference image. So inputs
    are converted to the VAE's own device/dtype, and encode's result is converted back
    to the caller's, because the pipeline concatenates it with the GPU latents.
    """
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.modeling_outputs import AutoencoderKLOutput

    vae = pipe.vae
    if device == "cpu":
        # device_map loading attaches accelerate hooks that move every input back to
        # the GPU; remove them first or the CPU weights receive CUDA tensors (PITFALLS).
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(vae, recurse=True)
        vae.to(device="cpu", dtype=dtype)
    else:
        vae.to(dtype)

    original_decode, original_encode = vae.decode, vae.encode

    def decode(z, *args, **kwargs):
        return original_decode(z.to(device=device or z.device, dtype=dtype), *args, **kwargs)

    def encode(x, *args, return_dict: bool = True, **kwargs):
        out = original_encode(x.to(device=device or x.device, dtype=dtype), *args, return_dict=return_dict, **kwargs)
        dist = out.latent_dist if return_dict else out[0]
        moved = DiagonalGaussianDistribution(dist.parameters.to(device=x.device, dtype=x.dtype),
                                             deterministic=dist.deterministic)
        return AutoencoderKLOutput(latent_dist=moved) if return_dict else (moved,)

    vae.decode, vae.encode = decode, encode


def saved_schedule_length(pipe) -> int | None:
    sigmas = getattr(pipe.config, "sample_sigmas", None)
    return len(sigmas) if sigmas else None


class QwenEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None
        self._loaded_name: str | None = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------ loading
    def loaded_model(self) -> str | None:
        return self._loaded_name

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self, model: str) -> None:
        """Load `model`, swapping out a different loaded one first (thread-safe)."""
        if self._pipe is not None and self._loaded_name == model:
            return
        with self._load_lock:
            if self._pipe is not None and self._loaded_name == model:
                return
            self._unload_locked()
            self._pipe = self._build(model)
            self._loaded_name = model

    def _build(self, name: str):
        import diffusers
        import transformers
        from diffusers import QwenImage21Pipeline

        ms = self.settings.model(name)
        mem = self.settings.memory
        run_on_ram, quantize = bool(mem["run_on_ram"]), bool(mem["quantize_transformer"])

        logger.info(f"Versions: torch {torch.__version__}, diffusers {diffusers.__version__}, "
                    f"transformers {transformers.__version__}")
        mode = "system RAM / CPU only" if run_on_ram else "GPU + CPU offload (balanced)"
        logger.info(f"Loading '{ms.label}' from {ms.path} — mode: {mode}, quantized transformer: {quantize}")

        kwargs = dict(dtype=torch.bfloat16, local_files_only=True)
        if quantize:
            from diffusers import TorchAoConfig
            from diffusers.quantizers import PipelineQuantizationConfig
            from torchao.quantization import Int8WeightOnlyConfig
            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_mapping={"transformer": TorchAoConfig(quant_type=Int8WeightOnlyConfig())})
        if not run_on_ram:
            # device_map handles placement; no separate pipe.to("cuda").
            kwargs["device_map"] = "balanced"
            kwargs["max_memory"] = {0: mem["gpu_max_memory"], "cpu": mem["cpu_max_memory"]}

        pipe = QwenImage21Pipeline.from_pretrained(ms.path, **kwargs)

        pipe.vae.enable_slicing()
        if mem.get("vae_on_cpu"):
            place_vae(pipe, "cpu")
            logger.info("VAE runs on the CPU in float32 (memory.vae_on_cpu: true).")
        elif mem.get("vae_fp32"):
            place_vae(pipe, None)
            logger.info("VAE runs in float32 (memory.vae_fp32: true).")
        # pipe.vae.enable_tiling()  # don't: visible purple seams (PITFALLS)

        fixed = saved_schedule_length(pipe)
        if fixed and fixed != ms.total_steps:
            logger.info(f"This checkpoint has a saved {fixed}-step schedule; it runs {fixed} steps "
                        f"whatever total_steps ({ms.total_steps}) says.")
        logger.info("Pipeline loaded and ready.")
        device_map = getattr(pipe, "hf_device_map", None)
        logger.info(f"Device map: {device_map if device_map else 'all on CPU (RAM)'}")
        return pipe

    def _unload_locked(self) -> None:
        if self._pipe is None:
            return
        logger.info(f"Unloading '{self._loaded_name}' to free memory...")
        try:
            self._pipe.remove_all_hooks()
        except Exception:  # noqa: BLE001
            pass
        self._pipe = None
        self._loaded_name = None
        self.free_memory()

    def free_memory(self) -> None:
        gc.collect()
        if not self.settings.run_on_ram and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ running
    def _call(self, job: RenderJob, hooks: JobHooks, **extra) -> RenderResult:
        hooks.set_phase("loading")
        self.load(job.model)
        pipe = self._pipe
        planned = saved_schedule_length(pipe) or job.steps
        counter = {"n": 0}

        def on_step_end(_pipe, step, timestep, callback_kwargs):
            counter["n"] += 1
            hooks.on_step()
            return callback_kwargs

        kwargs = dict(prompt=job.prompt, width=job.width, height=job.height,
                      num_inference_steps=job.steps,
                      generator=torch.Generator("cpu").manual_seed(job.seed),
                      callback_on_step_end=on_step_end, **extra)
        # True CFG only runs when scale > 1 AND a negative prompt is given
        # (a single space is the standard "empty" negative prompt).
        if job.true_cfg_scale > 1:
            kwargs["true_cfg_scale"] = job.true_cfg_scale
            kwargs["negative_prompt"] = job.negative_prompt.strip() or " "

        hooks.set_phase("denoising", total_steps=planned)
        with torch.inference_mode():
            image = pipe(**kwargs).images[0]
        return RenderResult(image=image, steps_ran=counter["n"], steps_planned=planned,
                            details={"cfg": "off" if job.true_cfg_scale <= 1 else job.true_cfg_scale})

    def text_to_image(self, job: RenderJob, hooks: JobHooks) -> RenderResult:
        return self._call(job, hooks)

    def edit(self, job: RenderJob, images: list[Image.Image], resolution: int,
             hooks: JobHooks) -> RenderResult:
        result = self._call(job, hooks, image=list(images), output_resolution=resolution)
        result.details["reference_images"] = len(images)
        return result
