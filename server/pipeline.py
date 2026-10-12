import threading
import torch
from diffusers import QwenImage21Pipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import TorchAoConfig as DiffusersTorchAoConfig
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

from monitoring import logger, is_oom_error, report_oom
from config import config

_pipe = None
_load_lock = threading.Lock()


def _build_pipe():
    run_on_ram = config["memory"]["run_on_ram"]
    quantize = config["memory"]["quantize_transformer"]

    mode = "system RAM / CPU only" if run_on_ram else "GPU + CPU offload (balanced)"
    logger.info(f"Loading local Qwen-Image-2.1 — mode: {mode}, quantized transformer: {quantize}")

    kwargs = dict(dtype=torch.bfloat16, local_files_only=True)

    if quantize:
        quant_mapping = {
            "transformer": DiffusersTorchAoConfig(quant_type=Int8WeightOnlyConfig()),
            # "text_encoder": TransformersTorchAoConfig(quant_type=Int8WeightOnlyConfig()),
        }
        kwargs["quantization_config"] = PipelineQuantizationConfig(quant_mapping=quant_mapping)

    if not run_on_ram:
        # device_map handles placement; no separate pipe.to("cuda").
        kwargs["device_map"] = "balanced"
        kwargs["max_memory"] = {
            0: config["memory"]["gpu_max_memory"],
            "cpu": config["memory"]["cpu_max_memory"],
        }
    # else: no device_map -> everything loads on CPU and stays there.

    pipe = QwenImage21Pipeline.from_pretrained("/app/models/Qwen-Image-2.1", **kwargs)

    pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling() # Don't do it, reduces quality. Purple lines.
    logger.info("Pipeline loaded and ready.")
    device_map = getattr(pipe, "hf_device_map", None)
    logger.info(f"Device map: {device_map if device_map else 'all on CPU (RAM)'}")
    return pipe


def get_pipe():
    """Thread-safe lazy load — builds the pipeline on first call, reuses it after."""
    global _pipe
    if _pipe is None:
        with _load_lock:
            if _pipe is None:  # re-check inside the lock, in case two requests raced here
                try:
                    _pipe = _build_pipe()
                except Exception as e:
                    if is_oom_error(e):
                        report_oom(e, stage="model load")
                    else:
                        logger.exception("Model load failed")
                    raise
    return _pipe


def is_loaded():
    return _pipe is not None
