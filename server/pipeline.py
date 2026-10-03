import threading
import torch
from diffusers import QwenImage21Pipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import TorchAoConfig as DiffusersTorchAoConfig
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

from monitoring import logger
from config import config

_pipe = None
_load_lock = threading.Lock()


def _build_pipe():
    logger.info("Loading local Qwen-Image-2.1 with VRAM optimizations...")

    quant_mapping = {
        "transformer": DiffusersTorchAoConfig(quant_type=Int8WeightOnlyConfig()),
        # "text_encoder": TransformersTorchAoConfig(quant_type=Int8WeightOnlyConfig()),
    }
    quant_config = PipelineQuantizationConfig(quant_mapping=quant_mapping)

    max_memory = {
        0: config["memory"]["gpu_max_memory"],
        "cpu": config["memory"]["cpu_max_memory"],
    }

    pipe = QwenImage21Pipeline.from_pretrained(
        "/app/models/Qwen-Image-2.1",
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        local_files_only=True,
        device_map="balanced",
        max_memory=max_memory,
    )

    # no separate pipe.to("cuda") — device_map handles placement
    pipe.vae.enable_slicing()
    logger.info(f"Device map: {pipe.hf_device_map}")
    # pipe.vae.enable_tiling() # Don't do it, reduces quality. Purple lines.
    logger.info("Pipeline loaded and ready.")
    logger.info(f"Device map: {pipe.hf_device_map}")
    return pipe


def get_pipe():
    """Thread-safe lazy load — builds the pipeline on first call, reuses it after."""
    global _pipe
    if _pipe is None:
        with _load_lock:
            if _pipe is None:  # re-check inside the lock, in case two requests raced here
                _pipe = _build_pipe()
    return _pipe


def is_loaded():
    return _pipe is not None