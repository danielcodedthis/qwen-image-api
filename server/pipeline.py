import gc
import threading
import torch
from diffusers import QwenImage21Pipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import TorchAoConfig as DiffusersTorchAoConfig
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

from monitoring import logger, is_oom_error, report_oom
from config import config, get_model_settings

_pipe = None
_loaded_name = None
_load_lock = threading.Lock()


def default_model():
    return config["default_model"]


def _build_pipe(name):
    settings = get_model_settings(name)
    run_on_ram = config["memory"]["run_on_ram"]
    quantize = config["memory"]["quantize_transformer"]

    import diffusers, transformers
    logger.info(
        f"Versions: torch {torch.__version__}, diffusers {diffusers.__version__}, "
        f"transformers {transformers.__version__}"
    )
    mode = "system RAM / CPU only" if run_on_ram else "GPU + CPU offload (balanced)"
    logger.info(
        f"Loading '{settings['label']}' from {settings['path']} — "
        f"mode: {mode}, quantized transformer: {quantize}"
    )

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

    pipe = QwenImage21Pipeline.from_pretrained(settings["path"], **kwargs)

    pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling() # Don't do it, reduces quality. Purple lines.
    logger.info("Pipeline loaded and ready.")
    device_map = getattr(pipe, "hf_device_map", None)
    logger.info(f"Device map: {device_map if device_map else 'all on CPU (RAM)'}")
    return pipe


def _unload_locked():
    """Drop the current pipeline and give its memory back. Caller must hold _load_lock."""
    global _pipe, _loaded_name
    if _pipe is None:
        return
    logger.info(f"Unloading '{_loaded_name}' to free memory...")
    _pipe = None
    _loaded_name = None
    gc.collect()
    if not config["memory"]["run_on_ram"] and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_pipe(name=None):
    """Thread-safe lazy load. Reuses the loaded model, or swaps to `name` if it differs."""
    global _pipe, _loaded_name
    name = name or default_model()
    if _pipe is not None and _loaded_name == name:
        return _pipe
    with _load_lock:
        if _pipe is None or _loaded_name != name:  # re-check inside the lock
            _unload_locked()
            try:
                _pipe = _build_pipe(name)
                _loaded_name = name
            except Exception as e:
                if is_oom_error(e):
                    report_oom(e, stage=f"model load ({name})")
                else:
                    logger.exception(f"Model load failed ({name})")
                raise
    return _pipe


def is_loaded():
    return _pipe is not None


def loaded_model():
    return _loaded_name
