import base64
import gc
from io import BytesIO
import os
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config import config

from schemas import ImageRequest
from pipeline import get_pipe, is_loaded
from monitoring import (
    logger,
    generation_state,
    progress_callback,
    start_generation,
    get_gpu_stats,
    get_ram_stats,
    get_recent_logs,
    get_active_warnings,
    is_oom_error,
    report_oom,
    clear_oom,
)

if not config["startup"]["lazy_load"]:
    get_pipe()  # eager load at startup, same behavior as before

OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_STEPS = config["generation"]["total_steps"]
MAX_PIXELS = config["generation"]["max_pixels"]
RUN_ON_RAM = config["memory"]["run_on_ram"]
TRUE_CFG_SCALE = float(config["generation"]["true_cfg_scale"])
DEFAULT_NEGATIVE = config["generation"]["negative_prompt"] or ""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def free_memory():
    """Release cached memory retained by Python and (when on GPU) ROCm."""
    gc.collect()
    if not RUN_ON_RAM and torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.get("/status")
def get_status():
    gpu = get_gpu_stats()
    ram = get_ram_stats()
    return {
        "generating": generation_state["active"],
        "model_loaded": is_loaded(),
        "run_on_ram": RUN_ON_RAM,
        "step": generation_state["step"],
        "total_steps": generation_state["total_steps"],
        "gpu": gpu,
        "ram": ram,
        "warnings": get_active_warnings(gpu, ram),
        "logs": get_recent_logs(),
    }


@app.get("/config")
def get_public_config():
    return {
        "available_sizes": config["ui"]["available_sizes"],
        "max_pixels": MAX_PIXELS,
        "negative_prompt": DEFAULT_NEGATIVE,
        "true_cfg_scale": TRUE_CFG_SCALE,
    }


@app.post("/v1/images/generations")
def generate_image(req: ImageRequest):
    try:
        w, h = map(int, req.size.split("x"))
    except Exception:
        w, h = 1024, 1024

    if w * h > MAX_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Resolution {w}x{h} ({w*h} px) exceeds max_pixels ({MAX_PIXELS}) in config.yaml.",
        )

    started = time.time()

    try:
        pipe = get_pipe()  # loads here on first call if lazy_load is true
    except Exception as e:
        free_memory()
        if is_oom_error(e):
            raise HTTPException(status_code=503, detail="Out of memory while loading the model. See logs.")
        raise

    # True CFG only runs when scale > 1 AND a negative prompt is supplied
    # (a single space is the standard "empty" negative prompt).
    cfg_kwargs = {}
    if TRUE_CFG_SCALE > 1:
        negative = req.negative_prompt if req.negative_prompt is not None else DEFAULT_NEGATIVE
        cfg_kwargs = {
            "true_cfg_scale": TRUE_CFG_SCALE,
            "negative_prompt": negative.strip() or " ",
        }
        cfg_info = f"cfg={TRUE_CFG_SCALE} negative='{negative.strip()[:60]}'"
    else:
        cfg_info = "cfg=off"

    prompt_preview = " ".join(req.prompt.split())[:60]
    logger.info(f"Generating: '{prompt_preview}' size={w}x{h} {cfg_info}")
    start_generation(TOTAL_STEPS)

    try:
        with torch.inference_mode():
            image = pipe(
                prompt=req.prompt,
                width=w,
                height=h,
                num_inference_steps=TOTAL_STEPS,
                callback_on_step_end=progress_callback,
                **cfg_kwargs,
            ).images[0]

        gen_secs = time.time() - started
        logger.info(
            f"Generation complete in {gen_secs:.1f}s "
            f"(peak VRAM {generation_state['peak_vram_gb']} GB, peak RAM {generation_state['peak_ram_gb']} GB)"
        )

        filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)
        logger.info(f"Saved output image to {filepath}")
        clear_oom()
    except Exception as e:
        if is_oom_error(e):
            report_oom(e, stage=f"generation ({w}x{h})")
            free_memory()
            raise HTTPException(
                status_code=503,
                detail=f"Out of memory generating {w}x{h}. Try a smaller size (see server logs for details).",
            )
        logger.exception("Generation failed")
        raise
    finally:
        generation_state["active"] = False

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    free_memory()

    return {
        "created": 123456789,
        "data": [{"b64_json": img_b64}],
        "generation_seconds": round(gen_secs, 1),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
