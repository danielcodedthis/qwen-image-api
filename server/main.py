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
    get_gpu_stats,
    get_ram_stats,
    get_recent_logs,
)

if not config["startup"]["lazy_load"]:
    get_pipe()  # eager load at startup, same behavior as before

OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_STEPS = config["generation"]["total_steps"]
MAX_PIXELS = config["generation"]["max_pixels"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
def get_status():
    return {
        "generating": generation_state["active"],
        "model_loaded": is_loaded(),
        "step": generation_state["step"],
        "total_steps": generation_state["total_steps"],
        "gpu": get_gpu_stats(),
        "ram": get_ram_stats(),
        "logs": get_recent_logs(),
    }


@app.get("/config")
def get_public_config():
    return {
        "available_sizes": config["ui"]["available_sizes"],
        "max_pixels": MAX_PIXELS,
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

    pipe = get_pipe()  # loads here on first call if lazy_load is true
    logger.info(f"Generating: '{req.prompt[:60]}' size={w}x{h}")
    generation_state.update({"active": True, "step": 0, "total_steps": TOTAL_STEPS})

    try:
        with torch.inference_mode():
            image = pipe(
                prompt=req.prompt,
                width=w,
                height=h,
                num_inference_steps=TOTAL_STEPS,
                callback_on_step_end=progress_callback,
            ).images[0]
        logger.info("Generation complete")

        filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)
        logger.info(f"Saved output image to {filepath}")
    finally:
        generation_state["active"] = False

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Flush cached memory retained by ROCm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "created": 123456789,
        "data": [{"b64_json": img_b64}],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
