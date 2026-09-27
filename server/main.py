import base64
import gc
from io import BytesIO

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import ImageRequest
from pipeline import pipe
from monitoring import (
    logger,
    generation_state,
    progress_callback,
    get_gpu_stats,
    get_ram_stats,
    get_recent_logs,
)

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
        "step": generation_state["step"],
        "total_steps": generation_state["total_steps"],
        "gpu": get_gpu_stats(),
        "ram": get_ram_stats(),
        "logs": get_recent_logs(),
    }


@app.post("/v1/images/generations")
def generate_image(req: ImageRequest):
    try:
        w, h = map(int, req.size.split("x"))
    except Exception:
        w, h = 1024, 1024

    logger.info(f"Generating: '{req.prompt[:60]}' size={w}x{h}")
    generation_state.update({"active": True, "step": 0, "total_steps": 40})

    try:
        with torch.inference_mode():
            image = pipe(
                prompt=req.prompt,
                width=w,
                height=h,
                callback_on_step_end=progress_callback,
            ).images[0]
        logger.info("Generation complete")
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
