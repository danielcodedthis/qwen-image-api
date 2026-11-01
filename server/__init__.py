"""Qwen Image API: FastAPI + MCP server around diffusers' QwenImage21Pipeline.

The same package runs in both containers (qwen-image-api for Qwen-Image-2.1 and
qwen-image-turbo-api for 2.1-Turbo); only the mounted config differs.

Module map (see skills/qwen-image-api-internals/SKILL.md for the full walkthrough):

    settings.py     config.yaml + defaults -> Settings
    app.py          create_app(): wires everything, routes, MCP, lifespan
    service.py      validation, one-job-at-a-time, logs/PNGs for each job
    engine/         the model (only place that imports torch/diffusers)
    imaging.py      Pillow helpers (uploads, edit sizing, PNG, thumbnails)
    storage.py      output/ files, per-job logs, gallery listing
    monitoring.py   logging, progress, GPU/RAM stats, OOM detection
    routes/         HTTP endpoints
    mcp_server.py   MCP tools at /mcp
"""
