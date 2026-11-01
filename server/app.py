"""FastAPI application factory.

    create_app()                     -> the real app (config from CONFIG_PATH, QwenEngine)
    create_app(settings, engine=...) -> the same app with another engine (tests use a fake)

The container runs `python -m server` (server/__main__.py). Equivalent:
    uvicorn --factory server.app:create_app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import contextlib
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .engine import Engine
from .monitoring import Monitor, logger, setup_logging
from .routes import gallery, images, system
from .routes.deps import install_error_handlers
from .service import ImageService
from .settings import Settings, load_settings
from .storage import OutputStore

API_VERSION = "2.0.0"


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    setup_logging()
    settings = settings or load_settings()
    if engine is None:
        from .engine.qwen import QwenEngine  # imports torch; only for the real app
        engine = QwenEngine(settings)

    store = OutputStore(settings.output_dir)
    store.ensure_dirs()
    monitor = Monitor(vram_warn_percent=settings.vram_warn_percent,
                      ram_warn_percent=settings.ram_warn_percent)
    service = ImageService(settings, engine, store, monitor)

    mcp = None

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.config_path is None:
            logger.warning("No config.yaml found; running with built-in defaults.")
        if not settings.lazy_load:
            # Load in the background so the API (and the UI's status) is up immediately;
            # requests made while it loads get a 409 like any other busy period.
            threading.Thread(target=service.preload, name="preload", daemon=True).start()
        async with contextlib.AsyncExitStack() as stack:
            if mcp is not None:
                await stack.enter_async_context(mcp.session_manager.run())
            yield

    app = FastAPI(title="Qwen Image API", version=API_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service

    # The UI is served from another port (nginx), so the API must allow cross-origin calls.
    # The server is meant for a local machine / trusted LAN; there is no authentication.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    install_error_handlers(app)

    app.include_router(system.router)
    app.include_router(images.router)
    app.include_router(gallery.router)

    if settings.mcp_enabled:
        from .mcp_server import mount_mcp
        mcp = mount_mcp(app, service)
        app.state.mcp = mcp

    return app

