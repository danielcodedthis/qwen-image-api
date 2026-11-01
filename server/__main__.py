"""`python -m server`: run the API with uvicorn (the container's CMD)."""
import os

import uvicorn

from .app import create_app


def main() -> None:
    uvicorn.run(
        create_app(),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        # Generations run in worker threads; one worker process keeps a single model in VRAM.
        workers=1,
    )


if __name__ == "__main__":
    main()
