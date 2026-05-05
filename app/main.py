"""FastAPI entry point for the Flockjay agentic chat service."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Allow direct invocation (`python app/main.py` or IDE debug launcher) by
# prepending the project root so `app.*` resolves. No-op when invoked as
# `python -m app.main` or via uvicorn.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.chat.router import router as chat_router
from app.runtime.runner import get_runtime, init_runtime
from app.settings import settings

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Tame noisy deps
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    log = logging.getLogger("flockjay.main")
    init_runtime()
    log.info("Flockjay agent service starting on :%s", settings.port)
    try:
        yield
    finally:
        log.info("Shutting down runtime…")
        await get_runtime().aclose()


app = FastAPI(
    title="Flockjay Agent",
    version="0.1.0",
    description="Agentic chat backend on top of the Flockjay MCP server.",
    lifespan=lifespan,
)

app.include_router(chat_router, tags=["chat"])

# Drop any local file into ./samples
# http://localhost:8000/samples/<filename>
SAMPLES_DIR.mkdir(exist_ok=True)
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    """Convenience entry: `poetry run python -m app.main`."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
        workers=1,  # in-memory state requires a single worker
    )


if __name__ == "__main__":
    run()
