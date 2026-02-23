"""
Injury Detection & Injury Risk Prediction System
=================================================
FastAPI entry point — clean rewrite with:
  - lifespan context manager (no deprecated startup/shutdown)
  - async model auto-download on first startup
  - serves built React frontend as static SPA
  - robust CORS
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import CORS_ORIGINS, MODEL_URLS, MODELS_DIR

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Model Auto-Download ─────────────────────────────────────────────────

async def _ensure_models():
    """Download any missing model .task files from Google storage."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for filename, url in MODEL_URLS.items():
            dest = os.path.join(MODELS_DIR, filename)
            if os.path.isfile(dest):
                logger.info(f"Model already present: {filename}")
                continue
            logger.info(f"Downloading model: {filename} ...")
            try:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                logger.info(f"Downloaded: {filename} ({os.path.getsize(dest) // 1024} KB)")
            except Exception as exc:
                logger.warning(f"Could not download {filename}: {exc} — analysis will run without it")

# ─── Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  InjuryGuard AI — Starting")
    logger.info("=" * 60)

    # Download models if missing
    await _ensure_models()

    # Pre-warm prediction engine (trains in background thread via thread pool)
    try:
        from routes.analysis import warmup
        await warmup()
    except Exception as exc:
        logger.warning(f"Pre-warm failed (non-fatal): {exc}")

    logger.info("System ready — accepting connections")
    yield
    logger.info("InjuryGuard AI — Shutting down")


# ─── App ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="InjuryGuard AI",
    description="Real-time injury detection & risk prediction for sports.",
    version="2.0.0",
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API Routes ──────────────────────────────────────────────────────────
from routes.analysis import router as analysis_router   # noqa: E402 (after app created)
app.include_router(analysis_router)

# ─── Serve Built React Frontend ──────────────────────────────────────────
_FRONTEND_DIST = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)

if os.path.isdir(_FRONTEND_DIST):
    logger.info(f"Serving frontend from: {_FRONTEND_DIST}")

    # Mount /assets directory (Vite puts JS/CSS here)
    _assets_dir = os.path.join(_FRONTEND_DIST, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    # SPA catch-all: serve index.html for every non-API, non-WS path
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Let API and WS routes through (they are registered before this)
        # For everything else, try the file, then fall back to index.html
        candidate = os.path.join(_FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))
else:
    logger.warning(
        f"Frontend dist not found at {_FRONTEND_DIST}. "
        "Run 'cd frontend && npm run build' first."
    )

    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "InjuryGuard AI backend is running. Frontend not built yet."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
