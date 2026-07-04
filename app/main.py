"""DoubleDown AI — FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.scheduler import start_scheduler, shutdown_scheduler
from app.routers import odds as odds_router
from app.routers import predictions as predictions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown."""
    # Startup
    print(f"[startup] Initializing {settings.app_name}...")
    init_db()
    print("[startup] Database tables created.")
    start_scheduler()
    print("[startup] Scheduler started.")
    yield
    # Shutdown
    shutdown_scheduler()
    print("[shutdown] Scheduler stopped.")


app = FastAPI(
    title="DoubleDown AI API",
    description="Value betting engine — compare ML model probabilities against sportsbook implied odds.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Lovable frontend and any other origins
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://9aaee183387ed73a09fb08b9ca8ed51a.ctonew.app",
        "https://*.lovable.app",
        "https://getdoubledown.com",
        "https://*.getdoubledown.com",
    ],
    allow_origin_regex=r"https://.*\.(ctonew\.app|getdoubledown\.com|lovable\.app|lovable\.dev)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(odds_router.router, prefix="/api/v1", tags=["Odds"])
app.include_router(predictions_router.router, prefix="/api/v1", tags=["Predictions"])


@app.get("/", tags=["Root"])
async def root():
    """API root — redirects to docs."""
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }