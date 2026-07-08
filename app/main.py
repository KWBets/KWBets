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

    # Auto-seed if database is empty (first Railway deploy, fresh DB)
    _auto_bootstrap()

    start_scheduler()
    print("[startup] Scheduler started.")
    yield
    # Shutdown
    shutdown_scheduler()
    print("[shutdown] Scheduler stopped.")


def _auto_bootstrap():
    """Seed historical data + train model + generate value bets on first boot.
    
    On Railway deploy the database starts empty. This auto-detects that
    and bootstraps the full pipeline so the API returns picks immediately.

    CRITICAL: Never runs if ANY real (non-seed) raw odds exist. Once real
    odds data has been fetched from The Odds API, seeding is permanently skipped.
    """
    from app.database import SessionLocal
    from app.models import RawOdds
    from sqlalchemy import text

    db = SessionLocal()
    try:
        # Check if raw odds exists from a live API fetch (non-seed rows)
        total_raw = db.query(RawOdds).count()
        seed_raw = db.execute(text("SELECT COUNT(*) FROM raw_odds WHERE id LIKE 'seed_%'")).scalar()
        real_raw = total_raw - seed_raw

        if real_raw > 0:
            print(f"[auto_bootstrap] Real odds data exists ({real_raw} rows). Skipping seed permanently.")
            return

        # Also skip if ProcessedFeatures already populated (safe guard)
        from app.models import ProcessedFeatures
        feat_count = db.query(ProcessedFeatures).count()
        if feat_count > 0:
            print(f"[auto_bootstrap] Database already populated ({feat_count} features). Skipping seed.")
            return

        print("[auto_bootstrap] Empty database detected. Running bootstrap pipeline...")

        # 1. Seed historical data (creates ProcessedFeatures + ValueBets + PickOutcomes)
        import seed_historical
        seed_historical.seed_inseason_games(db, num_games=350)
        seed_historical.seed_futures_markets(db, num_entries=250)

        # 2. Train XGBoost model on seeded outcomes
        from app.train import run_training_pipeline
        version = run_training_pipeline(db)
        if version:
            print(f"[auto_bootstrap] Trained model: {version}")

        # 4. Generate value bets
        from app.ev import run_ev_pipeline
        bet_count = run_ev_pipeline(db)
        print(f"[auto_bootstrap] Bootstrap complete — {bet_count} value bets generated.")

    except Exception as e:
        print(f"[auto_bootstrap] Error during bootstrap: {e}", exc_info=True)
    finally:
        db.close()


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