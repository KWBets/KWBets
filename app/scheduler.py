"""APScheduler setup for periodic tasks."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.config import settings

scheduler = AsyncIOScheduler()


def start_scheduler():
    """Initialize and start the APScheduler with all periodic jobs.
    
    IMPORTANT: Do NOT set next_run_time=None on any job — that prevents
    the scheduler from ever firing the job. APScheduler auto-calculates
    the first run time from the trigger when next_run_time is not set.
    """
    from app.odds_ingestion import run_odds_fetch
    from app.grade_outcomes import scheduled_grading
    from app.health_sentinel import run_health_sentinel

    # 6-hourly odds fetch — runs ~6h after startup, then every 6h
    print("[scheduler] Registering odds fetch job (every 6h)...")
    scheduler.add_job(
        run_odds_fetch,
        trigger=IntervalTrigger(hours=settings.odds_fetch_interval_hours),
        id="hourly_odds_fetch",
        name="Fetch odds from The Odds API",
        replace_existing=True,
    )

    # Daily model retraining
    print("[scheduler] Registering model retrain job (every 24h)...")
    scheduler.add_job(
        run_daily_retrain,
        trigger=IntervalTrigger(hours=settings.model_retrain_interval_hours),
        id="daily_model_retrain",
        name="Retrain ML model on accumulated outcomes + regenerate picks",
        replace_existing=True,
    )

    # Grading pipeline every 12 hours
    print("[scheduler] Registering grading pipeline job (every 12h)...")
    scheduler.add_job(
        run_grading,
        trigger=IntervalTrigger(hours=12),
        id="grading_pipeline",
        name="Grade pending bets via scores API + write PickOutcomes",
        replace_existing=True,
    )

    # Health sentinel every 6 hours — runs all 9 checks including quota
    print("[scheduler] Registering health sentinel job (every 6h)...")
    scheduler.add_job(
        run_health_sentinel,
        trigger=IntervalTrigger(hours=6),
        id="health_sentinel",
        name="Run health checks every 6h, email on failure",
        replace_existing=True,
    )

    scheduler.start()
    print(f"[scheduler] APScheduler started: odds fetch(6h), retrain(24h), grading(12h), health sentinel(6h).")


async def run_daily_retrain():
    """Retrain the ML model on accumulated outcomes, then regenerate all picks."""
    from app.database import SessionLocal
    from app.train import run_training_pipeline
    from app.ev import run_ev_pipeline

    print("[scheduler] Daily retrain started: retraining model on new outcomes...")

    db = SessionLocal()
    try:
        version = run_training_pipeline(db)
        if version:
            print(f"[scheduler] Model retrained: {version}")

        # Regenerate picks with the fresh model
        from app.models import ModelRegistry, ValueBet
        bets = run_ev_pipeline(db)
        print(f"[scheduler] Regenerated {bets} value bets with model {version}")
    except Exception as e:
        print(f"[scheduler] Retrain error: {e}", exc_info=True)
    finally:
        db.close()


async def trigger_odds_fetch_now():
    """Manually trigger an odds fetch + full pipeline (called from API endpoint)."""
    from app.odds_ingestion import run_odds_fetch
    result = await run_odds_fetch()
    return result


async def run_grading():
    """Run the grading pipeline on pending bets to create PickOutcome labels."""
    from app.grade_outcomes import run_grading_pipeline
    print("[scheduler] Grading pipeline started...")
    result = await run_grading_pipeline()
    print(f"[scheduler] Grading complete: {result}")


def shutdown_scheduler():
    """Gracefully shut down the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)