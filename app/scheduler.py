"""APScheduler setup for periodic tasks."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.config import settings

scheduler = AsyncIOScheduler()


def start_scheduler():
    """Initialize and start the APScheduler with all jobs."""
    from app.odds_ingestion import run_odds_fetch
    from app.grade_outcomes import scheduled_grading

    # Hourly odds fetch
    scheduler.add_job(
        run_odds_fetch,
        trigger=IntervalTrigger(hours=settings.odds_fetch_interval_hours),
        id="hourly_odds_fetch",
        name="Fetch odds from The Odds API",
        replace_existing=True,
        next_run_time=None,  # Don't run immediately on startup
    )

    # Daily model retraining
    scheduler.add_job(
        run_daily_retrain,
        trigger=IntervalTrigger(hours=settings.model_retrain_interval_hours),
        id="daily_model_retrain",
        name="Retrain ML model on accumulated outcomes + regenerate picks",
        replace_existing=True,
        next_run_time=None,
    )

    # Grading pipeline every 12 hours (max 2x/day to conserve API quota)
    scheduler.add_job(
        run_grading,
        trigger=IntervalTrigger(hours=12),
        id="grading_pipeline",
        name="Grade pending bets via scores API + write PickOutcomes",
        replace_existing=True,
        next_run_time=None,
    )

    scheduler.start()
    print(f"[scheduler] Started APScheduler: odds fetch every {settings.odds_fetch_interval_hours}h + EV chain, daily retrain ({settings.model_retrain_interval_hours}h), grading (12h).")


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