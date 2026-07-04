"""APScheduler setup for periodic tasks."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.config import settings

scheduler = AsyncIOScheduler()


def start_scheduler():
    """Initialize and start the APScheduler with all jobs."""
    from app.odds_ingestion import run_odds_fetch
    from app.retrain import scheduled_retrain

    # Hourly odds fetch
    scheduler.add_job(
        run_odds_fetch,
        trigger=IntervalTrigger(hours=settings.odds_fetch_interval_hours),
        id="hourly_odds_fetch",
        name="Fetch odds from The Odds API",
        replace_existing=True,
        next_run_time=None,  # Don't run immediately on startup
    )

    # Daily model retraining — calls the full retraining pipeline
    scheduler.add_job(
        scheduled_retrain,
        trigger=IntervalTrigger(hours=settings.model_retrain_interval_hours),
        id="daily_model_retrain",
        name="Run model retraining pipeline",
        replace_existing=True,
        next_run_time=None,
    )

    scheduler.start()
    print("[scheduler] Started APScheduler with hourly odds fetch and daily retrain pipeline.")


async def trigger_odds_fetch_now():
    """Manually trigger an odds fetch (called from API endpoint)."""
    from app.odds_ingestion import run_odds_fetch
    result = await run_odds_fetch()
    return result


def shutdown_scheduler():
    """Gracefully shut down the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)