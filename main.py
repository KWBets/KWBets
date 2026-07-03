import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler

import config
from api.routes import odds, parlays, predictions, props
from ingestion.fetch_odds import fetch_and_save_odds
from retrain.auto_retrain import run_retrain_if_needed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(fetch_and_save_odds, "interval", hours=1, id="fetch_odds")
    scheduler.add_job(run_retrain_if_needed, "interval", hours=config.RETRAIN_INTERVAL_HOURS, id="auto_retrain")
    scheduler.start()
    logger.info("Scheduler started")
    yield
    scheduler.shutdown()
    logger.info("Scheduler stopped")


app = FastAPI(
    title="KWBets",
    description="Sports betting predictions, odds ingestion, and alert engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(predictions.router, prefix=config.API_PREFIX)
app.include_router(parlays.router, prefix=config.API_PREFIX)
app.include_router(props.router, prefix=config.API_PREFIX)
app.include_router(odds.router, prefix=config.API_PREFIX)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
