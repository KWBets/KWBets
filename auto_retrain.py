from __future__ import annotations

import logging
from datetime import datetime, timezone

import config
from alerts.alert_engine import find_value_bets
from alerts.notifier import notify_value_bets
from ingestion.fetch_odds import load_latest_odds
from models.train_models import DEFAULT_MODEL_NAME, predict, train_model
from processing.feature_engineering import build_features
from retrain.model_registry import get_best_model, register_model, set_active_model

logger = logging.getLogger(__name__)


def run_retrain_if_needed(force: bool = False) -> dict | None:
    """Retrain when enough labeled data exists, then promote if metrics improve."""
    processed_path = config.PROCESSED_DATA_DIR / "labeled_features.parquet"
    if not processed_path.exists() and not force:
        logger.info("No labeled training data at %s; skipping retrain", processed_path)
        return None

    import pandas as pd

    features = pd.read_parquet(processed_path)
    if len(features) < config.MIN_SAMPLES_FOR_RETRAIN and not force:
        logger.info("Only %d samples; need %d for retrain", len(features), config.MIN_SAMPLES_FOR_RETRAIN)
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = f"{DEFAULT_MODEL_NAME}_{timestamp}"

    try:
        metrics = train_model(features, model_name=model_name)
    except ValueError as exc:
        logger.warning("Retrain skipped: %s", exc)
        return None

    register_model(model_name, metrics)

    best = get_best_model()
    if best:
        set_active_model(best)
        logger.info("Promoted %s as active model", best)

    return metrics


def run_prediction_pipeline() -> int:
    """Fetch latest odds, predict, and alert on value bets."""
    odds = load_latest_odds()
    if odds.empty:
        logger.info("No odds data available for prediction pipeline")
        return 0

    features = build_features(odds)
    active = get_best_model() or DEFAULT_MODEL_NAME

    try:
        predictions = predict(features, model_name=active)
    except FileNotFoundError:
        logger.warning("No trained model found; skipping predictions")
        return 0

    value_bets = find_value_bets(predictions)
    return notify_value_bets(value_bets)
