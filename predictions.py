from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from alerts.alert_engine import find_value_bets
from ingestion.fetch_odds import fetch_and_save_odds, load_latest_odds
from models.train_models import DEFAULT_MODEL_NAME, load_model, predict
from processing.feature_engineering import build_features
from retrain.model_registry import get_active_model

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/")
def get_predictions(
    sport: Optional[str] = Query(None),
    min_edge: Optional[float] = Query(None),
    model_name: Optional[str] = Query(None),
):
    odds = load_latest_odds()
    if odds.empty:
        raise HTTPException(status_code=404, detail="No odds data available. Run ingestion first.")

    if sport:
        odds = odds[odds["sport"] == sport]

    features = build_features(odds)
    active = model_name or get_active_model() or DEFAULT_MODEL_NAME

    try:
        predictions = predict(features, model_name=active)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{active}' not found. Train a model first.")

    value_bets = find_value_bets(predictions, min_edge=min_edge)
    return {
        "model": active,
        "total_predictions": len(predictions),
        "value_bets_count": len(value_bets),
        "predictions": predictions.to_dict(orient="records"),
        "value_bets": value_bets.to_dict(orient="records"),
    }


@router.get("/model")
def get_model_info(model_name: Optional[str] = Query(None)):
    name = model_name or get_active_model() or DEFAULT_MODEL_NAME
    artifact = load_model(name)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {
        "model_name": name,
        "trained_at": artifact.get("trained_at"),
        "metrics": artifact.get("metrics"),
        "feature_columns": artifact.get("feature_columns"),
    }
