from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ingestion.fetch_odds import load_latest_odds
from models.train_models import DEFAULT_MODEL_NAME, predict
from processing.feature_engineering import build_features
from retrain.model_registry import get_active_model

router = APIRouter(prefix="/props", tags=["props"])

PROP_MARKETS = {"player_points", "player_rebounds", "player_assists", "player_pass_tds", "player_rush_yds"}


@router.get("/")
def get_props(
    sport: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    player: Optional[str] = Query(None),
):
    odds = load_latest_odds()
    if odds.empty:
        raise HTTPException(status_code=404, detail="No odds data available")

    props = odds[odds["market"].str.startswith("player_", na=False)]

    if sport:
        props = props[props["sport"] == sport]
    if market:
        props = props[props["market"] == market]
    if player:
        props = props[props["outcome"].str.contains(player, case=False, na=False)]

    if props.empty:
        return {"props": [], "count": 0}

    features = build_features(props)
    active = get_active_model() or DEFAULT_MODEL_NAME

    try:
        predictions = predict(features, model_name=active)
    except FileNotFoundError:
        return {
            "props": props.to_dict(orient="records"),
            "count": len(props),
            "predictions_available": False,
        }

    return {
        "props": predictions.to_dict(orient="records"),
        "count": len(predictions),
        "predictions_available": True,
        "model": active,
    }


@router.get("/markets")
def list_prop_markets():
    odds = load_latest_odds()
    if odds.empty:
        return {"markets": []}

    markets = odds[odds["market"].str.startswith("player_", na=False)]["market"].unique().tolist()
    return {"markets": sorted(markets)}
