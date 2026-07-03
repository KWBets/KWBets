import itertools

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from alerts.alert_engine import find_value_bets
from ingestion.fetch_odds import load_latest_odds
from models.train_models import DEFAULT_MODEL_NAME, predict
from processing.feature_engineering import american_to_implied_prob, build_features
from retrain.model_registry import get_active_model

router = APIRouter(prefix="/parlays", tags=["parlays"])


class ParlayRequest(BaseModel):
    legs: list[str] = Field(..., min_length=2, max_length=12, description="Event IDs to combine")
    stake: float = Field(100.0, gt=0)


def _american_parlay_odds(prices: list[float]) -> float:
    decimal_odds = []
    for price in prices:
        if price > 0:
            decimal_odds.append(1 + price / 100)
        else:
            decimal_odds.append(1 + 100 / abs(price))
    combined = 1.0
    for d in decimal_odds:
        combined *= d
    return round((combined - 1) * 100)


@router.post("/build")
def build_parlay(request: ParlayRequest):
    odds = load_latest_odds()
    if odds.empty:
        raise HTTPException(status_code=404, detail="No odds data available")

    features = build_features(odds)
    active = get_active_model() or DEFAULT_MODEL_NAME

    try:
        predictions = predict(features, model_name=active)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No trained model available")

    legs = []
    for event_id in request.legs:
        leg = predictions[predictions["event_id"] == event_id]
        if leg.empty:
            raise HTTPException(status_code=404, detail=f"No prediction for event {event_id}")
        best = leg.loc[leg["predicted_prob"].idxmax()]
        legs.append(best.to_dict())

    prices = [leg["price"] for leg in legs]
    combined_prob = 1.0
    for leg in legs:
        combined_prob *= leg["predicted_prob"]

    parlay_odds = _american_parlay_odds(prices)
    implied = american_to_implied_prob(parlay_odds) if parlay_odds else 0
    edge = combined_prob - implied

    return {
        "legs": legs,
        "leg_count": len(legs),
        "combined_model_prob": round(combined_prob, 4),
        "parlay_odds": parlay_odds,
        "implied_prob": round(implied, 4),
        "edge": round(edge, 4),
        "potential_payout": round(request.stake * (parlay_odds / 100), 2) if parlay_odds > 0 else None,
    }


@router.get("/suggestions")
def suggest_parlays(
    min_legs: int = Query(2, ge=2, le=6),
    max_legs: int = Query(3, ge=2, le=6),
    min_edge: float = Query(0.03),
    limit: int = Query(5, ge=1, le=20),
):
    odds = load_latest_odds()
    if odds.empty:
        raise HTTPException(status_code=404, detail="No odds data available")

    features = build_features(odds)
    active = get_active_model() or DEFAULT_MODEL_NAME

    try:
        predictions = predict(features, model_name=active)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No trained model available")

    value_bets = find_value_bets(predictions, min_edge=min_edge)
    if value_bets.empty:
        return {"suggestions": []}

    # One best bet per event
    top_per_event = (
        value_bets.sort_values("edge", ascending=False)
        .drop_duplicates("event_id")
        .head(max_legs * limit)
    )

    suggestions = []
    for size in range(min_legs, max_legs + 1):
        for combo in itertools.combinations(top_per_event.itertuples(), size):
            events = {row.event_id for row in combo}
            if len(events) != size:
                continue
            prices = [row.price for row in combo]
            combined_prob = 1.0
            for row in combo:
                combined_prob *= row.predicted_prob
            parlay_odds = _american_parlay_odds(prices)
            implied = american_to_implied_prob(parlay_odds)
            edge = combined_prob - implied
            if edge >= min_edge:
                suggestions.append({
                    "legs": [row._asdict() for row in combo],
                    "combined_model_prob": round(combined_prob, 4),
                    "parlay_odds": parlay_odds,
                    "edge": round(edge, 4),
                })

    suggestions.sort(key=lambda s: s["edge"], reverse=True)
    return {"suggestions": suggestions[:limit]}
