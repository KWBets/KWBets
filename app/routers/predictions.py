"""API router for predictions (value bets) and parlays."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import ValueBet, ModelPrediction
from app.schemas import (
    ValueBetResponse,
    PredictionsListResponse,
    ParlaySuggestion,
    ParlayLeg,
    ParlayBuildRequest,
    ParlayBuildResponse,
    PropsListResponse,
    PropValueBet,
)

router = APIRouter()


@router.get("/predictions", response_model=PredictionsListResponse, tags=["Predictions"])
async def get_predictions(
    sport: Optional[str] = Query(None, description="Filter by sport key"),
    confidence: Optional[str] = Query(None, description="Filter by confidence tier (high, medium, low)"),
    min_edge: float = Query(0.0, ge=0, description="Minimum edge percentage"),
    limit: int = Query(50, ge=1, le=200),
    sort_by: str = Query("edge_percentage", pattern="^(edge_percentage|expected_value|confidence_score)$"),
    db: Session = Depends(get_db),
):
    """Get upcoming value bets, deduplicated by matchup, sorted by EV.

    Only returns picks with commence_time within the next 14 days
    (filters out distant futures).
    """
    now = datetime.now(timezone.utc)
    fourteen_days = now + timedelta(days=14)

    # Base query: only upcoming games (within 14 days), pending status
    query = db.query(ValueBet).filter(
        ValueBet.status == "pending",
        ValueBet.commence_time > now,
        ValueBet.commence_time <= fourteen_days,
    )

    if sport:
        query = query.filter(ValueBet.sport_key == sport)
    if confidence:
        query = query.filter(ValueBet.confidence_tier == confidence)
    if min_edge > 0:
        query = query.filter(ValueBet.edge_percentage >= min_edge)

    # Fetch all matching rows (limit is applied after dedup)
    sort_col = getattr(ValueBet, sort_by, ValueBet.expected_value)
    query = query.order_by(desc(ValueBet.expected_value))
    rows = query.all()

    # Deduplicate: for each (home_team, away_team) matchup, keep the pick with highest expected_value
    seen = {}
    for r in rows:
        key = (r.home_team, r.away_team, r.market_type)
        if key not in seen or r.expected_value > seen[key].expected_value:
            seen[key] = r

    # Sort deduplicated results by the requested sort field
    deduped = list(seen.values())
    deduped.sort(key=lambda r: getattr(r, sort_by, r.expected_value), reverse=True)

    # Apply limit after dedup
    deduped = deduped[:limit]

    return PredictionsListResponse(
        count=len(deduped),
        predictions=[ValueBetResponse.model_validate(r) for r in deduped],
    )


@router.get("/predictions/{prediction_id}", response_model=ValueBetResponse, tags=["Predictions"])
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Get a single value bet by ID."""
    bet = db.query(ValueBet).filter(ValueBet.id == prediction_id).first()
    if not bet:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return ValueBetResponse.model_validate(bet)


@router.get("/parlays/suggestions", response_model=list[ParlaySuggestion], tags=["Parlays"])
async def get_parlay_suggestions(
    sport: Optional[str] = Query(None),
    min_edge: float = Query(1.0),
    max_legs: int = Query(4, ge=2, le=8),
    db: Session = Depends(get_db),
):
    """Generate AI parlay suggestions from uncorrelated high-value legs."""
    query = (
        db.query(ValueBet)
        .filter(
            ValueBet.status == "pending",
            ValueBet.confidence_tier.in_(["high", "medium"]),
            ValueBet.edge_percentage >= min_edge,
        )
        .order_by(desc(ValueBet.edge_percentage))
    )

    if sport:
        query = query.filter(ValueBet.sport_key == sport)

    top_bets = query.limit(50).all()

    if len(top_bets) < 2:
        return []

    # Build parlays by grouping legs from different events (avoid correlation)
    # Simple strategy: pair top high-confidence bets from different events
    suggestions = []

    # Group by event to avoid correlated legs
    from collections import defaultdict
    by_event = defaultdict(list)
    for bet in top_bets:
        by_event[bet.event_id].append(bet)

    # Select one leg per event
    event_ids = list(by_event.keys())

    # Build 2-leg parlays from the best legs in different events
    for i in range(min(3, len(event_ids))):
        for j in range(i + 1, min(i + 4, len(event_ids))):
            leg1 = by_event[event_ids[i]][0]
            leg2 = by_event[event_ids[j]][0]

            combined_odds = leg1.odds * leg2.odds
            combined_implied = (1 / leg1.odds + 1 / leg2.odds)
            combined_model = leg1.model_probability * leg2.model_probability
            combined_edge = (combined_model - combined_implied) * 100

            # Check if events are from the same sport
            correlation_warning = None
            if leg1.sport_key == leg2.sport_key:
                correlation_warning = "Both legs are from the same sport — partial correlation possible."

            suggestions.append(ParlaySuggestion(
                legs=[
                    ParlayLeg(
                        event_id=leg1.event_id,
                        sport=leg1.sport,
                        team=leg1.team,
                        market_type=leg1.market_type,
                        pick_label=leg1.pick_label,
                        odds=leg1.odds,
                        model_probability=leg1.model_probability,
                        edge_percentage=leg1.edge_percentage,
                    ),
                    ParlayLeg(
                        event_id=leg2.event_id,
                        sport=leg2.sport,
                        team=leg2.team,
                        market_type=leg2.market_type,
                        pick_label=leg2.pick_label,
                        odds=leg2.odds,
                        model_probability=leg2.model_probability,
                        edge_percentage=leg2.edge_percentage,
                    ),
                ],
                combined_odds=round(combined_odds, 2),
                combined_implied_prob=round(1 / combined_odds * 100, 2),
                combined_model_prob=round(combined_model * 100, 2),
                combined_edge=round(combined_edge, 2),
                confidence_tier="high" if combined_edge > 10 else "medium" if combined_edge > 5 else "low",
                correlation_warning=correlation_warning,
            ))

            if len(suggestions) >= 5:
                break
        if len(suggestions) >= 5:
            break

    return suggestions[:5]


@router.post("/parlays/build", response_model=ParlayBuildResponse, tags=["Parlays"])
async def build_parlay(request: ParlayBuildRequest, db: Session = Depends(get_db)):
    """Calculate combined odds and edge for a custom parlay."""
    legs = db.query(ValueBet).filter(ValueBet.id.in_(request.leg_ids)).all()

    if len(legs) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 legs for a parlay")

    if len(legs) != len(set(request.leg_ids)):
        raise HTTPException(status_code=400, detail="One or more leg IDs not found")

    # Check for same-event correlation
    event_ids = [l.event_id for l in legs]
    if len(event_ids) != len(set(event_ids)):
        raise HTTPException(
            status_code=400,
            detail="Parlay contains legs from the same event — correlated legs increase risk",
        )

    combined_odds = 1.0
    combined_model_prob = 1.0

    for leg in legs:
        combined_odds *= leg.odds
        combined_model_prob *= leg.model_probability

    combined_implied_prob = 1 / combined_odds * 100
    combined_model_prob_pct = combined_model_prob * 100
    combined_edge = ((combined_model_prob * combined_odds) - 1) * 100

    return ParlayBuildResponse(
        legs=[ValueBetResponse.model_validate(l) for l in legs],
        combined_odds=round(combined_odds, 2),
        combined_implied_prob=round(combined_implied_prob, 2),
        combined_edge=round(combined_edge, 2),
        disclaimer="Parlays carry higher risk. Combined edge assumes independent events. Past performance does not guarantee future results.",
    )


@router.get("/props", response_model=PropsListResponse, tags=["Props"])
async def get_props(
    sport: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Get player prop value bets. (Player props require special API access.)"""
    # For now, return standard value bets as stand-in for props
    # Player props will be added once The Odds API provides them
    query = (
        db.query(ValueBet)
        .filter(ValueBet.status == "pending")
        .order_by(desc(ValueBet.edge_percentage))
        .limit(limit)
    )

    if sport:
        query = query.filter(ValueBet.sport_key == sport)

    rows = query.all()
    props = []
    for r in rows:
        # Use team as player_name placeholder until real props arrive
        props.append(PropValueBet(
            id=r.id,
            sport=r.sport,
            player_name=r.team,
            team=r.home_team if r.team == r.home_team else r.away_team,
            market_type=r.market_type,
            line=0.0,
            odds=r.odds,
            model_probability=r.model_probability,
            implied_probability=r.implied_probability,
            edge_percentage=r.edge_percentage,
            confidence_tier=r.confidence_tier,
        ))

    return PropsListResponse(count=len(props), props=props)