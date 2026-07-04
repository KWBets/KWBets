"""EV Calculation Engine for DoubleDown AI.

Compares model-predicted probabilities against market-implied probabilities
to identify value bets with positive expected value.

Edge = Model Probability - Implied Probability
EV = (Model Probability * Decimal Odds) - 1

Confidence Tiers:
- Low: 3-5% edge
- Medium: 5-8% edge
- High: 8-12% edge
- Elite: 12%+ edge
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    ProcessedFeatures,
    ModelPrediction,
    ValueBet,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Odds conversion utilities
# ---------------------------------------------------------------------------

def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds.

    Positive: +150 -> 2.50
    Negative: -110 -> 1.909
    """
    if american_odds > 0:
        return 1.0 + (american_odds / 100.0)
    else:
        return 1.0 + (100.0 / abs(american_odds))


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability.

    P(implied) = 1 / decimal_odds
    """
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def implied_prob_to_decimal(prob: float) -> float:
    """Convert implied probability back to decimal odds."""
    if prob <= 0 or prob >= 1:
        return 0.0
    return 1.0 / prob


# ---------------------------------------------------------------------------
# Core EV calculation
# ---------------------------------------------------------------------------

def calculate_edge(model_prob: float, implied_prob: float) -> float:
    """Calculate edge = model_prob - implied_prob (as percentage)."""
    return (model_prob - implied_prob) * 100.0


def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """Calculate expected value.

    EV = (model_prob * decimal_odds) - 1.0

    Returns value > 0 if positive expected value.
    """
    return (model_prob * decimal_odds) - 1.0


def get_confidence_tier(edge_percentage: float) -> str:
    """Map edge percentage to confidence tier.

    0-3%: No value (not flagged)
    3-5%: Low
    5-8%: Medium
    8-12%: High
    12%+: Elite
    """
    if edge_percentage >= 12.0:
        return "elite"
    elif edge_percentage >= 8.0:
        return "high"
    elif edge_percentage >= 5.0:
        return "medium"
    elif edge_percentage >= 3.0:
        return "low"
    else:
        return "none"


def get_confidence_score(edge_percentage: float) -> float:
    """Normalized confidence score (0.0 - 1.0) from edge percentage.

    Maps linearly: 0% edge -> 0.0, 15% edge -> 1.0 (cap)
    """
    score = edge_percentage / 15.0
    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Value bet generation from model predictions
# ---------------------------------------------------------------------------

def compute_value_bets_for_event(
    feature: ProcessedFeatures,
    model_prediction: ModelPrediction,
    min_edge: float = 3.0,
) -> list[dict]:
    """Compute value bets for a single event by comparing model vs market.

    Args:
        feature: ProcessedFeatures row with market data
        model_prediction: ModelPrediction row with model output
        min_edge: Minimum edge percentage to flag (default 3.0%)

    Returns:
        List of value bet dicts ready to insert as ValueBet rows.
    """
    value_bets = []

    # Only process h2h markets for now (spreads/totals need more work)
    if feature.market_type != "h2h":
        return []

    model_home = model_prediction.home_win_probability
    model_away = model_prediction.away_win_probability

    market_home = feature.home_implied_prob
    market_away = feature.away_implied_prob

    if model_home is None or model_away is None:
        return []

    # Home team value bet
    if market_home and market_home > 0:
        home_edge = calculate_edge(model_home, market_home)
        home_odds = implied_prob_to_decimal(market_home)
        home_ev = calculate_ev(model_home, home_odds)

        if home_edge >= min_edge and home_ev > 0:
            value_bets.append(_build_value_bet_dict(
                feature=feature,
                team=feature.home_team,
                market_type="h2h",
                pick_label=feature.home_team,
                odds=home_odds,
                model_probability=model_home,
                implied_probability=market_home,
                edge_percentage=home_edge,
                expected_value=home_ev,
                model_version=model_prediction.model_version,
            ))

    # Away team value bet
    if market_away and market_away > 0:
        away_edge = calculate_edge(model_away, market_away)
        away_odds = implied_prob_to_decimal(market_away)
        away_ev = calculate_ev(model_away, away_odds)

        if away_edge >= min_edge and away_ev > 0:
            value_bets.append(_build_value_bet_dict(
                feature=feature,
                team=feature.away_team,
                market_type="h2h",
                pick_label=feature.away_team,
                odds=away_odds,
                model_probability=model_away,
                implied_probability=market_away,
                edge_percentage=away_edge,
                expected_value=away_ev,
                model_version=model_prediction.model_version,
            ))

    # Draw value bet (if applicable)
    model_draw = model_prediction.draw_probability
    market_draw = feature.draw_implied_prob
    if model_draw is not None and market_draw and market_draw > 0:
        draw_edge = calculate_edge(model_draw, market_draw)
        draw_odds = implied_prob_to_decimal(market_draw)
        draw_ev = calculate_ev(model_draw, draw_odds)

        if draw_edge >= min_edge and draw_ev > 0:
            value_bets.append(_build_value_bet_dict(
                feature=feature,
                team="Draw",
                market_type="h2h",
                pick_label=f"Draw ({feature.home_team} vs {feature.away_team})",
                odds=draw_odds,
                model_probability=model_draw,
                implied_probability=market_draw,
                edge_percentage=draw_edge,
                expected_value=draw_ev,
                model_version=model_prediction.model_version,
            ))

    return value_bets


def _build_value_bet_dict(
    feature: ProcessedFeatures,
    team: str,
    market_type: str,
    pick_label: str,
    odds: float,
    model_probability: float,
    implied_probability: float,
    edge_percentage: float,
    expected_value: float,
    model_version: str,
) -> dict:
    """Build a value bet dict from computed metrics."""
    edge = edge_percentage
    confidence_tier = get_confidence_tier(edge)
    confidence_score = get_confidence_score(edge)

    # Simple reasoning factors
    reasoning = {
        "model_probability": round(model_probability, 4),
        "implied_probability": round(implied_probability, 4),
        "edge_pct": round(edge, 2),
        "expected_value": round(expected_value, 4),
        "market_consensus_odds": round(odds, 2),
        "sportsbook_coverage": feature.odds_displayed_count or 0,
    }

    return {
        "event_id": feature.event_id,
        "sport": feature.sport,
        "sport_key": feature.sport_key,
        "home_team": feature.home_team,
        "away_team": feature.away_team,
        "commence_time": feature.commence_time,
        "team": team,
        "market_type": market_type,
        "pick_label": pick_label,
        "odds": round(odds, 2),
        "model_probability": round(model_probability, 4),
        "implied_probability": round(implied_probability, 4),
        "edge_percentage": round(edge, 2),
        "expected_value": round(expected_value, 4),
        "confidence_tier": confidence_tier,
        "confidence_score": confidence_score,
        "reasoning_factors": reasoning,
        "model_version": model_version,
        "best_bookmaker": None,  # would be populated from raw odds analysis
        "is_live": False,
        "status": "pending",
    }


# ---------------------------------------------------------------------------
# Full pipeline: run EV computation across all events
# ---------------------------------------------------------------------------

def run_ev_pipeline(
    db: Session,
    model_run_id: Optional[str] = None,
    min_edge: float = 3.0,
) -> int:
    """Run the full EV calculation pipeline.

    Steps:
    1. Load processed features with matching model predictions
    2. For each feature+prediction pair, compute value bets
    3. Store value bets in the database

    Args:
        db: Database session
        model_run_id: Specific model run to use (None = latest active)
        min_edge: Minimum edge percentage to flag (default 3.0)

    Returns:
        Number of value bets created
    """
    from app.models import ModelRegistry

    # 1. Determine which model version to use
    if model_run_id:
        model_reg = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.model_run_id == model_run_id)
            .first()
        )
        if not model_reg:
            logger.warning(f"Model run {model_run_id} not found in registry")
            return 0
        model_version = model_reg.model_version
    else:
        # Use the active model
        model_reg = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.is_active == True)
            .order_by(ModelRegistry.created_at.desc())
            .first()
        )
        if not model_reg:
            logger.warning("No active model found in registry")
            return 0
        model_version = model_reg.model_version

    logger.info(f"Running EV pipeline with model version: {model_version}")

    # 2. Get all features and predictions
    features = db.query(ProcessedFeatures).all()
    predictions = (
        db.query(ModelPrediction)
        .filter(ModelPrediction.model_version == model_version)
        .all()
    )

    if not features:
        logger.info("No features to process")
        return 0
    if not predictions:
        logger.info(f"No predictions for model version {model_version}")
        return 0

    # Index predictions by event_id
    pred_by_event = {p.event_id: p for p in predictions}

    # 3. Compute value bets
    now = datetime.now(timezone.utc)
    value_bets_created = 0

    for feature in features:
        if feature.event_id not in pred_by_event:
            continue

        prediction = pred_by_event[feature.event_id]
        bets = compute_value_bets_for_event(feature, prediction, min_edge)

        for bet_data in bets:
            # Check if value bet already exists for this event+team+market
            existing = (
                db.query(ValueBet)
                .filter(
                    ValueBet.event_id == bet_data["event_id"],
                    ValueBet.team == bet_data["team"],
                    ValueBet.market_type == bet_data["market_type"],
                    ValueBet.status == "pending",
                )
                .first()
            )
            if existing:
                # Update existing bet
                for key, value in bet_data.items():
                    setattr(existing, key, value)
                existing.created_at = now
            else:
                bet = ValueBet(**bet_data)
                db.add(bet)
                value_bets_created += 1

    db.commit()
    logger.info(f"EV pipeline complete: {value_bets_created} new value bets, "
                f"{len(predictions)} predictions evaluated")
    return value_bets_created


# ---------------------------------------------------------------------------
# Standalone helpers for model prediction integration
# ---------------------------------------------------------------------------

def evaluate_prediction(
    model_probs: dict[str, float],
    market_data: dict,
    feature_row: ProcessedFeatures,
    model_version: str,
) -> Optional[dict]:
    """Evaluate a single prediction and return value bet if applicable.

    Args:
        model_probs: Dict with keys 'home', 'away', 'draw'
        market_data: Dict with implied probs from market
        feature_row: Associated ProcessedFeatures row
        model_version: Version string for the model that made prediction

    Returns:
        Value bet dict or None if no edge found.
    """
    home_prob = model_probs.get("home", 0.5)
    away_prob = model_probs.get("away", 0.5)
    draw_prob = model_probs.get("draw")

    implied_home = market_data.get("home_implied", 0.5)
    implied_away = market_data.get("away_implied", 0.5)
    implied_draw = market_data.get("draw_implied")

    home_edge = calculate_edge(home_prob, implied_home)
    away_edge = calculate_edge(away_prob, implied_away)

    best_edge = 0.0
    best_bet = None

    if home_edge >= 3.0:
        odds = implied_prob_to_decimal(implied_home)
        ev = calculate_ev(home_prob, odds)
        best_edge = home_edge
        best_bet = _build_value_bet_dict(
            feature=feature_row,
            team=feature_row.home_team,
            market_type="h2h",
            pick_label=feature_row.home_team,
            odds=odds,
            model_probability=home_prob,
            implied_probability=implied_home,
            edge_percentage=home_edge,
            expected_value=ev,
            model_version=model_version,
        )

    if away_edge > best_edge and away_edge >= 3.0:
        odds = implied_prob_to_decimal(implied_away)
        ev = calculate_ev(away_prob, odds)
        best_edge = away_edge
        best_bet = _build_value_bet_dict(
            feature=feature_row,
            team=feature_row.away_team,
            market_type="h2h",
            pick_label=feature_row.away_team,
            odds=odds,
            model_probability=away_prob,
            implied_probability=implied_away,
            edge_percentage=away_edge,
            expected_value=ev,
            model_version=model_version,
        )

    if draw_prob is not None and implied_draw:
        draw_edge = calculate_edge(draw_prob, implied_draw)
        if draw_edge > best_edge and draw_edge >= 3.0:
            odds = implied_prob_to_decimal(implied_draw)
            ev = calculate_ev(draw_prob, odds)
            best_edge = draw_edge
            best_bet = _build_value_bet_dict(
                feature=feature_row,
                team="Draw",
                market_type="h2h",
                pick_label=f"Draw",
                odds=odds,
                model_probability=draw_prob,
                implied_probability=implied_draw,
                edge_percentage=draw_edge,
                expected_value=ev,
                model_version=model_version,
            )

    return best_bet
