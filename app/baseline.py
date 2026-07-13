"""Baseline predictor — generates model_predictions and value_bets from
de-vigged consensus odds for the cold-start path.

Two responsibilities:
1. run_baseline_predictions() — writes ModelPrediction rows for upcoming events
2. run_baseline_value_bets() — writes ValueBet rows for ALL baseline predictions
   (bypasses the 3% edge gate, since baseline ≈ market means edge ≈ 0)

INTEGRITY: run_baseline_value_bets filters to commence_time > now so it never
creates "pending" picks for events that already started. A pick timestamped after
commencement corrupts the graded track record.

Idempotency: checks existing (event_id, version) before writing.
Transaction ownership: does NOT commit — caller owns the commit.
"""

import logging
from datetime import datetime, timezone

from app.models import ProcessedFeatures, ModelPrediction, ValueBet

logger = logging.getLogger(__name__)

BASELINE_VERSION = "baseline-v0"


def run_baseline_predictions(db) -> int:
    """Generate baseline ModelPrediction rows for upcoming h2h events.

    Uses de-vigged consensus implied probabilities from ProcessedFeatures.
    Skips events already predicted (idempotent), skips started events.

    Does NOT commit — caller owns transaction.
    """
    now = datetime.now(timezone.utc)

    features = (
        db.query(ProcessedFeatures)
        .filter(
            ProcessedFeatures.market_type == "h2h",
            ProcessedFeatures.commence_time > now,
        )
        .all()
    )
    if not features:
        return 0

    existing_preds = set(
        row[0]
        for row in db.query(ModelPrediction.event_id)
        .filter(ModelPrediction.model_version == BASELINE_VERSION)
        .all()
    )

    count = 0
    for feature in features:
        if feature.event_id in existing_preds:
            continue

        home_prob = feature.home_implied_prob
        away_prob = feature.away_implied_prob
        draw_prob = feature.draw_implied_prob

        if home_prob is None or away_prob is None:
            continue

        total = home_prob + away_prob + (draw_prob or 0.0)
        if total <= 0:
            continue

        home_norm = round(home_prob / total, 4)
        away_norm = round(away_prob / total, 4)
        draw_norm = round(draw_prob / total, 4) if draw_prob else None

        pred = ModelPrediction(
            event_id=feature.event_id,
            sport=feature.sport,
            home_team=feature.home_team,
            away_team=feature.away_team,
            commence_time=feature.commence_time,
            market_type=feature.market_type,
            home_win_probability=home_norm,
            away_win_probability=away_norm,
            draw_probability=draw_norm,
            model_version=BASELINE_VERSION,
            model_run_id=BASELINE_VERSION,
            feature_timestamp=now,
            created_at=now,
        )
        db.add(pred)
        count += 1
        existing_preds.add(feature.event_id)

    if count > 0:
        logger.info(
            "[baseline] Created %d predictions (version=%s)", count, BASELINE_VERSION
        )
    return count


def run_baseline_value_bets(db) -> int:
    """Create ValueBet rows for ALL baseline predictions, bypassing edge gate.

    Called only in cold-start mode (no active model in registry) — NOT based on
    value_bets == 0, which could be a normal quiet-day outcome from a real model.

    INTEGRITY GUARD: filters to predictions where commence_time > now.
    Never creates "pending" picks for events that already started or finished —
    a pick timestamped after commencement would corrupt the graded track record.

    Keeps both-sides picks (home AND away for each event) for calibration.

    Does NOT commit — caller owns transaction.
    """
    now = datetime.now(timezone.utc)

    # CRITICAL: only baseline predictions for future events
    # A prediction for an event that already started must never become a ValueBet
    predictions = (
        db.query(ModelPrediction)
        .filter(
            ModelPrediction.model_version == BASELINE_VERSION,
            ModelPrediction.commence_time > now,
        )
        .all()
    )
    if not predictions:
        return 0

    # Idempotency: skip events that already have baseline ValueBets
    existing_vbs = set(
        row[0]
        for row in db.query(ValueBet.event_id)
        .filter(ValueBet.model_version == BASELINE_VERSION)
        .all()
    )

    # Index features by event_id for odds data
    # Only include features with fresh odds data (commence_time > now)
    features = (
        db.query(ProcessedFeatures)
        .filter(ProcessedFeatures.commence_time > now)
        .all()
    )
    feat_by_event = {f.event_id: f for f in features}

    count = 0
    for pred in predictions:
        if pred.event_id in existing_vbs:
            continue

        feature = feat_by_event.get(pred.event_id)
        if not feature:
            # Feature data is stale or missing — skip to avoid corrupting track record
            logger.debug(
                "[baseline] Skipping ValueBet for %s: no fresh feature data",
                pred.event_id,
            )
            continue

        home_prob = pred.home_win_probability or 0.5
        away_prob = pred.away_win_probability or 0.5
        market_home = feature.home_implied_prob
        market_away = feature.away_implied_prob

        if market_home is None or market_away is None:
            continue

        home_odds = 1.0 / market_home
        away_odds = 1.0 / market_away
        home_ev = (home_prob * home_odds) - 1.0
        away_ev = (away_prob * away_odds) - 1.0
        home_edge = (home_prob - market_home) * 100
        away_edge = (away_prob - market_away) * 100

        # Both-sides picks: create ValueBets for home AND away teams
        # This is deliberate — calibration needs both outcomes to measure
        # accuracy, not just the "favored" side

        # Home team pick
        db.add(ValueBet(
            event_id=pred.event_id,
            sport=pred.sport,
            sport_key=feature.sport_key,
            home_team=pred.home_team,
            away_team=pred.away_team,
            commence_time=pred.commence_time,
            team=pred.home_team,
            market_type="h2h",
            pick_label=pred.home_team,
            odds=round(home_odds, 2),
            model_probability=round(home_prob, 4),
            implied_probability=round(market_home, 4),
            edge_percentage=round(home_edge, 2),
            expected_value=round(home_ev, 4),
            confidence_tier="low",
            confidence_score=round(abs(home_edge) / 15.0, 4),
            reasoning_factors={
                "source": BASELINE_VERSION,
                "method": "de-vigged_consensus",
            },
            model_version=BASELINE_VERSION,
            status="pending",
        ))
        count += 1

        # Away team pick
        db.add(ValueBet(
            event_id=pred.event_id,
            sport=pred.sport,
            sport_key=feature.sport_key,
            home_team=pred.home_team,
            away_team=pred.away_team,
            commence_time=pred.commence_time,
            team=pred.away_team,
            market_type="h2h",
            pick_label=pred.away_team,
            odds=round(away_odds, 2),
            model_probability=round(away_prob, 4),
            implied_probability=round(market_away, 4),
            edge_percentage=round(away_edge, 2),
            expected_value=round(away_ev, 4),
            confidence_tier="low",
            confidence_score=round(abs(away_edge) / 15.0, 4),
            reasoning_factors={
                "source": BASELINE_VERSION,
                "method": "de-vigged_consensus",
            },
            model_version=BASELINE_VERSION,
            status="pending",
        ))
        count += 1

        existing_vbs.add(pred.event_id)

    if count > 0:
        logger.info(
            "[baseline] Created %d value bets from %d baseline predictions",
            count,
            len(predictions),
        )
    return count
