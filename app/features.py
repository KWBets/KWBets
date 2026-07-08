"""Feature Engineering Pipeline for DoubleDown AI.

Builds enriched feature vectors from raw odds data for ML model consumption.
Handles: implied probability, consensus lines, line movement, home/away,
team form, rest days, and head-to-head history.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import RawOdds, ProcessedFeatures

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------

def implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def remove_juice(home_prob: float, away_prob: float,
                 draw_prob: Optional[float] = None
                 ) -> tuple[float, float, Optional[float]]:
    """Remove the bookmaker overround (juice) from implied probabilities."""
    total = home_prob + away_prob + (draw_prob or 0.0)
    if total <= 0:
        return home_prob, away_prob, draw_prob
    return (home_prob / total,
            away_prob / total,
            (draw_prob / total) if draw_prob is not None else None)


# ---------------------------------------------------------------------------
# Line movement analysis
# ---------------------------------------------------------------------------

def compute_line_movement(df_raw: pd.DataFrame,
                          event_id: str,
                          market_type: str,
                          home_team: str,
                          away_team: str
                          ) -> dict:
    """Compute line movement metrics (change in odds over time).

    Returns dict with keys like home_odds_open, home_odds_current, home_odds_delta_pct, etc.
    Uses earliest and latest odds_timestamp for open/current proxies.
    """
    result = {
        "home_odds_open": None,
        "home_odds_current": None,
        "home_odds_delta_pct": 0.0,
        "away_odds_open": None,
        "away_odds_current": None,
        "away_odds_delta_pct": 0.0,
    }

    subset = df_raw[
        (df_raw["_event_id"] == event_id) &
        (df_raw["market_key"] == market_type)
        ]
    if subset.empty:
        return result

    # Get home and away outcome prices
    home_rows = subset[subset["outcome_name"] == home_team].sort_values("odds_timestamp")
    away_rows = subset[subset["outcome_name"] == away_team].sort_values("odds_timestamp")

    if not home_rows.empty:
        result["home_odds_open"] = float(home_rows.iloc[0]["outcome_price"])
        result["home_odds_current"] = float(home_rows.iloc[-1]["outcome_price"])
        if result["home_odds_open"] and result["home_odds_open"] > 0:
            result["home_odds_delta_pct"] = round(
                ((result["home_odds_current"] - result["home_odds_open"])
                 / result["home_odds_open"]) * 100, 2
            )

    if not away_rows.empty:
        result["away_odds_open"] = float(away_rows.iloc[0]["outcome_price"])
        result["away_odds_current"] = float(away_rows.iloc[-1]["outcome_price"])
        if result["away_odds_open"] and result["away_odds_open"] > 0:
            result["away_odds_delta_pct"] = round(
                ((result["away_odds_current"] - result["away_odds_open"])
                 / result["away_odds_open"]) * 100, 2
            )

    return result


# ---------------------------------------------------------------------------
# Team form estimation
# ---------------------------------------------------------------------------

def estimate_team_form(team: str,
                       df_outcomes: pd.DataFrame,
                       lookback: int = 5
                       ) -> float:
    """Estimate recent form as rolling win rate.

    Uses past outcomes for a team (won=1.0, lost=0.0, draw=0.5).
    Returns win rate (0.0-1.0) over last `lookback` games, or 0.5 if no data.
    """
    team_games = df_outcomes[
        (df_outcomes["home_team"] == team) |
        (df_outcomes["away_team"] == team)
        ].sort_values("commence_time", ascending=False)

    if team_games.empty:
        return 0.5  # neutral prior

    recent = team_games.head(lookback)
    results = []
    for _, row in recent.iterrows():
        result = row.get("result")
        if result is None:
            continue
        if result == "home_win":
            results.append(1.0 if row["home_team"] == team else 0.0)
        elif result == "away_win":
            results.append(1.0 if row["away_team"] == team else 0.0)
        elif result == "draw":
            results.append(0.5)
        else:
            results.append(0.0)

    return float(np.mean(results)) if results else 0.5


def estimate_rest_days(team: str,
                       event_date: datetime,
                       df_outcomes: pd.DataFrame
                       ) -> float:
    """Estimate rest days since the team's last game before event_date."""
    team_games = df_outcomes[
        ((df_outcomes["home_team"] == team) |
         (df_outcomes["away_team"] == team)) &
        (df_outcomes["commence_time"] < event_date)
        ].sort_values("commence_time", ascending=False)

    if team_games.empty:
        return 7.0  # default to a week (well-rested prior)

    last_game = team_games.iloc[0]
    if isinstance(event_date, str):
        event_date = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
    if isinstance(last_game["commence_time"], str):
        last_time = datetime.fromisoformat(
            last_game["commence_time"].replace("Z", "+00:00")
        )
    else:
        last_time = last_game["commence_time"]

    delta = event_date - last_time
    return max(1.0, delta.total_seconds() / 86400.0)  # at least 1 day


def estimate_h2h_record(team_a: str, team_b: str,
                        df_outcomes: pd.DataFrame,
                        lookback: int = 10
                        ) -> dict:
    """Estimate head-to-head record between two teams.

    Returns dict with team_a_wins, team_b_wins, draws, total_meetings.
    """
    h2h = df_outcomes[
        ((df_outcomes["home_team"] == team_a) & (df_outcomes["away_team"] == team_b)) |
        ((df_outcomes["home_team"] == team_b) & (df_outcomes["away_team"] == team_a))
        ].sort_values("commence_time", ascending=False).head(lookback)

    a_wins = 0
    b_wins = 0
    draws = 0

    for _, row in h2h.iterrows():
        result = row.get("result")
        home = row["home_team"]
        away = row["away_team"]
        if result == "home_win":
            if home == team_a:
                a_wins += 1
            else:
                b_wins += 1
        elif result == "away_win":
            if away == team_a:
                a_wins += 1
            else:
                b_wins += 1
        elif result == "draw":
            draws += 1

    return {
        "team_a_wins": a_wins,
        "team_b_wins": b_wins,
        "draws": draws,
        "total_meetings": len(h2h),
    }


# ---------------------------------------------------------------------------
# Main feature engineering pipeline
# ---------------------------------------------------------------------------

def build_enhanced_features(db: Session, lookback_days: int = 90) -> int:
    """Build enhanced features from raw odds data.

    Reads from raw_odds, computes all engineered features, and writes
    to processed_features. Returns count of features created.

    Also handles outcome data for form/H2H features if available
    from pick_outcomes or externally loaded result data.
    """
    logger.info("Building enhanced features...")

    # 1. Load raw odds into a DataFrame for analysis
    # Exclude outrights/futures: market_key != 'outrights' and outcome_name != 'Field'
    raw_rows = (
        db.query(RawOdds)
        .filter(RawOdds.market_key != "outrights")
        .filter(RawOdds.outcome_name != "Field")
        .all()
    )
    if not raw_rows:
        logger.info("No raw odds data to process.")
        return 0

    records = []
    for r in raw_rows:
        # Extract event_id from the raw odds ID format.
        # Format: seed_{sport_key}_{home[:3]}_{away[:3]}_{count}_{bookmaker}_{market}_{outcome_name}
        # The event_id is the first 5 underscore-delimited components.
        id_parts = r.id.split("_") if "_" in r.id else [r.id]
        if len(id_parts) >= 5:
            # event_id = first 5 parts: seed_sportkey_home_away_count
            event_id = "_".join(id_parts[:5])
        else:
            event_id = r.id

        records.append({
            "_event_id": event_id,
            "sport": r.sport,
            "sport_key": r.sport_key,
            "commence_time": r.commence_time,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "bookmaker_key": r.bookmaker_key,
            "bookmaker_title": r.bookmaker_title,
            "market_key": r.market_key,
            "outcome_name": r.outcome_name,
            "outcome_price": r.outcome_price,
            "outcome_point": r.outcome_point,
            "odds_timestamp": r.odds_timestamp,
            "fetched_at": r.fetched_at,
        })

    df_raw = pd.DataFrame(records)

    # 2. Load outcome data if available (for form/H2H)
    # Try to load from pick_outcomes joined with value_bets
    try:
        from app.models import PickOutcome, ValueBet
        outcome_rows = (
            db.query(PickOutcome, ValueBet)
            .join(ValueBet, PickOutcome.value_bet_id == ValueBet.id)
            .all()
        )
        outcome_data = []
        for po, vb in outcome_rows:
            outcome_data.append({
                "home_team": vb.home_team,
                "away_team": vb.away_team,
                "commence_time": vb.commence_time,
                "result": po.actual_outcome,
                "home_score": po.home_score,
                "away_score": po.away_score,
            })

        # Also look for results in a potential results table
        from sqlalchemy import inspect
        inspector = inspect(db.bind)
        if "game_results" in inspector.get_table_names():
            result_rows = db.execute(text("SELECT * FROM game_results")).fetchall()
            for rr in result_rows:
                outcome_data.append({
                    "home_team": rr.home_team,
                    "away_team": rr.away_team,
                    "commence_time": rr.commence_time,
                    "result": rr.result,
                    "home_score": getattr(rr, "home_score", None),
                    "away_score": getattr(rr, "away_score", None),
                })
        df_outcomes = pd.DataFrame(outcome_data) if outcome_data else pd.DataFrame()
    except Exception:
        df_outcomes = pd.DataFrame()

    logger.info(f"Loaded {len(df_raw)} raw odds rows, {len(df_outcomes)} outcome rows")

    # 3. Build feature vectors per event + market
    # First, clear existing processed features
    db.execute(text("DELETE FROM processed_features"))

    now = datetime.now(timezone.utc)
    features_created = 0

    # Group by unique (event_id, market_type)
    grouped = df_raw.groupby(["_event_id", "market_key"])

    for (event_id, market_type), group in grouped:
        row = group.iloc[0]
        home_team = row["home_team"]
        away_team = row["away_team"]
        commence_time = row["commence_time"]

        # Implied probabilities from average consensus odds
        home_odds = group[group["outcome_name"] == home_team]["outcome_price"]
        away_odds = group[group["outcome_name"] == away_team]["outcome_price"]
        draw_mask = ~group["outcome_name"].isin([home_team, away_team])
        draw_odds = group[draw_mask]["outcome_price"] if draw_mask.any() else pd.Series(dtype=float)

        home_odds_avg = home_odds.mean() if not home_odds.empty else None
        away_odds_avg = away_odds.mean() if not away_odds.empty else None
        draw_odds_avg = draw_odds.mean() if not draw_odds.empty else None

        home_prob = implied_prob(home_odds_avg) if home_odds_avg else None
        away_prob = implied_prob(away_odds_avg) if away_odds_avg else None
        draw_prob = implied_prob(draw_odds_avg) if draw_odds_avg else None

        # Remove juice
        if home_prob and away_prob:
            h, a, d = remove_juice(home_prob, away_prob, draw_prob)
            home_implied, away_implied = h, a
            draw_implied = d
        else:
            home_implied = home_prob
            away_implied = away_prob
            draw_implied = draw_prob

        # Spread / total fields
        home_spread = None
        away_spread = None
        over_line = None
        under_line = None

        if market_type == "spreads":
            hs = group[group["outcome_name"] == home_team]
            a_s = group[group["outcome_name"] == away_team]
            if not hs.empty:
                home_spread = hs["outcome_point"].iloc[0] if pd.notna(hs["outcome_point"].iloc[0]) else None
            if not a_s.empty:
                away_spread = a_s["outcome_point"].iloc[0] if pd.notna(a_s["outcome_point"].iloc[0]) else None
        elif market_type == "totals":
            over_d = group[group["outcome_name"].str.lower().str.contains("over", na=False)]
            under_d = group[group["outcome_name"].str.lower().str.contains("under", na=False)]
            if not over_d.empty:
                over_line = over_d["outcome_point"].iloc[0] if pd.notna(over_d["outcome_point"].iloc[0]) else None
            if not under_d.empty:
                under_line = under_d["outcome_point"].iloc[0] if pd.notna(under_d["outcome_point"].iloc[0]) else None

        # Line movement
        movement = compute_line_movement(df_raw, event_id, market_type, home_team, away_team)

        # Home/away indicator (dummy: home=1, away=0)
        home_indicator = 1.0

        # Team form (if outcome data available)
        home_form = estimate_team_form(home_team, df_outcomes) if not df_outcomes.empty else 0.5
        away_form = estimate_team_form(away_team, df_outcomes) if not df_outcomes.empty else 0.5

        # Rest days
        home_rest = estimate_rest_days(home_team, commence_time, df_outcomes) if not df_outcomes.empty else 7.0
        away_rest = estimate_rest_days(away_team, commence_time, df_outcomes) if not df_outcomes.empty else 7.0

        # H2H record
        h2h = estimate_h2h_record(home_team, away_team, df_outcomes) if not df_outcomes.empty else {
            "team_a_wins": 0, "team_b_wins": 0, "draws": 0, "total_meetings": 0
        }

        # Odds dispersion (std dev)
        home_odds_std = home_odds.std() if len(home_odds) > 1 else None
        away_odds_std = away_odds.std() if len(away_odds) > 1 else None

        # Number of bookmakers covering this market
        bookmaker_count = group["bookmaker_key"].nunique()

        feature = ProcessedFeatures(
            sport=row["sport"],
            sport_key=row["sport_key"],
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
            market_type=market_type,
            home_implied_prob=home_implied,
            away_implied_prob=away_implied,
            draw_implied_prob=draw_implied,
            home_odds_avg=home_odds_avg,
            away_odds_avg=away_odds_avg,
            draw_odds_avg=draw_odds_avg,
            home_spread=home_spread,
            away_spread=away_spread,
            over_line=over_line,
            under_line=under_line,
            odds_displayed_count=bookmaker_count,
            home_odds_std=home_odds_std,
            away_odds_std=away_odds_std,
            created_at=now,
        )
        db.add(feature)
        features_created += 1

    db.commit()
    logger.info(f"Enhanced features built: {features_created} rows")
    return features_created


def extract_feature_vector(feature_row: ProcessedFeatures) -> np.ndarray:
    """Convert a ProcessedFeatures row into a numerical feature vector.

    Returns a 1-D numpy array suitable for model inference.
    """
    features = []

    # 1. Implied probabilities (normalized)
    features.append(feature_row.home_implied_prob or 0.5)
    features.append(feature_row.away_implied_prob or 0.5)
    features.append(feature_row.draw_implied_prob or 0.0)

    # 2. Consensus odds
    features.append(feature_row.home_odds_avg or 2.0)
    features.append(feature_row.away_odds_avg or 2.0)
    features.append(feature_row.draw_odds_avg or 0.0)

    # 3. Spread info
    features.append(feature_row.home_spread or 0.0)
    features.append(feature_row.away_spread or 0.0)

    # 4. Over/Under lines
    features.append(feature_row.over_line or 0.0)
    features.append(feature_row.under_line or 0.0)

    # 5. Market depth
    features.append(feature_row.odds_displayed_count or 0)

    # 6. Odds dispersion
    features.append(feature_row.home_odds_std or 0.0)
    features.append(feature_row.away_odds_std or 0.0)

    # 7. Home implied edge (difference from 50%)
    home_edge = (feature_row.home_implied_prob or 0.5) - 0.5
    away_edge = (feature_row.away_implied_prob or 0.5) - 0.5
    features.append(home_edge)
    features.append(away_edge)

    return np.array(features, dtype=np.float32)


def extract_feature_matrix(rows: list[ProcessedFeatures]) -> np.ndarray:
    """Extract feature matrix from a list of ProcessedFeatures rows.

    Returns a 2-D numpy array (samples x features).
    """
    vectors = [extract_feature_vector(r) for r in rows]
    return np.vstack(vectors) if vectors else np.array([])
