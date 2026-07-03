from __future__ import annotations

import logging

import pandas as pd

import config
from processing.feature_engineering import american_to_implied_prob, implied_prob_to_american

logger = logging.getLogger(__name__)


def find_value_bets(
    predictions: pd.DataFrame,
    min_edge: float | None = None,
) -> pd.DataFrame:
    """Return bets where model probability exceeds implied odds by min_edge."""
    if predictions.empty:
        return pd.DataFrame()

    min_edge = min_edge if min_edge is not None else config.MIN_EDGE_THRESHOLD
    df = predictions.copy()

    if "implied_prob" not in df.columns and "price" in df.columns:
        df["implied_prob"] = df["price"].apply(american_to_implied_prob)

    df["edge"] = df["predicted_prob"] - df["implied_prob"]
    df["expected_value"] = df["predicted_prob"] * df["price"].apply(
        lambda p: p / 100 if p > 0 else 100 / abs(p)
    ) - (1 - df["predicted_prob"])
    df["fair_odds"] = df["predicted_prob"].apply(implied_prob_to_american)

    value = df[df["edge"] >= min_edge].sort_values("edge", ascending=False)
    logger.info("Found %d value bets (edge >= %.1f%%)", len(value), min_edge * 100)
    return value


def format_alert(bet: pd.Series) -> str:
    return (
        f"VALUE BET: {bet.get('outcome', '?')} "
        f"({bet.get('market', '?')}) — "
        f"{bet.get('home_team', '?')} vs {bet.get('away_team', '?')}\n"
        f"  Book: {bet.get('bookmaker', '?')} @ {bet.get('price', '?')}\n"
        f"  Model: {bet.get('predicted_prob', 0):.1%} | Implied: {bet.get('implied_prob', 0):.1%} | "
        f"Edge: {bet.get('edge', 0):.1%}"
    )
