import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def american_to_implied_prob(price: float) -> float:
    if price > 0:
        return 100 / (price + 100)
    return abs(price) / (abs(price) + 100)


def implied_prob_to_american(prob: float) -> float:
    if prob <= 0 or prob >= 1:
        return 0.0
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    return round(100 * (1 - prob) / prob)


def remove_vig(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total == 0:
        return probs
    return [p / total for p in probs]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    features = df.copy()
    features["implied_prob"] = features["price"].apply(american_to_implied_prob)

    # Best line per event/market/outcome across bookmakers
    best = (
        features.sort_values("price", ascending=False)
        .groupby(["event_id", "market", "outcome"], as_index=False)
        .first()
    )

    # Market consensus: average implied prob across bookmakers
    consensus = (
        features.groupby(["event_id", "market", "outcome"])["implied_prob"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "consensus_prob", "std": "prob_std", "count": "bookmaker_count"})
    )

    best = best.merge(consensus, on=["event_id", "market", "outcome"], how="left")
    best["edge_vs_consensus"] = best["implied_prob"] - best["consensus_prob"]

    # Spread/total distance from line
    if "point" in best.columns:
        best["point"] = best["point"].fillna(0)

    best["is_home"] = (best["outcome"] == best["home_team"]).astype(int)
    best["commence_time"] = pd.to_datetime(best["commence_time"], utc=True)
    best["hours_until_start"] = (
        (best["commence_time"] - pd.Timestamp.now(tz="UTC")).dt.total_seconds() / 3600
    )

    logger.info("Built features for %d rows", len(best))
    return best


def prepare_training_data(features: pd.DataFrame, target_col: str = "won") -> tuple[pd.DataFrame, pd.Series]:
    if features.empty or target_col not in features.columns:
        return pd.DataFrame(), pd.Series(dtype=float)

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {target_col, "price"}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = features[feature_cols].fillna(0)
    y = features[target_col]
    return X, y
