"""Utility functions for odds comparison and computation."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.models import RawOdds, ValueBet


def get_best_odds_for_outcome(
    db: Session,
    event_id: str,
    market_key: str,
    outcome_name: str,
    max_age_hours: int = 168,  # 7 days — covers gaps between 6h fetches
    max_price: float = 50.0,
) -> Optional[dict]:
    """Query raw_odds for the best (highest) price across all bookmakers
    for a given event + market + outcome combination.

    Filters:
      - max_age_hours: only consider odds fetched within this many hours
      - max_price: exclude unreasonable prices above this threshold
      - Only returns pre-game lines (commence_time in the future)
      - Only returns bettable odds (1.10 - 15.0 range)

    Returns:
        dict with "price" and "bookmaker" keys, or None if no data found.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)

    result = (
        db.query(
            func.max(RawOdds.outcome_price).label("max_price"),
            RawOdds.bookmaker_title,
        )
        .filter(
            RawOdds.id.like(f"{event_id}%"),
            RawOdds.market_key == market_key,
            RawOdds.outcome_name == outcome_name,
            RawOdds.fetched_at >= cutoff,
            RawOdds.commence_time > now,
            RawOdds.outcome_price.between(1.10, 15.0),
        )
        .group_by(RawOdds.bookmaker_title)
        .order_by(func.max(RawOdds.outcome_price).desc())
        .first()
    )

    if result and result.max_price:
        return {
            "price": round(float(result.max_price), 2),
            "bookmaker": result.bookmaker_title,
        }
    return None


def get_best_odds_for_value_bet(
    db: Session,
    value_bet: ValueBet,
    max_age_hours: int = 168,
    max_price: float = 50.0,
) -> Optional[dict]:
    """Get the best available odds for a given ValueBet row."""
    outcome_name = value_bet.team

    return get_best_odds_for_outcome(
        db=db,
        event_id=value_bet.event_id,
        market_key=value_bet.market_type,
        outcome_name=outcome_name,
        max_age_hours=max_age_hours,
        max_price=max_price,
    )


def get_consensus_implied_prob(
    db: Session,
    event_id: str,
    market_key: str,
    max_age_hours: int = 168,
    max_price: float = 50.0,
) -> Optional[float]:
    """Compute the median implied probability across all bookmakers
    for a given event + market.

    Filters:
      - max_age_hours: only consider odds fetched within this many hours
      - Only pre-game lines (commence_time in the future)
      - Only bettable odds (1.10 - 15.0 range)

    Returns the median implied probability as a percentage (0-100),
    or None if no data found.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)

    rows = (
        db.query(RawOdds.outcome_price, RawOdds.outcome_name)
        .filter(
            RawOdds.id.like(f"{event_id}%"),
            RawOdds.market_key == market_key,
            RawOdds.fetched_at >= cutoff,
            RawOdds.commence_time > now,
            RawOdds.outcome_price.between(1.10, 15.0),
        )
        .all()
    )

    if not rows:
        return None

    # Compute implied probabilities for each outcome (remove vig)
    prices = []
    for row in rows:
        if row.outcome_price and row.outcome_price > 0:
            prices.append(1.0 / row.outcome_price)

    if not prices:
        return None

    # Return median implied probability
    prices.sort()
    mid = len(prices) // 2
    if len(prices) % 2 == 0:
        median = (prices[mid - 1] + prices[mid]) / 2
    else:
        median = prices[mid]

    return round(median * 100, 2)