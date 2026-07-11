"""Utility functions for odds comparison and computation."""

from typing import Optional
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.models import RawOdds, ValueBet


def get_best_odds_for_outcome(
    db: Session,
    event_id: str,
    market_key: str,
    outcome_name: str,
) -> Optional[dict]:
    """Query raw_odds for the best (highest) price across all bookmakers
    for a given event + market + outcome combination.

    Returns:
        dict with "price" and "bookmaker" keys, or None if no data found.
    """
    result = (
        db.query(
            func.max(RawOdds.outcome_price).label("max_price"),
            RawOdds.bookmaker_title,
        )
        .filter(
            RawOdds.id.like(f"{event_id}%"),
            RawOdds.market_key == market_key,
            RawOdds.outcome_name == outcome_name,
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
) -> Optional[dict]:
    """Get the best available odds for a given ValueBet row."""
    # Determine the outcome name from the value bet's team/pick_label
    outcome_name = value_bet.team

    return get_best_odds_for_outcome(
        db=db,
        event_id=value_bet.event_id,
        market_key=value_bet.market_type,
        outcome_name=outcome_name,
    )


def get_consensus_implied_prob(
    db: Session,
    event_id: str,
    market_key: str,
) -> Optional[float]:
    """Compute the median implied probability across all bookmakers
    for a given event + market.

    Uses the home_team odds from the first outcome to compute a
    consensus view. Returns None if no data found.
    """
    # Get all unique bookmaker prices for this event+market
    rows = (
        db.query(RawOdds.outcome_price, RawOdds.outcome_name)
        .filter(
            RawOdds.id.like(f"{event_id}%"),
            RawOdds.market_key == market_key,
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