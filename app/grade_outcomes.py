"""Grading pipeline — fetches scores from The Odds API, matches them to value bets,
writes PickOutcome records, and updates bet statuses.

This creates a real labelled dataset for model retraining.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import settings
from app.database import SessionLocal
from app.models import ValueBet, PickOutcome

logger = logging.getLogger(__name__)

# Stale cutoff: bets older than this many days with no score data → cancelled
STALE_DAYS = 7

# The Odds API base
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


# ---------------------------------------------------------------------------
# API: fetch scores for a given sport
# ---------------------------------------------------------------------------

async def fetch_scores_for_sport(
    client: httpx.AsyncClient,
    sport: str,
    api_key: str,
    days_from: int = 3,
) -> list[dict]:
    """Fetch completed event scores from The Odds API /scores endpoint.

    Args:
        client: httpx async client
        sport: sport key (e.g. 'baseball_mlb')
        api_key: The Odds API key
        days_from: how many days back to look for completed events

    Returns:
        List of event dicts with keys:
        {id, commence_time, home_team, away_team, scores: [{name, score}], completed}
    """
    url = f"{ODDS_API_BASE}/sports/{sport}/scores/"
    params = {
        "apiKey": api_key,
        "daysFrom": days_from,
    }
    try:
        resp = await client.get(url, params=params, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(f"[grade] Sport {sport} not found or no scores available")
            return []
        logger.warning(f"[grade] HTTP {e.response.status_code} for {sport}: {e.response.text[:200]}")
        return []
    except httpx.TimeoutException:
        logger.warning(f"[grade] Timeout fetching scores for {sport}")
        return []
    except Exception as e:
        logger.warning(f"[grade] Error fetching scores for {sport}: {e}")
        return []


# ---------------------------------------------------------------------------
# Team name normalization
# ---------------------------------------------------------------------------

def normalize_team(name: str) -> str:
    """Normalize a team name for matching.

    Steps:
    - lowercase
    - strip leading/trailing whitespace
    - remove punctuation except hyphens and apostrophes
    - strip common suffixes (FC, SC, AFC, etc.)
    """
    if not name:
        return ""

    name = name.lower().strip()

    # Remove punctuation (keep hyphens, apostrophes, periods for abbreviations)
    import re
    name = re.sub(r'[^\w\s\'-.]', '', name)

    # Remove common suffixes
    suffixes = [
        r'\bfc\b', r'\bsc\b', r'\bafc\b', r'\bnfc\b',
        r'\bcf\b', r'\breal\b', r'\bunited\b', r'\bcity\b',
        r'\bac\b', r'\bbc\b', r'\bde\b', r'\bcska\b',
        r'\blos angeles\b', r'\bnew york\b', r'\bsan francisco\b',
        r'\blas vegas\b',
    ]
    for pat in suffixes:
        name = re.sub(pat, '', name)

    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name


# ---------------------------------------------------------------------------
# Match a value bet to a scored event
# ---------------------------------------------------------------------------

def match_bet_to_event(
    value_bet: ValueBet,
    scored_events: list[dict],
) -> Optional[dict]:
    """Match a value bet to a scored event.

    Strategy:
    1. Try exact event_id match if the value_bet's event_id matches the API event id.
    2. Fall back to (normalize(home_team), normalize(away_team), commence_time date).

    Args:
        value_bet: A ValueBet row to match
        scored_events: List of event dicts from The Odds API scores endpoint

    Returns:
        The matched event dict, or None if no match found.
    """
    bet_eid = (value_bet.event_id or "").strip()

    # --- Strategy 1: exact event_id match ---
    for ev in scored_events:
        api_eid = (ev.get("id") or "").strip()
        if api_eid and bet_eid and api_eid == bet_eid:
            return ev

    # --- Strategy 2: normalized team + date match ---
    bet_home_norm = normalize_team(value_bet.home_team)
    bet_away_norm = normalize_team(value_bet.away_team)
    bet_date = value_bet.commence_time.date() if value_bet.commence_time else None

    for ev in scored_events:
        ev_home_norm = normalize_team(ev.get("home_team", ""))
        ev_away_norm = normalize_team(ev.get("away_team", ""))
        try:
            ev_date = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).date() if ev.get("commence_time") else None
        except (ValueError, TypeError):
            ev_date = None

        # Check if teams match (home/away can be swapped in some APIs)
        if bet_date and ev_date and bet_date == ev_date:
            if (bet_home_norm == ev_home_norm and bet_away_norm == ev_away_norm):
                return ev
            # Try swapped (some APIs list teams alphabetically)
            if (bet_home_norm == ev_away_norm and bet_away_norm == ev_home_norm):
                return ev

    return None


# ---------------------------------------------------------------------------
# Grade a single pick
# ---------------------------------------------------------------------------

def grade_pick(
    value_bet: ValueBet,
    event_data: dict,
) -> dict:
    """Determine the outcome of a value bet based on actual scores.

    Args:
        value_bet: The ValueBet row
        event_data: Scored event dict from The Odds API

    Returns:
        Dict with: actual_outcome, home_score, away_score, covered_spread, over_hit
    """
    # Extract scores
    scores_list = event_data.get("scores", [])
    home_score = None
    away_score = None
    for s in scores_list:
        if s.get("name", "").lower() == value_bet.home_team.lower():
            home_score = s.get("score")
        if s.get("name", "").lower() == value_bet.away_team.lower():
            away_score = s.get("score")

    # Guard: scores must be present and numeric
    if home_score is None or away_score is None:
        return {
            "actual_outcome": "unknown",
            "home_score": home_score,
            "away_score": away_score,
            "covered_spread": None,
            "over_hit": None,
        }

    try:
        home_score = float(home_score)
        away_score = float(away_score)
    except (ValueError, TypeError):
        return {
            "actual_outcome": "unknown",
            "home_score": home_score,
            "away_score": away_score,
            "covered_spread": None,
            "over_hit": None,
        }

    total_score = home_score + away_score
    market = value_bet.market_type or "h2h"
    team = (value_bet.team or "").lower()
    home_team = (value_bet.home_team or "").lower()
    away_team = (value_bet.away_team or "").lower()

    actual_outcome = "lost"
    covered_spread = None
    over_hit = None

    if market == "h2h":
        # For h2h, pick wins if picked team's score > opponent's
        if team == home_team and home_score > away_score:
            actual_outcome = "won"
        elif team == away_team and away_score > home_score:
            actual_outcome = "won"
        elif home_score == away_score:
            actual_outcome = "push"

    elif market == "spreads":
        # Spread: pick wins if (team score + spread_line) > opponent score
        pick_point = value_bet.odds_displayed_count or 0  # not available here; use implied
        # We need the spread line. It's stored in ValueBet.pick_label sometimes.
        # Default: if we can't determine, mark unknown.
        # For now use a simple heuristic: the pick label often contains the spread
        import re
        spread_match = re.search(r'([+-]?\d+\.?\d*)', value_bet.pick_label or "")
        spread_line = float(spread_match.group(1)) if spread_match else 0.0

        if team == home_team:
            adjusted_score = home_score + spread_line
            if adjusted_score > away_score:
                actual_outcome = "won"
            elif adjusted_score == away_score:
                actual_outcome = "push"
            covered_spread = adjusted_score > away_score
        elif team == away_team:
            adjusted_score = away_score + spread_line
            if adjusted_score > home_score:
                actual_outcome = "won"
            elif adjusted_score == home_score:
                actual_outcome = "push"
            covered_spread = adjusted_score > home_score

    elif market == "totals":
        # Totals: pick wins if total score over/under the line
        import re
        total_match = re.search(r'(\d+\.?\d*)', value_bet.pick_label or "")
        total_line = float(total_match.group(1)) if total_match else 0.0
        pick_lower = (value_bet.pick_label or "").lower()

        if "over" in pick_lower:
            if total_score > total_line:
                actual_outcome = "won"
            elif total_score == total_line:
                actual_outcome = "push"
            over_hit = total_score > total_line
        elif "under" in pick_lower:
            if total_score < total_line:
                actual_outcome = "won"
            elif total_score == total_line:
                actual_outcome = "push"
            over_hit = total_score > total_line

    return {
        "actual_outcome": actual_outcome,
        "home_score": home_score,
        "away_score": away_score,
        "covered_spread": covered_spread,
        "over_hit": over_hit,
    }


# ---------------------------------------------------------------------------
# Main orchestrating pipeline
# ---------------------------------------------------------------------------

async def run_grading_pipeline(db: Optional[Session] = None) -> dict:
    """Run the grading pipeline: fetch scores, match, grade, write outcomes.

    Args:
        db: Optional database session (created internally if None)

    Returns:
        Dict with counts: graded, stale, unmatched, errors
    """
    logger.info("=== Running grading pipeline ===")

    result = {
        "graded": 0,
        "stale": 0,
        "unmatched": 0,
        "errors": 0,
        "sports_processed": 0,
    }

    own_session = False
    if db is None:
        db = SessionLocal()
        own_session = True

    try:
        now = datetime.now(timezone.utc)

        # 1. Query pending bets that have already started
        pending_bets = (
            db.query(ValueBet)
            .filter(
                ValueBet.status == "pending",
                ValueBet.commence_time < now,
            )
            .all()
        )

        if not pending_bets:
            logger.info("[grade] No pending bets to grade")
            return result

        logger.info(f"[grade] Found {len(pending_bets)} pending bets to check")

        # 2. Group by sport_key
        bets_by_sport: dict[str, list[ValueBet]] = {}
        for bet in pending_bets:
            sk = bet.sport_key or "unknown"
            bets_by_sport.setdefault(sk, []).append(bet)

        logger.info(f"[grade] Grouped into {len(bets_by_sport)} sports: {list(bets_by_sport.keys())}")

        # 3. For each sport, fetch scores and grade
        api_key = settings.odds_api_key
        if not api_key:
            logger.warning("[grade] No ODDS_API_KEY configured — skipping API fetch")
            return result

        async with httpx.AsyncClient() as client:
            cutoff_date = now - timedelta(days=STALE_DAYS)

            for sport_key, sport_bets in bets_by_sport.items():
                logger.info(f"[grade] Processing {sport_key}: {len(sport_bets)} bets")

                # Only try API for sports with active seasons that have scores
                # Fetch completed events
                scored_events = await fetch_scores_for_sport(client, sport_key, api_key, days_from=3)

                # Index scored events by event_id for fast lookup
                scored_by_id = {ev.get("id", ""): ev for ev in scored_events if ev.get("id")}

                logger.info(f"[grade]   Fetched {len(scored_events)} scored events for {sport_key}")

                for bet in sport_bets:
                    try:
                        # Check stale cutoff first
                        if bet.commence_time:
                            # Make cutoff_date naive for comparison with DB timestamps
                            cutoff_naive = cutoff_date.replace(tzinfo=None)
                            bet_ct = bet.commence_time
                            # Make bet.commence_time also naive if it has tzinfo
                            if hasattr(bet_ct, 'tzinfo') and bet_ct.tzinfo is not None:
                                bet_ct = bet_ct.replace(tzinfo=None)
                            if bet_ct < cutoff_naive:
                                # Try to match before marking stale
                                event = match_bet_to_event(bet, scored_events)
                                if not event:
                                    # Truly stale — no scores available after 7 days
                                    bet.status = "cancelled"
                                    db.add(bet)
                                    result["stale"] += 1
                                    continue

                        # Try to match to a scored event
                        event = match_bet_to_event(bet, scored_events)
                        if not event:
                            result["unmatched"] += 1
                            logger.debug(f"[grade]   Unmatched bet: {bet.home_team} vs {bet.away_team} ({bet.event_id})")
                            continue

                        # Grade the matched bet
                        grade_result = grade_pick(bet, event)

                        if grade_result["actual_outcome"] == "unknown":
                            logger.debug(f"[grade]   Unknown outcome (no scores): {bet.home_team} vs {bet.away_team}")
                            continue

                        # Write PickOutcome record
                        outcome = PickOutcome(
                            value_bet_id=bet.id,
                            event_id=bet.event_id,
                            model_probability=bet.model_probability,
                            implied_probability=bet.implied_probability,
                            market_type=bet.market_type,
                            pick_team=bet.team,
                            actual_outcome=grade_result["actual_outcome"],
                            home_score=grade_result["home_score"],
                            away_score=grade_result["away_score"],
                            covered_spread=grade_result["covered_spread"],
                            over_hit=grade_result["over_hit"],
                            resolved_at=now,
                            created_at=now,
                        )
                        db.add(outcome)

                        # Update value_bet status
                        bet.status = grade_result["actual_outcome"]
                        bet.actual_result = grade_result["actual_outcome"]
                        bet.updated_at = now
                        db.add(bet)

                        result["graded"] += 1

                    except Exception as e:
                        logger.error(f"[grade] Error grading bet {bet.id}: {e}", exc_info=True)
                        result["errors"] += 1

                result["sports_processed"] += 1

        db.commit()
        logger.info(
            f"[grade] Pipeline complete: {result['graded']} graded, "
            f"{result['stale']} stale/cancelled, "
            f"{result['unmatched']} unmatched, "
            f"{result['errors']} errors"
        )

    except Exception as e:
        logger.error(f"[grade] Pipeline error: {e}", exc_info=True)
        db.rollback()
    finally:
        if own_session:
            db.close()

    return result


# ---------------------------------------------------------------------------
# Synchronous wrapper (for scheduler)
# ---------------------------------------------------------------------------

def scheduled_grading():
    """Synchronous entry point called by APScheduler."""
    logger.info("[grade] Scheduled grading triggered")
    result = asyncio.run(run_grading_pipeline())
    logger.info(f"[grade] Scheduled grading result: {result}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(run_grading_pipeline())
    print(f"Result: {result}")