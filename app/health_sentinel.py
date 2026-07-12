"""Health sentinel — 9 automated checks run every 6 hours, emails on failure."""

from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import RawOdds, PickOutcome

# ---------------------------------------------------------------------------
# Module-level state for trend checks
# ---------------------------------------------------------------------------
_last_raw_odds_count: int | None = None
_last_pick_outcome_count: int | None = None
_last_pick_outcome_time: datetime | None = None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

async def check_odds_freshness(db: Session) -> dict[str, Any]:
    """Last odds fetch < 8 hours ago."""
    last = db.query(func.max(RawOdds.fetched_at)).scalar()
    if not last:
        return {
            "check_name": "odds_freshness",
            "status": "critical",
            "actual": "never_fetched",
            "expected": "fetched < 8h ago",
            "message": "No odds have ever been fetched.",
        }
    hours_ago = (datetime.now(timezone.utc) - last).total_seconds() / 3600
    if hours_ago < 8:
        return {
            "check_name": "odds_freshness",
            "status": "pass",
            "actual": f"{hours_ago:.1f}h ago",
            "expected": "fetched < 8h ago",
            "message": f"Last fetch was {hours_ago:.1f} hours ago.",
        }
    return {
        "check_name": "odds_freshness",
        "status": "critical",
        "actual": f"{hours_ago:.1f}h ago",
        "expected": "fetched < 8h ago",
        "message": f"Last fetch was {hours_ago:.1f} hours ago — exceeds 8h threshold.",
    }


async def check_games_board(db: Session) -> dict[str, Any]:
    """/games returns count > 0 AND at least one game within 48h."""
    now = datetime.now(timezone.utc)
    fortyeight = now + timedelta(hours=48)
    count = (
        db.query(func.count(func.distinct(RawOdds.id)))
        .filter(
            RawOdds.market_key == "h2h",
            RawOdds.commence_time > now,
            RawOdds.commence_time <= fortyeight,
            RawOdds.outcome_price.between(1.10, 15.0),
        )
        .scalar()
    ) or 0
    if count > 0:
        return {
            "check_name": "games_board",
            "status": "pass",
            "actual": f"{count} upcoming games",
            "expected": "> 0 games within 48h",
            "message": f"{count} upcoming games available.",
        }
    return {
        "check_name": "games_board",
        "status": "critical",
        "actual": f"{count} upcoming games",
        "expected": "> 0 games within 48h",
        "message": "No upcoming games available in the next 48 hours.",
    }


async def check_no_past_games(db: Session) -> dict[str, Any]:
    """No game has commence_time in the past."""
    now = datetime.now(timezone.utc)
    past = db.query(RawOdds).filter(RawOdds.commence_time < now).first()
    if past:
        return {
            "check_name": "no_past_games",
            "status": "warn",
            "actual": "past games exist",
            "expected": "no games with commence_time in the past",
            "message": f"Found game(s) with commence_time in the past (e.g. {past.home_team} vs {past.away_team}).",
        }
    return {
        "check_name": "no_past_games",
        "status": "pass",
        "actual": "no past games",
        "expected": "no games with commence_time in the past",
        "message": "No games have commence_time in the past.",
    }


async def check_valid_odds(db: Session) -> dict[str, Any]:
    """Every game has valid best_odds (1.10-15.0, US bookmaker)."""
    us_keys = set(settings.us_bookmakers) if isinstance(settings.us_bookmakers, list) else set()
    bad = (
        db.query(RawOdds)
        .filter(
            RawOdds.market_key == "h2h",
            RawOdds.commence_time > datetime.now(timezone.utc),
        )
        .filter(
            (RawOdds.outcome_price < 1.10)
            | (RawOdds.outcome_price > 15.0)
            | (~RawOdds.bookmaker_key.in_(us_keys))
        )
        .first()
    )
    if bad:
        return {
            "check_name": "valid_odds",
            "status": "warn",
            "actual": f"bad row: {bad.bookmaker_key} @ {bad.outcome_price}",
            "expected": "all odds 1.10-15.0 from US books",
            "message": f"Found invalid odds: {bad.bookmaker_key} price={bad.outcome_price} for {bad.home_team} vs {bad.away_team}.",
        }
    return {
        "check_name": "valid_odds",
        "status": "pass",
        "actual": "all valid",
        "expected": "all odds 1.10-15.0 from US books",
        "message": "All odds are in valid range from US bookmakers.",
    }


async def check_bookmaker_allowlist(db: Session) -> dict[str, Any]:
    """No bookmaker outside the allowlist in raw_odds."""
    us_keys = set(settings.us_bookmakers) if isinstance(settings.us_bookmakers, list) else set()
    bad = (
        db.query(RawOdds.bookmaker_key)
        .distinct()
        .filter(~RawOdds.bookmaker_key.in_(us_keys))
        .all()
    )
    if bad:
        keys = [r[0] for r in bad]
        return {
            "check_name": "bookmaker_allowlist",
            "status": "warn",
            "actual": f"{len(keys)} disallowed: {', '.join(keys)}",
            "expected": "only US bookmakers in data",
            "message": f"Found {len(keys)} bookmaker(s) outside allowlist: {', '.join(keys)}.",
        }
    return {
        "check_name": "bookmaker_allowlist",
        "status": "pass",
        "actual": "all allowed",
        "expected": "only US bookmakers in data",
        "message": "All bookmakers in the allowlist.",
    }


async def check_sport_keys(db: Session) -> dict[str, Any]:
    """All sport_keys in the feed are from the known set."""
    known = set(settings.supported_sports)
    found = {r[0] for r in db.query(RawOdds.sport_key).distinct().all()}
    unknown = found - known
    if unknown:
        return {
            "check_name": "sport_keys",
            "status": "warn",
            "actual": f"{len(unknown)} unknown: {', '.join(unknown)}",
            "expected": "only known sport keys",
            "message": f"Found {len(unknown)} unknown sport key(s): {', '.join(unknown)}.",
        }
    return {
        "check_name": "sport_keys",
        "status": "pass",
        "actual": "all known",
        "expected": "only known sport keys",
        "message": "All sport keys are from the known set.",
    }


async def check_odds_api_quota() -> dict[str, Any]:
    """Check Odds API x-requests-remaining header."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{settings.odds_api_base_url}/sports",
                params={"apiKey": settings.odds_api_key},
            )
        remaining = int(resp.headers.get("x-requests-remaining", 0))
        used = int(resp.headers.get("x-requests-used", 0))
        total = remaining + used
        pct = (remaining / max(total, 1)) * 100
        if total == 0:
            return {
                "check_name": "odds_api_quota",
                "status": "pass",
                "actual": "unknown (no quota info)",
                "expected": "> 20% remaining",
                "message": "Could not determine API quota from response headers.",
            }
        if pct < 10:
            return {
                "check_name": "odds_api_quota",
                "status": "critical",
                "actual": f"{remaining}/{total} ({pct:.0f}%)",
                "expected": "> 20% remaining",
                "message": f"API quota critically low: {remaining}/{total} remaining ({pct:.0f}%).",
            }
        if pct < 20:
            return {
                "check_name": "odds_api_quota",
                "status": "warn",
                "actual": f"{remaining}/{total} ({pct:.0f}%)",
                "expected": "> 20% remaining",
                "message": f"API quota getting low: {remaining}/{total} remaining ({pct:.0f}%).",
            }
        return {
            "check_name": "odds_api_quota",
            "status": "pass",
            "actual": f"{remaining}/{total} ({pct:.0f}%)",
            "expected": "> 20% remaining",
            "message": f"API quota healthy: {remaining}/{total} remaining ({pct:.0f}%).",
        }
    except Exception as e:
        return {
            "check_name": "odds_api_quota",
            "status": "critical",
            "actual": f"error: {e}",
            "expected": "API reachable",
            "message": f"Failed to check Odds API quota: {e}",
        }


async def check_db_health(db: Session) -> dict[str, Any]:
    """DB reachable, raw_odds count not decreasing unexpectedly."""
    global _last_raw_odds_count
    try:
        db.execute(text("SELECT 1"))
        count = db.query(RawOdds).count()
        if _last_raw_odds_count is not None and count < _last_raw_odds_count * 0.5:
            result = {
                "check_name": "db_health",
                "status": "critical",
                "actual": f"{count} rows (was {_last_raw_odds_count})",
                "expected": "raw_odds count stable or increasing",
                "message": f"Raw odds count dropped from {_last_raw_odds_count} to {count} — possible data loss.",
            }
        else:
            result = {
                "check_name": "db_health",
                "status": "pass",
                "actual": f"{count} rows",
                "expected": "DB reachable, count stable",
                "message": f"Database reachable with {count} raw odds rows.",
            }
        _last_raw_odds_count = count
        return result
    except Exception as e:
        return {
            "check_name": "db_health",
            "status": "critical",
            "actual": f"error: {e}",
            "expected": "DB reachable",
            "message": f"Database health check failed: {e}",
        }


async def check_grading_pipeline(db: Session) -> dict[str, Any]:
    """PickOutcome count non-decreasing, alert if no new in 48h."""
    global _last_pick_outcome_count, _last_pick_outcome_time
    try:
        count = db.query(PickOutcome).count()
        newest = db.query(func.max(PickOutcome.created_at)).scalar()

        if _last_pick_outcome_count is not None and count < _last_pick_outcome_count:
            result = {
                "check_name": "grading_pipeline",
                "status": "critical",
                "actual": f"{count} outcomes (was {_last_pick_outcome_count})",
                "expected": "PickOutcome count non-decreasing",
                "message": f"PickOutcome count decreased from {_last_pick_outcome_count} to {count}.",
            }
        elif newest and (datetime.now(timezone.utc) - newest).total_seconds() > 48 * 3600:
            result = {
                "check_name": "grading_pipeline",
                "status": "warn",
                "actual": f"last graded {newest.isoformat()}",
                "expected": "graded < 48h ago",
                "message": f"No new PickOutcomes in over 48 hours (last: {newest.isoformat()}).",
            }
        else:
            result = {
                "check_name": "grading_pipeline",
                "status": "pass",
                "actual": f"{count} outcomes, last: {newest.isoformat() if newest else 'never'}",
                "expected": "count non-decreasing, graded < 48h ago",
                "message": f"Grading pipeline healthy: {count} PickOutcomes.",
            }

        _last_pick_outcome_count = count
        _last_pick_outcome_time = newest
        return result
    except Exception as e:
        return {
            "check_name": "grading_pipeline",
            "status": "critical",
            "actual": f"error: {e}",
            "expected": "PickOutcome table queryable",
            "message": f"Grading pipeline check failed: {e}",
        }


# ---------------------------------------------------------------------------
# Email alerting via Resend
# ---------------------------------------------------------------------------

def send_alert(failing: list[dict]) -> None:
    """Send one email per failure batch via Resend."""
    import resend

    resend.api_key = settings.resend_api_key

    subject = f"[DoubleDown ALERT] {len(failing)} health checks failing"
    body = "The following health checks are failing:\n\n"
    for check in failing:
        body += f"  ❌ {check['check_name']}\n"
        body += f"     Actual: {check['actual']}\n"
        body += f"     Expected: {check['expected']}\n"
        body += f"     {check['message']}\n\n"

    try:
        resend.Emails.send({
            "from": "health@getdoubledown.com",
            "to": [settings.health_alert_email],
            "subject": subject,
            "text": body,
        })
        print(f"[health_sentinel] Alert sent: {subject}")
    except Exception as e:
        print(f"[health_sentinel] Failed to send alert: {e}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_health_checks() -> list[dict]:
    """Run all 9 checks, return list of results."""
    results: list[dict] = []
    db = SessionLocal()
    try:
        results.append(await check_odds_freshness(db))
        results.append(await check_games_board(db))
        results.append(await check_no_past_games(db))
        results.append(await check_valid_odds(db))
        results.append(await check_bookmaker_allowlist(db))
        results.append(await check_sport_keys(db))
        results.append(await check_odds_api_quota())
        results.append(await check_db_health(db))
        results.append(await check_grading_pipeline(db))
    finally:
        db.close()
    return results


async def run_health_sentinel() -> None:
    """Wrapper for APScheduler: run checks, email on failure."""
    print("[health_sentinel] Running 6-hourly health checks...")
    try:
        results = await run_health_checks()
        failing = [r for r in results if r["status"] in ("warn", "critical")]
        if failing:
            send_alert(failing)
            print(f"[health_sentinel] {len(failing)} check(s) failing — alert sent.")
        else:
            print(f"[health_sentinel] All {len(results)} checks passed.")
    except Exception as e:
        print(f"[health_sentinel] Orchestrator error: {e}", exc_info=True)