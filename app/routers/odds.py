"""API router for health checks and odds operations."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from app.database import get_db
from app.models import RawOdds, ValueBet, ModelRegistry
from app.schemas import (
    HealthResponse,
    FetchOddsResponse,
    OddsResponse,
    OddsListResponse,
    SportsListResponse,
    SportInfo,
)
from app.odds_ingestion import run_odds_fetch
from app.scheduler import trigger_odds_fetch_now
from app.config import settings
from app.health_sentinel import run_health_checks

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    # Check if DB is alive
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    # Check if model is active
    active_model = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.is_active.is_(True))
        .first()
    )

    # Check last odds fetch
    last_odds = (
        db.query(func.max(RawOdds.fetched_at)).scalar()
    )

    return HealthResponse(
        status="ok" if db_ok else "db_error",
        version="1.0.0",
        environment=settings.environment,
        model_active=active_model is not None,
        odds_last_fetch=last_odds,
    )


@router.get("/health/detailed", tags=["System"])
async def health_detailed():
    """Run all 9 health checks and return detailed results as JSON."""
    from datetime import datetime, timezone
    results = await run_health_checks()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(results),
            "passing": sum(1 for r in results if r["status"] == "pass"),
            "warnings": sum(1 for r in results if r["status"] == "warn"),
            "critical": sum(1 for r in results if r["status"] == "critical"),
        },
        "checks": results,
    }


@router.post("/odds/fetch", response_model=FetchOddsResponse, tags=["Odds"])
async def fetch_odds():
    """Manually trigger odds fetch from The Odds API."""
    result = await trigger_odds_fetch_now()
    return FetchOddsResponse(**result)


@router.get("/odds", response_model=list[OddsResponse], tags=["Odds"])
async def get_odds(
    sport: str = Query(None, description="Filter by sport key"),
    market: str = Query(None, description="Filter by market type (h2h, spreads, totals)"),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get latest stored odds, optionally filtered by sport and/or market."""
    # Get distinct events with their latest odds
    # For simplicity, return aggregated view
    subq = (
        db.query(
            RawOdds.sport_key,
            RawOdds.commence_time,
            RawOdds.home_team,
            RawOdds.away_team,
            func.max(RawOdds.fetched_at).label("last_fetch"),
        )
        .group_by(
            RawOdds.sport_key,
            RawOdds.commence_time,
            RawOdds.home_team,
            RawOdds.away_team,
        )
    )

    if sport:
        subq = subq.filter(RawOdds.sport_key == sport)

    subq = subq.subquery()

    # Get raw odds for these events
    query = (
        db.query(RawOdds)
        .join(
            subq,
            (RawOdds.sport_key == subq.c.sport_key)
            & (RawOdds.commence_time == subq.c.commence_time)
            & (RawOdds.home_team == subq.c.home_team)
            & (RawOdds.away_team == subq.c.away_team)
            & (RawOdds.fetched_at == subq.c.last_fetch),
        )
        .order_by(RawOdds.commence_time)
        .limit(limit)
    )

    if market:
        query = query.filter(RawOdds.market_key == market)

    rows = query.all()

    # Group into response format
    from collections import defaultdict

    events = defaultdict(lambda: {"bookmakers": defaultdict(lambda: {"markets": defaultdict(list)})})

    for r in rows:
        event_key = f"{r.sport_key}|{r.commence_time}|{r.home_team}|{r.away_team}"
        bm_key = f"{r.bookmaker_key}|{r.bookmaker_title}"

        if market and r.market_key != market:
            continue

        outcome = {
            "name": r.outcome_name,
            "price": r.outcome_price,
        }
        if r.outcome_point is not None:
            outcome["point"] = r.outcome_point

        events[event_key]["id"] = r.id.split("_")[0] if "_" in r.id else r.id
        events[event_key]["sport_key"] = r.sport_key
        events[event_key]["sport_title"] = r.sport
        events[event_key]["commence_time"] = r.commence_time
        events[event_key]["home_team"] = r.home_team
        events[event_key]["away_team"] = r.away_team
        events[event_key]["bookmakers"][bm_key]["title"] = r.bookmaker_title
        events[event_key]["bookmakers"][bm_key]["key"] = r.bookmaker_key
        events[event_key]["bookmakers"][bm_key]["last_update"] = r.last_update
        events[event_key]["bookmakers"][bm_key]["markets"][r.market_key].append(outcome)

    # Convert to response — only process entries that have event data
    result = []
    for event_key, ev in events.items():
        if "id" not in ev:
            continue
        bms = []
        for bm_key, bm in ev["bookmakers"].items():
            market_list = [
                {"key": mk, "outcomes": outs}
                for mk, outs in bm["markets"].items()
            ]
            bms.append({
                "key": bm["key"],
                "title": bm["title"],
                "last_update": bm["last_update"],
                "markets": market_list,
            })
        result.append(OddsResponse(
            id=ev["id"],
            sport_key=ev["sport_key"],
            sport_title=ev["sport_title"],
            commence_time=ev["commence_time"],
            home_team=ev["home_team"],
            away_team=ev["away_team"],
            bookmakers=bms,
        ))

    return result


@router.get("/odds/sports", response_model=SportsListResponse, tags=["Odds"])
async def get_sports(db: Session = Depends(get_db)):
    """Get list of sports with active odds in the database."""
    results = (
        db.query(
            RawOdds.sport_key,
            RawOdds.sport,
            func.count(func.distinct(RawOdds.id)).label("odds_count"),
        )
        .group_by(RawOdds.sport_key, RawOdds.sport)
        .order_by(RawOdds.sport)
        .all()
    )

    # Also get all supported sports to show inactive ones
    seen = set()
    sports = []
    for r in results:
        seen.add(r.sport_key)
        sports.append(SportInfo(
            key=r.sport_key,
            title=r.sport,
            active=True,
            has_odds=r.odds_count > 0,
        ))

    # Add supported sports that have no data yet
    for sk in settings.supported_sports:
        if sk not in seen:
            from app.odds_ingestion import _sport_title
            sports.append(SportInfo(
                key=sk,
                title=_sport_title(sk),
                active=True,
                has_odds=False,
            ))

    return SportsListResponse(count=len(sports), sports=sports)