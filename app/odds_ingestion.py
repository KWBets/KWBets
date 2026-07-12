"""Odds ingestion from The Odds API."""
import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import RawOdds, ProcessedFeatures


# Sport title mapping
SPORT_TITLES = {
    "americanfootball_nfl": "NFL",
    "americanfootball_ncaaf": "NCAA Football",
    "baseball_mlb": "MLB",
    "basketball_nba": "NBA",
    "basketball_ncaab": "NCAA Basketball",
    "icehockey_nhl": "NHL",
    "soccer_fifa_world_cup": "FIFA World Cup",
    "soccer_usa_mls": "MLS",
    "soccer_epl": "English Premier League",
    "soccer_uefa_champs_league": "UEFA Champions League",
    "soccer_la_liga": "La Liga",
    "soccer_bundesliga": "Bundesliga",
    "golf_pga_championship": "PGA Championship",
    "golf_us_open": "US Open",
    "tennis_atp_wimbledon": "Wimbledon (ATP)",
    "tennis_wta_wimbledon": "Wimbledon (WTA)",
}


def _sport_title(sport_key: str) -> str:
    return SPORT_TITLES.get(sport_key, sport_key.replace("_", " ").title())


def _implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def _remove_juice(home_prob: float, away_prob: float, draw_prob: Optional[float] = None) -> tuple[float, float, Optional[float]]:
    """Remove the bookmaker's overround (juice) from implied probabilities."""
    total = home_prob + away_prob + (draw_prob or 0.0)
    if total <= 0:
        return home_prob, away_prob, draw_prob
    return home_prob / total, away_prob / total, (draw_prob / total) if draw_prob is not None else None


async def fetch_odds_for_sport(
    client: httpx.AsyncClient,
    sport: str,
    api_key: str,
    regions: str = "us,us2",
    markets: str = "h2h,spreads,totals",
) -> list[dict]:
    """Fetch odds for a single sport from The Odds API."""
    url = f"{settings.odds_api_base_url}/sports/{sport}/odds/"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }
    try:
        resp = await client.get(url, params=params, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Sport not available
            return []
        print(f"[odds] HTTP error for {sport}: {e.response.status_code}")
        return []
    except Exception as e:
        print(f"[odds] Error fetching {sport}: {e}")
        return []


async def fetch_all_odds(api_key: Optional[str] = None) -> list[dict]:
    """Fetch odds for all supported sports. Returns list of raw event dicts.

    Excludes futures/outright markets: any sport key containing 'winner',
    'championship', or ending in '_futures' is skipped.

    Quota optimization: only fetches sports with upcoming games (active=True
    in the /sports endpoint). Dormant sports are skipped to conserve API quota.
    """
    key = api_key or settings.odds_api_key
    if not key or key == "your_odds_api_key_here":
        print("[odds] No valid ODDS_API_KEY configured — using demo/seed data")
        return []

    # Futures/outright exclusion patterns
    import re
    _futures_pattern = re.compile(r"(winner|championship|_futures)", re.IGNORECASE)

    # First, fetch active sports list (1 request) to skip dormant sports
    async with httpx.AsyncClient() as client:
        try:
            sports_resp = await client.get(
                f"{settings.odds_api_base_url}/sports",
                params={"apiKey": key},
                timeout=15.0,
            )
            if sports_resp.status_code == 200:
                all_sports = sports_resp.json()
                active_sports = {s["key"] for s in all_sports if s.get("active")}
                print(f"[odds] Active sports: {len(active_sports)} of {len(all_sports)} total")
            else:
                active_sports = set(settings.supported_sports)
                print(f"[odds] Could not fetch sports list (HTTP {sports_resp.status_code}), fetching all supported sports")
        except Exception as e:
            active_sports = set(settings.supported_sports)
            print(f"[odds] Error fetching sports list: {e}, falling back to all supported sports")

        all_events = []
        for sport in settings.supported_sports:
            # Skip futures/outrights markets
            if _futures_pattern.search(sport):
                print(f"[odds] Skipping futures market: {sport}")
                continue

            # Skip dormant sports (no upcoming games)
            if sport not in active_sports:
                print(f"[odds] Skipping dormant sport: {sport} (no upcoming events)")
                continue

            events = await fetch_odds_for_sport(client, sport, key)
            print(f"[odds] {sport}: fetched {len(events)} events")
            for ev in events:
                ev["_sport_key"] = sport
                ev["_sport_title"] = _sport_title(sport)
            all_events.extend(events)
            # Small delay between sports to avoid rate limits
            await asyncio.sleep(0.25)

    return all_events


async def store_odds(events: list[dict], db: Session) -> int:
    """Store raw odds events into the database. Returns count of rows stored."""
    count = 0
    now = datetime.now(timezone.utc)

    for event in events:
        sport_key = event.get("_sport_key", "")
        sport_title = event.get("_sport_title", "")
        commence_time_str = event.get("commence_time")
        commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00")) if commence_time_str else now
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        bookmakers = event.get("bookmakers", [])
        for bm in bookmakers:
            bm_key = bm.get("key", "")
            bm_title = bm.get("title", "")
            last_update_str = bm.get("last_update")
            last_update = datetime.fromisoformat(last_update_str.replace("Z", "+00:00")) if last_update_str else now

            markets = bm.get("markets", [])
            for market in markets:
                mk = market.get("key", "")
                outcomes = market.get("outcomes", [])
                for outcome in outcomes:
                    row_id = f"{event.get('id', '')}_{bm_key}_{mk}_{outcome.get('name', '')}"
                    row = RawOdds(
                        id=row_id,
                        sport=sport_title,
                        sport_key=sport_key,
                        commence_time=commence_time,
                        home_team=home_team,
                        away_team=away_team,
                        bookmaker_key=bm_key,
                        bookmaker_title=bm_title,
                        market_key=mk,
                        outcome_name=outcome.get("name", ""),
                        outcome_price=float(outcome.get("price", 0)),
                        outcome_point=float(outcome["point"]) if outcome.get("point") is not None else None,
                        odds_timestamp=last_update,
                        fetched_at=now,
                        last_update=now,
                    )
                    db.merge(row)  # upsert by primary key
                    count += 1

    db.commit()
    return count


async def build_processed_features(db: Session) -> int:
    """Aggregate raw odds into processed features for model training.

    Filters out futures/outrights: rows where market_key is 'outrights' or
    outcome_name is 'Field' are excluded.
    """
    from sqlalchemy import text

    # Clear existing features and rebuild
    db.execute(text("DELETE FROM processed_features"))

    # Group by event and market, compute averages
    rows = (
        db.query(RawOdds)
        .filter(RawOdds.market_key != "outrights")
        .filter(RawOdds.outcome_name != "Field")
        .all()
    )

    # Use pandas for aggregation
    if not rows:
        return 0

    data = []
    for r in rows:
        data.append({
            "sport": r.sport,
            "sport_key": r.sport_key,
            "event_id": r.id.split("_")[0] if "_" in r.id else r.id,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "commence_time": r.commence_time,
            "market_type": r.market_key,
            "bookmaker_key": r.bookmaker_key,
            "outcome_name": r.outcome_name,
            "outcome_price": r.outcome_price,
            "outcome_point": r.outcome_point,
        })

    df = pd.DataFrame(data)

    if df.empty:
        return 0

    now = datetime.now(timezone.utc)
    features_created = 0

    # For each unique event + market
    for (event_id, market_type), group in df.groupby(["event_id", "market_type"]):
        row = group.iloc[0]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Get prices for home/away/draw
        home_odds = group[group["outcome_name"] == home_team]["outcome_price"]
        away_odds = group[group["outcome_name"] == away_team]["outcome_price"]
        draw_mask = ~group["outcome_name"].isin([home_team, away_team])
        draw_odds = group[draw_mask]["outcome_price"] if draw_mask.any() else pd.Series(dtype=float)

        home_odds_avg = home_odds.mean() if not home_odds.empty else None
        away_odds_avg = away_odds.mean() if not away_odds.empty else None
        draw_odds_avg = draw_odds.mean() if not draw_odds.empty else None

        home_prob = _implied_prob(home_odds_avg) if home_odds_avg else None
        away_prob = _implied_prob(away_odds_avg) if away_odds_avg else None
        draw_prob = _implied_prob(draw_odds_avg) if draw_odds_avg else None

        # Remove juice
        if home_prob and away_prob:
            h, a, d = _remove_juice(home_prob, away_prob, draw_prob)
            home_implied, away_implied = h, a
            draw_implied = d
        else:
            home_implied = home_prob
            away_implied = away_prob
            draw_implied = draw_prob

        home_spread = None
        away_spread = None
        over_line = None
        under_line = None

        if market_type == "spreads":
            home_spread_data = group[group["outcome_name"] == home_team]
            away_spread_data = group[group["outcome_name"] == away_team]
            if not home_spread_data.empty:
                home_spread = home_spread_data["outcome_point"].iloc[0]
            if not away_spread_data.empty:
                away_spread = away_spread_data["outcome_point"].iloc[0]
        elif market_type == "totals":
            over_data = group[group["outcome_name"].str.lower().str.contains("over")]
            under_data = group[group["outcome_name"].str.lower().str.contains("under")]
            if not over_data.empty:
                over_line = over_data["outcome_point"].iloc[0]
            if not under_data.empty:
                under_line = under_data["outcome_point"].iloc[0]

        home_odds_std = home_odds.std() if len(home_odds) > 1 else None
        away_odds_std = away_odds.std() if len(away_odds) > 1 else None

        feature = ProcessedFeatures(
            sport=row["sport"],
            sport_key=row["sport_key"],
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            commence_time=row["commence_time"],
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
            odds_displayed_count=len(group["bookmaker_key"].unique()),
            home_odds_std=home_odds_std,
            away_odds_std=away_odds_std,
            created_at=now,
        )
        db.add(feature)
        features_created += 1

    db.commit()
    return features_created


async def run_odds_fetch(api_key: Optional[str] = None) -> dict:
    """Full pipeline: fetch odds → store → build features → batch predictions → EV pipeline.

    After every odds fetch, the ML predictions and value bets are regenerated
    so the /predictions endpoint always reflects the latest market data.
    """
    start = time.time()
    api_key = api_key or settings.odds_api_key
    print(f"[odds_fetch] Starting fetch cycle at {datetime.now(timezone.utc).isoformat()}...")

    if not api_key or api_key == "your_odds_api_key_here":
        print("[odds_fetch] No valid ODDS_API_KEY configured — skipping")
        return {
            "status": "skipped",
            "message": "No valid ODDS_API_KEY configured. Set it in .env to fetch live odds.",
            "sports_fetched": [],
            "total_odds_stored": 0,
            "fetch_duration_seconds": 0,
        }

    events = await fetch_all_odds(api_key)
    print(f"[odds_fetch] fetch_all_odds returned {len(events)} total events")

    # Initialize all counters before the try block so they're always defined
    # (the try/finally has no except — if an exception occurs, the finally
    # closes the DB, then the exception propagates, and the code below won't
    # run. But if the try completes, all counters must be available.)
    stored = 0
    features = 0
    predictions = 0
    value_bets = 0

    db = SessionLocal()
    try:
        # Step 1: Store raw odds
        stored = await store_odds(events, db)

        # Step 2: Build processed features from latest odds
        features = await build_processed_features(db)

        # Step 3: Run batch predictions through the active ML model
        from app.train import load_active_model, run_batch_predictions
        import uuid
        model = load_active_model(db)
        if model is not None:
            from app.models import ModelRegistry
            active = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()
            if active:
                run_id = uuid.uuid4().hex
                predictions = run_batch_predictions(db, model, active.model_version, run_id)
            else:
                predictions = 0
        else:
            predictions = 0

        # Step 4: Run EV pipeline to generate value bets
        from app.ev import run_ev_pipeline
        value_bets = run_ev_pipeline(db)

        # Step 5: Clear old pipeline data before inserting fresh (done inside each function)
        db.commit()
    finally:
        db.close()

    elapsed = time.time() - start
    sports_with_data = list(set(e.get("_sport_key", "") for e in events))
    print(f"[odds_fetch] Cycle complete: {len(sports_with_data)} sports, {stored} odds rows, {features} features, {value_bets} value bets in {elapsed:.1f}s")

    return {
        "status": "success",
        "message": f"Fetched odds for {len(sports_with_data)} sports, stored {stored} rows, built {features} features, {predictions} predictions, {value_bets} value bets.",
        "sports_fetched": sports_with_data,
        "total_odds_stored": stored,
        "total_predictions": predictions,
        "total_value_bets": value_bets,
        "fetch_duration_seconds": round(elapsed, 2),
    }