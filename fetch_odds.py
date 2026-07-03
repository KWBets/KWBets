from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _fetch_sport_odds(client: httpx.Client, sport: str) -> list[dict]:
    url = f"{config.ODDS_API_BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": config.ODDS_REGIONS,
        "markets": ",".join(config.ODDS_MARKETS),
        "oddsFormat": "american",
    }
    response = client.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _flatten_event(event: dict) -> list[dict]:
    rows = []
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                rows.append(
                    {
                        "event_id": event["id"],
                        "sport": event.get("sport_key"),
                        "commence_time": event.get("commence_time"),
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "bookmaker": bookmaker["key"],
                        "market": market["key"],
                        "outcome": outcome["name"],
                        "price": outcome["price"],
                        "point": outcome.get("point"),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
    return rows


def fetch_odds() -> pd.DataFrame:
    if not config.ODDS_API_KEY:
        logger.warning("ODDS_API_KEY not set; returning empty DataFrame")
        return pd.DataFrame()

    all_rows: list[dict] = []
    with httpx.Client() as client:
        for sport in config.ODDS_SPORTS:
            sport = sport.strip()
            if not sport:
                continue
            try:
                events = _fetch_sport_odds(client, sport)
                for event in events:
                    all_rows.extend(_flatten_event(event))
                logger.info("Fetched %d events for %s", len(events), sport)
            except httpx.HTTPError as exc:
                logger.error("Failed to fetch odds for %s: %s", sport, exc)

    return pd.DataFrame(all_rows)


def save_odds(df: pd.DataFrame, output_dir: Path | None = None) -> Path | None:
    if df.empty:
        return None

    output_dir = output_dir or config.RAW_DATA_DIR
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"odds_{timestamp}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def fetch_and_save_odds() -> Path | None:
    df = fetch_odds()
    return save_odds(df)


def load_latest_odds(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or config.RAW_DATA_DIR
    files = sorted(raw_dir.glob("odds_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])
