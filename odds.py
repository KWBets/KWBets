from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ingestion.fetch_odds import fetch_and_save_odds, fetch_odds, load_latest_odds

router = APIRouter(prefix="/odds", tags=["odds"])


@router.get("/")
def get_odds(
    sport: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    bookmaker: Optional[str] = Query(None),
):
    df = load_latest_odds()
    if df.empty:
        raise HTTPException(status_code=404, detail="No odds data available. POST /odds/fetch to ingest.")

    if sport:
        df = df[df["sport"] == sport]
    if market:
        df = df[df["market"] == market]
    if bookmaker:
        df = df[df["bookmaker"] == bookmaker]

    return {"count": len(df), "odds": df.to_dict(orient="records")}


@router.post("/fetch")
def trigger_fetch():
    path = fetch_and_save_odds()
    if path is None:
        raise HTTPException(status_code=502, detail="Failed to fetch odds. Check ODDS_API_KEY.")
    df = load_latest_odds()
    return {"saved_to": str(path), "rows": len(df)}


@router.get("/sports")
def list_sports():
    df = load_latest_odds()
    if df.empty:
        return {"sports": []}
    return {"sports": sorted(df["sport"].unique().tolist())}


@router.get("/bookmakers")
def list_bookmakers():
    df = load_latest_odds()
    if df.empty:
        return {"bookmakers": []}
    return {"bookmakers": sorted(df["bookmaker"].unique().tolist())}
