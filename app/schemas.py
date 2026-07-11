"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    environment: str = "development"
    model_active: bool = False
    odds_last_fetch: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------
class Outcome(BaseModel):
    name: str
    price: float
    point: Optional[float] = None


class Market(BaseModel):
    key: str
    outcomes: list[Outcome]


class Bookmaker(BaseModel):
    key: str
    title: str
    last_update: datetime
    markets: list[Market]


class OddsResponse(BaseModel):
    id: str
    sport_key: str
    sport_title: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmakers: list[Bookmaker]


class OddsListResponse(BaseModel):
    count: int
    sports: list[str]
    odds: list[OddsResponse]


# ---------------------------------------------------------------------------
# Predictions / Value Bets
# ---------------------------------------------------------------------------
class PredictionFactor(BaseModel):
    factor: str
    contribution: float


class ValueBetResponse(BaseModel):
    id: int
    event_id: str
    sport: str
    sport_key: str
    home_team: str
    away_team: str
    commence_time: datetime
    team: str
    market_type: str
    pick_label: str
    odds: float
    model_probability: float
    implied_probability: float
    edge_percentage: float
    expected_value: float
    confidence_tier: str
    confidence_score: float
    reasoning_factors: Optional[dict[str, Any]] = None
    model_version: str
    best_bookmaker: Optional[str] = None
    best_odds: Optional[dict] = None  # {"price": 2.5, "bookmaker": "DraftKings"}
    consensus_implied_prob: Optional[float] = None  # median implied prob across books
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionsListResponse(BaseModel):
    count: int
    predictions: list[ValueBetResponse]


# ---------------------------------------------------------------------------
# Parlays
# ---------------------------------------------------------------------------
class ParlayLeg(BaseModel):
    event_id: str
    sport: str
    team: str
    market_type: str
    pick_label: str
    odds: float
    model_probability: float
    edge_percentage: float


class ParlaySuggestion(BaseModel):
    legs: list[ParlayLeg]
    combined_odds: float
    combined_implied_prob: float
    combined_model_prob: float
    combined_edge: float
    confidence_tier: str
    correlation_warning: Optional[str] = None


class ParlayBuildRequest(BaseModel):
    leg_ids: list[int] = Field(..., min_length=2, max_length=8)


class ParlayBuildResponse(BaseModel):
    legs: list[ValueBetResponse]
    combined_odds: float
    combined_implied_prob: float
    combined_edge: float
    disclaimer: str


# ---------------------------------------------------------------------------
# Games (model-independent feed from raw_odds)
# ---------------------------------------------------------------------------
class GameOutcome(BaseModel):
    name: str  # team name or "Over"/"Under"/"Draw"
    price: float  # best available decimal odds across all books
    best_odds_bookmaker: Optional[str] = None  # which book offers this price


class GameEvent(BaseModel):
    event_id: str  # raw event hash (first UUID segment of RawOdds.id)
    sport: str
    sport_key: str
    home_team: str
    away_team: str
    commence_time: datetime
    outcomes: list[GameOutcome]  # one per unique outcome_name for h2h
    consensus_implied_prob: Optional[float] = None


class GamesListResponse(BaseModel):
    count: int
    games: list[GameEvent]


# ---------------------------------------------------------------------------
# Sports
# ---------------------------------------------------------------------------
class SportInfo(BaseModel):
    key: str
    title: str
    active: bool
    has_odds: bool


class SportsListResponse(BaseModel):
    count: int
    sports: list[SportInfo]


# ---------------------------------------------------------------------------
# Props
# ---------------------------------------------------------------------------
class PropValueBet(BaseModel):
    id: int
    sport: str
    player_name: str
    team: str
    market_type: str
    line: float
    odds: float
    model_probability: float
    implied_probability: float
    edge_percentage: float
    confidence_tier: str


class PropsListResponse(BaseModel):
    count: int
    props: list[PropValueBet]


# ---------------------------------------------------------------------------
# Fetch / Sync
# ---------------------------------------------------------------------------
class FetchOddsResponse(BaseModel):
    status: str
    message: str
    sports_fetched: list[str]
    total_odds_stored: int
    fetch_duration_seconds: float