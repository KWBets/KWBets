"""SQLAlchemy ORM models for DoubleDown AI."""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, JSON, ForeignKey, Enum as SAEnum
)
from sqlalchemy.orm import relationship
from app.database import Base
import enum


class PredictionStatus(str, enum.Enum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    CANCELLED = "cancelled"


class ConfidenceTier(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Raw Odds — as fetched from The Odds API
# ---------------------------------------------------------------------------
class RawOdds(Base):
    """Raw odds data ingested from The Odds API."""

    __tablename__ = "raw_odds"

    id = Column(String, primary_key=True)  # composite: sport_event_market_outcome
    sport = Column(String, index=True, nullable=False)
    sport_key = Column(String, index=True, nullable=False)
    commence_time = Column(DateTime, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    bookmaker_key = Column(String, nullable=False)
    bookmaker_title = Column(String, nullable=False)
    market_key = Column(String, nullable=False)  # h2h, spreads, totals
    outcome_name = Column(String, nullable=False)
    outcome_price = Column(Float, nullable=False)
    outcome_point = Column(Float, nullable=True)  # for spreads/totals
    odds_timestamp = Column(DateTime, nullable=False)
    fetched_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    last_update = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = ()


# ---------------------------------------------------------------------------
# ProcessedFeatures — engineered features for model training
# ---------------------------------------------------------------------------
class ProcessedFeatures(Base):
    """Feature-engineered data ready for model consumption."""

    __tablename__ = "processed_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String, index=True, nullable=False)
    sport_key = Column(String, nullable=False)
    event_id = Column(String, index=True, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    commence_time = Column(DateTime, nullable=False)
    market_type = Column(String, nullable=False)

    # Implied probabilities from sportsbooks
    home_implied_prob = Column(Float, nullable=True)
    away_implied_prob = Column(Float, nullable=True)
    draw_implied_prob = Column(Float, nullable=True)

    # Market consensus
    home_odds_avg = Column(Float, nullable=True)
    away_odds_avg = Column(Float, nullable=True)
    draw_odds_avg = Column(Float, nullable=True)

    # Spread / total info
    home_spread = Column(Float, nullable=True)
    away_spread = Column(Float, nullable=True)
    over_line = Column(Float, nullable=True)
    under_line = Column(Float, nullable=True)

    # Derived features
    odds_displayed_count = Column(Integer, nullable=True)  # number of bookmakers
    home_odds_std = Column(Float, nullable=True)
    away_odds_std = Column(Float, nullable=True)
    home_implied_edge = Column(Float, nullable=True)  # over round removed
    away_implied_edge = Column(Float, nullable=True)

    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# ModelPrediction — ML model output
# ---------------------------------------------------------------------------
class ModelPrediction(Base):
    """Predictions output by the ML model."""

    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, index=True, nullable=False)
    sport = Column(String, index=True, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    commence_time = Column(DateTime, nullable=False)
    market_type = Column(String, nullable=False)  # h2h, spread, total

    # Model output
    home_win_probability = Column(Float, nullable=True)
    away_win_probability = Column(Float, nullable=True)
    draw_probability = Column(Float, nullable=True)
    predicted_spread = Column(Float, nullable=True)
    predicted_total = Column(Float, nullable=True)

    # Model metadata
    model_version = Column(String, nullable=False)
    model_run_id = Column(String, nullable=False)
    feature_timestamp = Column(DateTime, nullable=False)

    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# ValueBet — actionable value bet derived from model + odds comparison
# ---------------------------------------------------------------------------
class ValueBet(Base):
    """Actionable value bets — where model probability exceeds implied probability."""

    __tablename__ = "value_bets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, index=True, nullable=False)
    sport = Column(String, index=True, nullable=False)
    sport_key = Column(String, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    commence_time = Column(DateTime, nullable=False)

    # The pick
    team = Column(String, nullable=False)  # which team/outcome to bet on
    market_type = Column(String, nullable=False)  # h2h, spread, total
    pick_label = Column(String, nullable=False)  # e.g. "Kansas City Chiefs -3.5" or "Over 47.5"
    odds = Column(Float, nullable=False)  # decimal odds offered

    # Value metrics
    model_probability = Column(Float, nullable=False)
    implied_probability = Column(Float, nullable=False)
    edge_percentage = Column(Float, nullable=False)  # (model_prob - implied_prob) * 100
    expected_value = Column(Float, nullable=False)  # edge * odds in decimal form

    # Confidence
    confidence_tier = Column(String, nullable=False)  # high / medium / low
    confidence_score = Column(Float, nullable=False)  # 0.0 - 1.0

    # Reasoning
    reasoning_factors = Column(JSON, nullable=True)  # dict of factor -> contribution
    model_version = Column(String, nullable=False)

    # Betting metadata
    best_bookmaker = Column(String, nullable=True)
    sportsbook_url = Column(String, nullable=True)
    is_live = Column(Boolean, default=False)

    # Outcome tracking
    status = Column(
        String,
        nullable=False,
        default=PredictionStatus.PENDING.value,
    )
    actual_result = Column(String, nullable=True)

    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# UserAlert — configurable value bet alerts
# ---------------------------------------------------------------------------
class UserAlert(Base):
    """User-defined alert thresholds for value bets."""

    __tablename__ = "user_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    sport = Column(String, nullable=True)  # None = all sports
    min_edge = Column(Float, nullable=False, default=2.0)  # minimum edge %
    min_confidence = Column(String, nullable=False, default=ConfidenceTier.MEDIUM.value)
    markets = Column(JSON, nullable=True)  # list of market types to alert on
    is_active = Column(Boolean, default=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    last_triggered_at = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# ModelRegistry — tracks trained model artifacts
# ---------------------------------------------------------------------------
class ModelRegistry(Base):
    """Registry of trained model versions and their performance."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False, unique=True)
    model_type = Column(String, nullable=False)  # xgboost, gradient_boosting, etc.
    model_path = Column(String, nullable=False)  # path to saved model file
    feature_set_version = Column(String, nullable=True)

    # Performance metrics
    roc_auc = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    log_loss = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)

    # Training metadata
    training_start = Column(DateTime, nullable=True)
    training_end = Column(DateTime, nullable=True)
    training_samples = Column(Integer, nullable=True)
    hyperparameters = Column(JSON, nullable=True)

    is_active = Column(Boolean, default=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# PickOutcome — tracks actual results of value bets for model feedback
# ---------------------------------------------------------------------------
class PickOutcome(Base):
    """Actual outcome of a value bet, used for model retraining feedback."""

    __tablename__ = "pick_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    value_bet_id = Column(Integer, ForeignKey("value_bets.id"), nullable=False)
    event_id = Column(String, index=True, nullable=False)

    # Prediction-time metadata (for retraining labels)
    model_probability = Column(Float, nullable=True)  # model's probability at prediction time
    implied_probability = Column(Float, nullable=True)  # market implied prob at prediction time
    market_type = Column(String, nullable=True)  # h2h, spreads, totals
    pick_team = Column(String, nullable=True)  # which team/outcome was picked

    # Results
    actual_outcome = Column(String, nullable=False)  # won, lost, push
    home_score = Column(Float, nullable=True)
    away_score = Column(Float, nullable=True)
    covered_spread = Column(Boolean, nullable=True)  # for spread bets
    over_hit = Column(Boolean, nullable=True)  # for totals

    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationship
    value_bet = relationship("ValueBet", backref="outcome")


# ---------------------------------------------------------------------------
# User — minimal user identity for referral tracking
# ---------------------------------------------------------------------------
class User(Base):
    """Minimal user identity, keyed by a client-generated UUID."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_uuid = Column(String, unique=True, nullable=False, index=True)
    referral_code = Column(String, unique=True, nullable=True, index=True)
    email = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)  # last known IP for fraud checks
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    last_active_at = Column(DateTime, nullable=True)

    # Relationships
    sent_referrals = relationship("Referral", foreign_keys="Referral.referrer_id", backref="referrer")
    received_referral = relationship("Referral", foreign_keys="Referral.referred_id", backref="referred", uselist=False)
    credits = relationship("ReferralCredit", backref="user")


# ---------------------------------------------------------------------------
# Referral — tracks referral relationships
# ---------------------------------------------------------------------------
class Referral(Base):
    """Tracks a referral: who referred whom, with fraud checks."""
    __tablename__ = "referrals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    referrer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    referred_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)  # one referral per user
    referral_code_used = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending, completed, flagged

    # Credit tracking
    referrer_credited = Column(Boolean, default=False)
    referred_credited = Column(Boolean, default=False)
    referrer_credit_days = Column(Integer, nullable=True)
    referred_credit_days = Column(Integer, nullable=True)

    # Fraud detection
    referrer_ip = Column(String, nullable=True)
    referred_ip = Column(String, nullable=True)

    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    completed_at = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# ReferralCredit — tracks free Pro time earned via referrals
# ---------------------------------------------------------------------------
class ReferralCredit(Base):
    """Free Pro subscription days earned through referrals."""
    __tablename__ = "referral_credits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount_days = Column(Integer, nullable=False)
    reason = Column(String, nullable=False)  # "referral_sent", "referral_received"
    related_referral_id = Column(Integer, ForeignKey("referrals.id"), nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )