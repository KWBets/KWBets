"""API router for admin endpoints — model progress, calibration, and grading stats."""

from datetime import datetime, timezone, timedelta, date
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from app.database import get_db
from app.models import PickOutcome, ValueBet, ModelRegistry

router = APIRouter()


@router.get("/admin/model-progress", tags=["Admin"])
async def get_model_progress(db: Session = Depends(get_db)):
    """Return model training progress, calibration, and grading stats.

    Calibration buckets show how well the model's predicted probabilities
    match actual outcomes. The 300-outcome threshold is required before
    model edge scores are shown to users.
    """
    now = datetime.now(timezone.utc)
    today = now.date()

    # ------------------------------------------------------------------
    # 1. Graded outcomes — total count
    # ------------------------------------------------------------------
    graded_total = (
        db.query(func.count(PickOutcome.id))
        .filter(PickOutcome.actual_outcome.in_(["won", "lost"]))
        .scalar()
    ) or 0

    # ------------------------------------------------------------------
    # 2. Graded outcomes — by sport
    # ------------------------------------------------------------------
    sport_rows = (
        db.query(
            ValueBet.sport_key,
            func.count(PickOutcome.id).label("cnt"),
        )
        .join(PickOutcome, PickOutcome.value_bet_id == ValueBet.id)
        .filter(PickOutcome.actual_outcome.in_(["won", "lost"]))
        .group_by(ValueBet.sport_key)
        .order_by(func.count(PickOutcome.id).desc())
        .all()
    )
    by_sport = {row.sport_key: row.cnt for row in sport_rows}

    # ------------------------------------------------------------------
    # 3. Graded outcomes — by day (last 30 days)
    # ------------------------------------------------------------------
    thirty_days_ago = now - timedelta(days=30)
    day_rows = (
        db.query(
            func.date(PickOutcome.created_at).label("day"),
            func.count(PickOutcome.id).label("cnt"),
        )
        .filter(
            PickOutcome.actual_outcome.in_(["won", "lost"]),
            PickOutcome.created_at >= thirty_days_ago,
        )
        .group_by(func.date(PickOutcome.created_at))
        .order_by(func.date(PickOutcome.created_at))
        .all()
    )
    by_day = [{"date": str(row.day), "count": row.cnt} for row in day_rows]

    # ------------------------------------------------------------------
    # 4. Progress — daily rate (last 7 days)
    # ------------------------------------------------------------------
    seven_days_ago = now - timedelta(days=7)
    graded_last_7 = (
        db.query(func.count(PickOutcome.id))
        .filter(
            PickOutcome.actual_outcome.in_(["won", "lost"]),
            PickOutcome.created_at >= seven_days_ago,
        )
        .scalar()
    ) or 0
    daily_rate = round(graded_last_7 / 7.0, 1)

    # ------------------------------------------------------------------
    # 5. Progress — estimated threshold date
    # ------------------------------------------------------------------
    THRESHOLD = 300
    pct = round((graded_total / THRESHOLD) * 100, 1) if THRESHOLD > 0 else 0.0
    remaining = max(0, THRESHOLD - graded_total)
    if daily_rate > 0:
        days_to_threshold = remaining / daily_rate
        est_threshold_date = (today + timedelta(days=int(days_to_threshold))).isoformat()
    else:
        est_threshold_date = None  # no grading activity — can't estimate

    # ------------------------------------------------------------------
    # 6. Calibration buckets — model_probability vs actual win rate
    # ------------------------------------------------------------------
    # Get all graded PickOutcomes with model_probability
    calibration_rows = (
        db.query(
            PickOutcome.model_probability,
            PickOutcome.actual_outcome,
        )
        .filter(
            PickOutcome.actual_outcome.in_(["won", "lost"]),
            PickOutcome.model_probability.isnot(None),
        )
        .all()
    )

    # Bucket definitions
    buckets = [
        {"label": "50-60%", "min_p": 0.50, "max_p": 0.60},
        {"label": "60-70%", "min_p": 0.60, "max_p": 0.70},
        {"label": "70-80%", "min_p": 0.70, "max_p": 0.80},
        {"label": "80%+", "min_p": 0.80, "max_p": 1.01},
    ]

    calibration = []
    for bucket in buckets:
        samples = [
            r for r in calibration_rows
            if bucket["min_p"] <= r.model_probability < bucket["max_p"]
        ]
        sample_size = len(samples)
        if sample_size == 0:
            continue  # skip empty buckets

        predicted_avg = round(
            sum(r.model_probability for r in samples) / sample_size, 4
        )
        won_count = sum(1 for r in samples if r.actual_outcome == "won")
        actual_win_rate = round(won_count / sample_size, 4)

        calibration.append({
            "bucket": bucket["label"],
            "predicted_avg": predicted_avg,
            "actual_win_rate": actual_win_rate,
            "sample_size": sample_size,
        })

    # ------------------------------------------------------------------
    # 7. Pending — value bets past their commence_time but not yet graded
    # ------------------------------------------------------------------
    pending_count = (
        db.query(func.count(ValueBet.id))
        .filter(
            ValueBet.status == "pending",
            ValueBet.commence_time < now,
        )
        .scalar()
    ) or 0

    # ------------------------------------------------------------------
    # 8. Grading runs — last run info
    # ------------------------------------------------------------------
    last_run_at = (
        db.query(func.max(PickOutcome.created_at))
        .scalar()
    )
    last_run_graded = 0
    if last_run_at is not None:
        # Count how many PickOutcomes share that exact max timestamp
        last_run_graded = (
            db.query(func.count(PickOutcome.id))
            .filter(PickOutcome.created_at == last_run_at)
            .scalar()
        ) or 0

    # ------------------------------------------------------------------
    # 9. Model — active model registry entry
    # ------------------------------------------------------------------
    active_model = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.is_active.is_(True))
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )

    model_info = {
        "active_model_version": None,
        "last_retrained_at": None,
        "training_label_count": 0,
        "retrain_status": "no model trained yet",
    }
    if active_model:
        model_info["active_model_version"] = active_model.model_version
        model_info["last_retrained_at"] = (
            active_model.training_end.isoformat() if active_model.training_end else None
        )
        model_info["training_label_count"] = active_model.training_samples or 0
        labels = active_model.training_samples or 0
        if labels >= 200:
            model_info["retrain_status"] = f"ready — model active with {labels} labels"
        elif labels > 0:
            model_info["retrain_status"] = (
                f"training — model active with {labels} labels"
            )
        else:
            model_info["retrain_status"] = "initializing — no training data yet"

    # ------------------------------------------------------------------
    # Assemble response
    # ------------------------------------------------------------------
    return {
        "graded_outcomes": {
            "total": graded_total,
            "by_sport": by_sport,
            "by_day": by_day,
        },
        "progress": {
            "threshold": THRESHOLD,
            "pct": pct,
            "daily_rate": daily_rate,
            "est_threshold_date": est_threshold_date,
        },
        "calibration": calibration,
        "pending": pending_count,
        "grading_runs": {
            "last_run_at": last_run_at.isoformat() if last_run_at else None,
            "last_run_graded": last_run_graded,
        },
        "model": model_info,
    }