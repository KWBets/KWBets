"""Public track record endpoint — no authentication.

Reuses the same graded-outcome definition as /admin/model-progress:
PickOutcome rows where actual_outcome is 'won' or 'lost'. Do not fork
a second definition of "graded" anywhere else.
"""

from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models import PickOutcome, ValueBet, ModelRegistry

router = APIRouter()

THRESHOLD = 300

MODEL_ATTRIBUTION = {
    "baseline-v0": "Market consensus baseline",
}


@router.get("/track-record", tags=["Public"])
async def get_track_record(db: Session = Depends(get_db)):
    """Public, unauthenticated track record. Every graded outcome, permanently."""
    now = datetime.now(timezone.utc)
    today = now.date()

    graded_rows = (
        db.query(PickOutcome.actual_outcome)
        .filter(PickOutcome.actual_outcome.in_(["won", "lost"]))
        .all()
    )
    graded_total = len(graded_rows)
    wins = sum(1 for r in graded_rows if r.actual_outcome == "won")
    losses = graded_total - wins

    pending = (
        db.query(func.count(ValueBet.id))
        .filter(
            ValueBet.status == "pending",
            ValueBet.commence_time < now,
        )
        .scalar()
    ) or 0

    first_graded_at = (
        db.query(func.min(PickOutcome.created_at))
        .filter(PickOutcome.actual_outcome.in_(["won", "lost"]))
        .scalar()
    )
    last_graded_at = (
        db.query(func.max(PickOutcome.created_at))
        .filter(PickOutcome.actual_outcome.in_(["won", "lost"]))
        .scalar()
    )

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

    bucket_defs = [
        {"label": "50-60%", "min_p": 0.50, "max_p": 0.60},
        {"label": "60-70%", "min_p": 0.60, "max_p": 0.70},
        {"label": "70-80%", "min_p": 0.70, "max_p": 0.80},
        {"label": "80%+",   "min_p": 0.80, "max_p": 1.01},
    ]

    calibration = []
    for b in bucket_defs:
        samples = [
            r for r in calibration_rows
            if b["min_p"] <= r.model_probability < b["max_p"]
        ]
        n = len(samples)
        if n == 0:
            calibration.append({
                "bucket": b["label"],
                "predicted_avg": None,
                "actual_win_rate": None,
                "sample_size": 0,
            })
        else:
            predicted_avg = round(
                sum(r.model_probability for r in samples) / n, 4
            )
            won = sum(1 for r in samples if r.actual_outcome == "won")
            calibration.append({
                "bucket": b["label"],
                "predicted_avg": predicted_avg,
                "actual_win_rate": round(won / n, 4),
                "sample_size": n,
            })

    version_rows = (
        db.query(
            ValueBet.model_version,
            func.count(PickOutcome.id).label("cnt"),
        )
        .join(PickOutcome, PickOutcome.value_bet_id == ValueBet.id)
        .filter(PickOutcome.actual_outcome.in_(["won", "lost"]))
        .group_by(ValueBet.model_version)
        .order_by(func.count(PickOutcome.id).desc())
        .all()
    )
    model_versions = [
        {
            "model_version": row.model_version,
            "description": MODEL_ATTRIBUTION.get(
                row.model_version, "Trained model"
            ),
            "graded_count": row.cnt,
        }
        for row in version_rows
    ]

    active_model = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.is_active.is_(True))
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )

    methodology_note = (
        "Phase 1: proving the grading pipeline before any model claims an edge. "
        "Picks are currently generated from de-vigged market consensus and are "
        "published on both sides of each game, so one wins and one loses by "
        "construction. Aggregate win rate is therefore meaningless as a measure "
        "of skill. Calibration — whether a stated probability matches the real "
        "world frequency — is the metric that matters, and it is shown above. "
        "Every graded outcome is included here permanently, with no filtering."
    )

    return {
        "summary": {
            "graded_total": graded_total,
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "threshold": THRESHOLD,
            "first_graded_at": (
                first_graded_at.isoformat() if first_graded_at else None
            ),
            "last_graded_at": (
                last_graded_at.isoformat() if last_graded_at else None
            ),
        },
        "calibration": calibration,
        "by_sport": by_sport,
        "by_day": by_day,
        "model_versions": model_versions,
        "active_model_version": (
            active_model.model_version if active_model else None
        ),
        "methodology_note": methodology_note,
    }
