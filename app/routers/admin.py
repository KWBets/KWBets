"""API router for admin endpoints — model progress, calibration, and grading stats.

Protected by ADMIN_API_KEY header. Returns 503 if no key configured, 403 if
key is wrong. All queries handle empty data gracefully (zero counts, empty
arrays, null for dates). Calibration buckets always return all 4 buckets
even when empty (sample_size=0, predicted_avg=null, actual_win_rate=null).
"""

from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import settings
from app.database import get_db
from app.models import PickOutcome, ValueBet, ModelRegistry, ReferralEvent, User
from app.schemas import AdminReferralEvent, AdminReferralStats

router = APIRouter()


async def verify_admin_key(x_admin_key: str = Header(...)):
    """Require a valid ADMIN_API_KEY header for all admin endpoints."""
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin API not configured")
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return True


@router.get("/admin/model-progress", tags=["Admin"])
async def get_model_progress(
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key),
):
    """Return model training progress, calibration, and grading stats.

    Calibration buckets show how well the model's predicted probabilities
    match actual outcomes. The 300-outcome threshold is required before
    model edge scores are shown to users.

    Requires X-Admin-Key header matching ADMIN_API_KEY env var.
    """
    now = datetime.now(timezone.utc)
    today = now.date()

    # ------------------------------------------------------------------
    # 1. Graded outcomes — total count (real PickOutcomes, not ModelRegistry)
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
    #    Always returns all 4 buckets even if empty (sample_size=0,
    #    predicted_avg=null, actual_win_rate=null).
    # ------------------------------------------------------------------
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
        last_run_graded = (
            db.query(func.count(PickOutcome.id))
            .filter(PickOutcome.created_at == last_run_at)
            .scalar()
        ) or 0

    # ------------------------------------------------------------------
    # 9. Model — active model registry entry + retrain_status from
    #    real PickOutcome labels (not ModelRegistry's stale training_samples)
    # ------------------------------------------------------------------
    active_model = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.is_active.is_(True))
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )

    # retrain_status is based on real graded PickOutcomes
    real_labels = graded_total  # PickOutcomes where actual_outcome IN ('won','lost')

    model_info = {
        "active_model_version": None,
        "last_retrained_at": None,
        "training_label_count": real_labels,
        "retrain_status": "no model — waiting for 300+ graded outcomes",
    }
    if active_model:
        model_info["active_model_version"] = active_model.model_version
        model_info["last_retrained_at"] = (
            active_model.training_end.isoformat() if active_model.training_end else None
        )
        model_info["training_label_count"] = real_labels

        if real_labels >= 200:
            model_info["retrain_status"] = (
                f"ready — model active with {real_labels} real labels"
            )
        elif real_labels > 0:
            model_info["retrain_status"] = (
                f"training — model active with {real_labels} real labels"
            )
        else:
            model_info["retrain_status"] = (
                "waiting for labels — retrains at 200+"
            )
    else:
        # No active model at all
        remaining_for_threshold = max(0, THRESHOLD - real_labels)
        model_info["retrain_status"] = (
            f"no model — waiting for {remaining_for_threshold} more graded outcomes"
        )

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


@router.get("/admin/referrals", response_model=AdminReferralStats, tags=["Admin"])
async def get_admin_referrals(
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key),
):
    """Admin view: referral stats, recent events, top referrers, flagged events.

    Requires X-Admin-Key header matching ADMIN_API_KEY env var.
    """
    now = datetime.now(timezone.utc)

    # Total referrals
    total_referrals = (
        db.query(func.count(ReferralEvent.id)).scalar()
    ) or 0

    # Active referrers (distinct referrer_ids with >= 1 event)
    active_referrers = (
        db.query(func.count(func.distinct(ReferralEvent.referrer_id)))
        .scalar()
    ) or 0

    # Recent events (last 20)
    recent_raw = (
        db.query(ReferralEvent)
        .order_by(ReferralEvent.created_at.desc())
        .limit(20)
        .all()
    )
    recent_events = [
        AdminReferralEvent(
            referrer_id=e.referrer_id,
            referred_id=e.referred_id,
            status=e.status,
            flag_reason=e.flag_reason,
            created_at=e.created_at,
            rewarded_at=e.rewarded_at,
        )
        for e in recent_raw
    ]

    # Top referrers (top 10 by rewarded count)
    top_raw = (
        db.query(
            ReferralEvent.referrer_id,
            func.count(ReferralEvent.id).label("cnt"),
        )
        .filter(ReferralEvent.status == "rewarded")
        .group_by(ReferralEvent.referrer_id)
        .order_by(func.count(ReferralEvent.id).desc())
        .limit(10)
        .all()
    )
    top_referrers = [
        {"referrer_id": row.referrer_id, "rewarded_count": row.cnt}
        for row in top_raw
    ]

    # Flagged events
    flagged_raw = (
        db.query(ReferralEvent)
        .filter(ReferralEvent.status == "flagged")
        .order_by(ReferralEvent.created_at.desc())
        .all()
    )
    flagged_events = [
        AdminReferralEvent(
            referrer_id=e.referrer_id,
            referred_id=e.referred_id,
            status=e.status,
            flag_reason=e.flag_reason,
            created_at=e.created_at,
            rewarded_at=e.rewarded_at,
        )
        for e in flagged_raw
    ]

    return AdminReferralStats(
        total_referrals=total_referrals,
        active_referrers=active_referrers,
        recent_events=recent_events,
        top_referrers=top_referrers,
        flagged_events=flagged_events,
    )