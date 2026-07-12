"""Admin endpoints for creator-tier management.

All endpoints require the X-Admin-Key header (same verify_admin_key
pattern as the other admin routes).

Confirmation model: earnings sit "pending" for 30 days (refund/dispute
window), then flip to "confirmed". Confirmation runs lazily whenever
the creators list is viewed, and can be triggered explicitly via
POST /admin/creators/confirm-mature. (A scheduled job can replace the
lazy path later without changing behavior.)
"""

from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models import User, CreatorEarning, ReferralEvent
from app.routers.admin import verify_admin_key
from app.schemas import (
    PromoteCreatorRequest,
    CreatorResponse,
    CreatorFunnel,
    CreatorBalances,
    CreatorListResponse,
    MarkPaidResponse,
)

router = APIRouter()

CONFIRMATION_WINDOW_DAYS = 30


def _confirm_mature_earnings(db: Session) -> int:
    """Flip pending earnings older than the confirmation window to
    confirmed. Returns the number of earnings confirmed."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=CONFIRMATION_WINDOW_DAYS)
    now = datetime.now(timezone.utc)
    mature = (
        db.query(CreatorEarning)
        .filter(
            CreatorEarning.status == "pending",
            CreatorEarning.created_at <= cutoff,
        )
        .all()
    )
    for e in mature:
        e.status = "confirmed"
        e.confirmed_at = now
    if mature:
        db.commit()
    return len(mature)


@router.post("/admin/creators", tags=["Admin"])
async def promote_to_creator(
    request: PromoteCreatorRequest,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key),
):
    """Promote a user to creator status by user_id or referral_code."""
    if not request.user_id and not request.referral_code:
        raise HTTPException(400, "Either user_id or referral_code is required")

    query = db.query(User)
    if request.user_id:
        query = query.filter(User.user_id == request.user_id)
    elif request.referral_code:
        query = query.filter(User.referral_code == request.referral_code)

    user = query.first()
    if not user:
        raise HTTPException(404, "User not found")

    user.is_creator = True
    user.payout_rate_cents = request.payout_rate_cents or 250
    user.payout_method_note = request.payout_method_note
    db.commit()

    return {"status": "ok", "user_id": user.user_id, "referral_code": user.referral_code}


@router.get("/admin/creators", response_model=CreatorListResponse, tags=["Admin"])
async def list_creators(
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key),
):
    """List all creators with funnel stats and balances.

    Also lazily confirms any pending earnings past the 30-day window,
    so balances shown here are always current.
    """
    _confirm_mature_earnings(db)

    creators = db.query(User).filter(User.is_creator.is_(True)).all()
    result = []

    for c in creators:
        earnings = (
            db.query(CreatorEarning)
            .filter(CreatorEarning.creator_id == c.user_id)
            .all()
        )

        pending = sum(e.amount_cents for e in earnings if e.status == "pending")
        confirmed = sum(e.amount_cents for e in earnings if e.status == "confirmed")
        paid = sum(e.amount_cents for e in earnings if e.status == "paid")
        clawed = sum(e.amount_cents for e in earnings if e.status == "clawed_back")

        referred_count = (
            db.query(func.count(ReferralEvent.id))
            .filter(ReferralEvent.referrer_id == c.user_id)
            .scalar()
        ) or 0

        paid_conversions = sum(
            1 for e in earnings if e.status in ("confirmed", "paid")
        )

        result.append(
            CreatorResponse(
                user_id=c.user_id,
                referral_code=c.referral_code,
                payout_rate_cents=c.payout_rate_cents,
                payout_method_note=c.payout_method_note,
                funnel=CreatorFunnel(
                    clicks=0,
                    signups=referred_count,
                    paid_conversions=paid_conversions,
                ),
                balances=CreatorBalances(
                    pending_cents=pending,
                    confirmed_cents=confirmed,
                    paid_cents=paid,
                    clawed_back_cents=clawed,
                ),
            )
        )

    return CreatorListResponse(creators=result)


@router.post("/admin/creators/confirm-mature", tags=["Admin"])
async def confirm_mature(
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key),
):
    """Explicitly confirm all pending earnings past the 30-day window."""
    count = _confirm_mature_earnings(db)
    return {"status": "ok", "confirmed": count}


@router.post(
    "/admin/creators/{creator_id}/mark-paid",
    response_model=MarkPaidResponse,
    tags=["Admin"],
)
async def mark_creator_paid(
    creator_id: str,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin_key),
):
    """Mark all confirmed earnings as paid for a creator."""
    now = datetime.now(timezone.utc)
    earnings = (
        db.query(CreatorEarning)
        .filter(
            CreatorEarning.creator_id == creator_id,
            CreatorEarning.status == "confirmed",
        )
        .all()
    )
    for e in earnings:
        e.status = "paid"
        e.paid_at = now
    db.commit()
    return MarkPaidResponse(marked_paid=len(earnings))
