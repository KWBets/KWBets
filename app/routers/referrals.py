"""Referral program router — endpoints for referral codes, claiming, activation, and entitlement."""

from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.database import get_db
from app.models import User, ReferralEvent, ProCreditUsage
from app.routers.auth import get_current_user
from app.schemas import (
    MyReferralResponse,
    ReferralHistoryEntry,
    ClaimReferralRequest,
    ClaimReferralResponse,
    CheckActivationResponse,
    EntitlementResponse,
)

router = APIRouter()

REWARD_DAYS = 7
ACTIVATION_HOURS = 24
MONTHLY_LIMIT = 10
SHARE_BASE_URL = "https://getdoubledown.com"


# ---------------------------------------------------------------------------
# GET /api/v1/me/referral — get current user's referral info
# ---------------------------------------------------------------------------
@router.get("/me/referral", response_model=MyReferralResponse, tags=["Referral"])
async def get_my_referral(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the current user's referral code, stats, and history."""
    events = (
        db.query(ReferralEvent)
        .filter(ReferralEvent.referrer_id == user.user_id)
        .order_by(ReferralEvent.created_at.desc())
        .all()
    )

    invited_count = len(events)
    activated_count = sum(1 for e in events if e.status in ("activated", "rewarded"))

    history = []
    for e in events:
        # Get referred user's ID as the "email" display
        referred_user = db.query(User).filter(User.user_id == e.referred_id).first()
        referred_email = referred_user.user_id if referred_user else e.referred_id
        history.append(ReferralHistoryEntry(
            referred_email=referred_email,
            status=e.status,
            created_at=e.created_at,
        ))

    return MyReferralResponse(
        code=user.referral_code,
        link=f"{SHARE_BASE_URL}?ref={user.referral_code}",
        invited_count=invited_count,
        activated_count=activated_count,
        pro_credit_days=user.pro_credit_days,
        referral_history=history,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/referrals/claim — claim a referral code on signup
# ---------------------------------------------------------------------------
@router.post("/referrals/claim", response_model=ClaimReferralResponse, tags=["Referral"])
async def claim_referral(
    request: ClaimReferralRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Claim a referral code — sets the current user's referred_by.

    Fraud checks:
    1. Self-referral: code belongs to current user
    2. Max 10/month: referrer has < 10 events this month
    3. Same-domain flag: email domains match
    """
    # Find referrer by code
    referrer = db.query(User).filter(User.referral_code == request.referral_code).first()
    if not referrer:
        raise HTTPException(status_code=404, detail="Invalid referral code")

    referrer_id = referrer.user_id

    # --- Fraud check 1: Self-referral ---
    if referrer_id == user.user_id:
        raise HTTPException(status_code=400, detail="You cannot refer yourself")

    # --- Fraud check 2: Already referred ---
    existing = db.query(ReferralEvent).filter(
        ReferralEvent.referred_id == user.user_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="You have already been referred")

    # --- Fraud check 3: Max 10/month for referrer ---
    now = datetime.now(timezone.utc)
    if not referrer.is_creator:
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_count = (
            db.query(func.count(ReferralEvent.id))
            .filter(
                ReferralEvent.referrer_id == referrer_id,
                ReferralEvent.created_at >= month_start,
                ReferralEvent.status.in_(["activated", "rewarded", "pending"]),
            )
            .scalar()
        ) or 0
        if monthly_count >= MONTHLY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="This referrer has reached the monthly referral limit",
            )

    # --- Fraud check 4: Same-domain flag ---
    flag_reason = None
    if user.email_domain and referrer.email_domain and user.email_domain == referrer.email_domain:
        flag_reason = "same_domain"

    # Create referral event
    event = ReferralEvent(
        referrer_id=referrer_id,
        referred_id=user.user_id,
        status="flagged" if flag_reason else "pending",
        flag_reason=flag_reason,
        created_at=now,
    )
    db.add(event)

    # Set referred_by on user
    user.referred_by = referrer_id
    db.commit()

    msg = "Referral claimed successfully!"
    if flag_reason:
        msg += " (flagged for review — same email domain)"

    return ClaimReferralResponse(
        status=event.status,
        message=msg,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/referrals/check-activation — check and reward eligible referrals
# ---------------------------------------------------------------------------
@router.post("/referrals/check-activation", response_model=CheckActivationResponse, tags=["Referral"])
async def check_activation(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Check for referrals that are eligible for activation (24h+ old) and reward them.

    Activation rules:
    - Referred user was created > 24h ago (ACTIVATION_HOURS)
    - Event status is 'pending'
    - When activated: both referrer and referred get +7 pro_credit_days
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=ACTIVATION_HOURS)
    activated_ids = []

    # Find pending referrals where the referred user was created > 24h ago
    pending_events = (
        db.query(ReferralEvent)
        .join(User, User.user_id == ReferralEvent.referred_id)
        .filter(
            ReferralEvent.referrer_id == user.user_id,
            ReferralEvent.status == "pending",
            User.created_at <= cutoff,
        )
        .all()
    )

    for event in pending_events:
        # Reward both users
        referrer = db.query(User).filter(User.user_id == event.referrer_id).first()
        referred = db.query(User).filter(User.user_id == event.referred_id).first()

        if referrer and not referrer.is_creator:
            referrer.pro_credit_days = (referrer.pro_credit_days or 0) + REWARD_DAYS
        if referred:
            referred.pro_credit_days = (referred.pro_credit_days or 0) + REWARD_DAYS

        event.status = "rewarded"
        event.rewarded_at = now

        activated_ids.append(event.referred_id)

    db.commit()

    return CheckActivationResponse(
        activated=activated_ids,
        total_rewarded=len(activated_ids),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/me/entitlement — check pro entitlement
# ---------------------------------------------------------------------------
@router.get("/me/entitlement", response_model=EntitlementResponse, tags=["Referral"])
async def get_entitlement(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Check the current user's pro entitlement status."""
    credit_days = user.pro_credit_days or 0

    if credit_days > 0:
        # Estimate expiry: credits burn at 1/day, so remaining days from now
        expires = datetime.now(timezone.utc) + timedelta(days=credit_days)
        return EntitlementResponse(
            is_pro=True,
            source="credit",
            credit_days_remaining=credit_days,
            credit_expires=expires,
        )

    return EntitlementResponse(
        is_pro=False,
        source="none",
        credit_days_remaining=0,
        credit_expires=None,
    )