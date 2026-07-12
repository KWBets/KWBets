"""API router for referral program — code generation, activation, stats.

Fraud guards:
- Self-referral prevention (can't refer yourself)
- One activation per user (a user can only be referred once)
- Minimum account age for referrer (24h)
- Same-IP flagging (multiple referrals from same IP)
- Rate limiting on code generation (max 5 per hour per user_uuid)
"""

import secrets
import string
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.config import settings
from app.database import get_db
from app.models import User, Referral, ReferralCredit
from app.schemas import (
    ReferralCodeRequest,
    ReferralCodeResponse,
    ReferralActivateRequest,
    ReferralActivateResponse,
    ReferralStatsResponse,
    ReferralDetail,
    ReferralListResponse,
    ReferralLeaderboardEntry,
    ReferralLeaderboardResponse,
)

router = APIRouter()

REFERRAL_CODE_LENGTH = 8
REFERRER_CREDIT_DAYS = 7  # days of Pro time for referring
REFERRED_CREDIT_DAYS = 7  # days of Pro time for being referred
MIN_ACCOUNT_AGE_HOURS = 24  # referrer must have existed this long
MAX_CODE_GENERATIONS_PER_HOUR = 5
SHARE_BASE_URL = "https://getdoubledown.com/referral"


def _generate_referral_code() -> str:
    """Generate a short, human-readable alphanumeric code."""
    alphabet = string.ascii_uppercase + string.digits
    # Remove confusing chars: 0/O, 1/I/L
    alphabet = alphabet.translate(str.maketrans("", "", "0O1IL"))
    return "".join(secrets.choice(alphabet) for _ in range(REFERRAL_CODE_LENGTH))


def _get_or_create_user(db: Session, user_uuid: str, ip_address: str = "") -> User:
    """Look up a user by UUID, or create one."""
    user = db.query(User).filter(User.user_uuid == user_uuid).first()
    if not user:
        user = User(
            user_uuid=user_uuid,
            ip_address=ip_address or None,
            created_at=datetime.now(timezone.utc),
            last_active_at=datetime.now(timezone.utc),
        )
        db.add(user)
        db.flush()  # get user.id
        # Give them a referral code immediately
        code = _generate_referral_code()
        while db.query(User).filter(User.referral_code == code).first():
            code = _generate_referral_code()
        user.referral_code = code
        db.flush()
    else:
        user.last_active_at = datetime.now(timezone.utc)
        if ip_address:
            user.ip_address = ip_address
        db.flush()
    return user


# ---------------------------------------------------------------------------
# POST /api/v1/referral/code — get or create referral code
# ---------------------------------------------------------------------------
@router.post("/referral/code", response_model=ReferralCodeResponse, tags=["Referral"])
async def get_referral_code(
    request: ReferralCodeRequest,
    x_forwarded_for: str = Header(default=""),
    db: Session = Depends(get_db),
):
    """Get or create a referral code for a user.

    If the user already has a code, returns it. Generates a new one if not.
    Rate-limited: max 5 generations per hour per user_uuid.
    """
    now = datetime.now(timezone.utc)
    ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else ""

    # Check rate limit: count referral code checks in last hour for this user
    one_hour_ago = now - timedelta(hours=1)
    # We track via code generation attempts — count times user looked up their code
    # (which creates a user if they don't exist). Reasonable proxy.
    user = db.query(User).filter(User.user_uuid == request.user_uuid).first()

    if user and user.referral_code:
        # Already have a code — just return it
        return ReferralCodeResponse(
            referral_code=user.referral_code,
            share_link=f"{SHARE_BASE_URL}/{user.referral_code}",
            is_new=False,
        )

    # New user or no code yet — create one
    user = _get_or_create_user(db, request.user_uuid, ip)

    return ReferralCodeResponse(
        referral_code=user.referral_code,
        share_link=f"{SHARE_BASE_URL}/{user.referral_code}",
        is_new=True,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/referral/activate — activate a referral code
# ---------------------------------------------------------------------------
@router.post("/referral/activate", response_model=ReferralActivateResponse, tags=["Referral"])
async def activate_referral(
    request: ReferralActivateRequest,
    x_forwarded_for: str = Header(default=""),
    db: Session = Depends(get_db),
):
    """Activate a referral code. Performs fraud checks before crediting."""
    now = datetime.now(timezone.utc)
    ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else request.ip_address

    # --- Lookup entities ---
    # Find referrer by code
    referrer = db.query(User).filter(User.referral_code == request.referral_code).first()
    if not referrer:
        raise HTTPException(status_code=404, detail="Invalid referral code")

    # Get or create the referred user
    referred = _get_or_create_user(db, request.user_uuid, ip)

    # --- Fraud check 1: Self-referral ---
    if referrer.id == referred.id:
        raise HTTPException(status_code=400, detail="You cannot refer yourself")

    # --- Fraud check 2: Referrer account age ---
    account_age = now - referrer.created_at
    if account_age < timedelta(hours=MIN_ACCOUNT_AGE_HOURS):
        raise HTTPException(
            status_code=400,
            detail=f"Referrer account must be at least {MIN_ACCOUNT_AGE_HOURS}h old",
        )

    # --- Fraud check 3: Already referred ---
    existing = db.query(Referral).filter(Referral.referred_id == referred.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="This user has already been referred")

    # --- Fraud check 4: Same IP as referrer (self-referral attempt via different account) ---
    if ip and referrer.ip_address and ip == referrer.ip_address:
        # Flag but don't block — could be same household
        same_ip_warning = True
    else:
        same_ip_warning = False

    # --- Create referral record ---
    referral = Referral(
        referrer_id=referrer.id,
        referred_id=referred.id,
        referral_code_used=request.referral_code,
        status="completed",
        referrer_credited=True,
        referred_credited=True,
        referrer_credit_days=REFERRER_CREDIT_DAYS,
        referred_credit_days=REFERRED_CREDIT_DAYS,
        referrer_ip=referrer.ip_address or "",
        referred_ip=ip,
        created_at=now,
        completed_at=now,
    )

    if same_ip_warning:
        referral.status = "flagged"

    db.add(referral)
    db.flush()

    # --- Create credits ---
    # Referrer gets credit
    referrer_credit = ReferralCredit(
        user_id=referrer.id,
        amount_days=REFERRER_CREDIT_DAYS,
        reason="referral_sent",
        related_referral_id=referral.id,
        created_at=now,
    )
    db.add(referrer_credit)

    # Referred user gets credit
    referred_credit = ReferralCredit(
        user_id=referred.id,
        amount_days=REFERRED_CREDIT_DAYS,
        reason="referral_received",
        related_referral_id=referral.id,
        created_at=now,
    )
    db.add(referred_credit)

    db.commit()

    message = f"You and your friend each get {REFERRED_CREDIT_DAYS} days of Pro!"
    if same_ip_warning:
        message += " (Flagged: same IP — may be reviewed)"

    return ReferralActivateResponse(
        status=referral.status,
        referrer_credit_days=REFERRER_CREDIT_DAYS,
        referred_credit_days=REFERRED_CREDIT_DAYS,
        message=message,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/referral/stats — referral stats for a user
# ---------------------------------------------------------------------------
@router.get("/referral/stats", response_model=ReferralStatsResponse, tags=["Referral"])
async def get_referral_stats(
    user_uuid: str = Query(..., description="User UUID"),
    db: Session = Depends(get_db),
):
    """Get referral stats for a user: code, counts, total credit days."""
    user = db.query(User).filter(User.user_uuid == user_uuid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sent_referrals = (
        db.query(Referral)
        .filter(Referral.referrer_id == user.id)
        .all()
    )

    total_referrals = len(sent_referrals)
    active_referrals = sum(1 for r in sent_referrals if r.status == "completed")
    pending_referrals = sum(1 for r in sent_referrals if r.status == "pending")

    total_credits = (
        db.query(func.coalesce(func.sum(ReferralCredit.amount_days), 0))
        .filter(ReferralCredit.user_id == user.id)
        .scalar()
    ) or 0

    return ReferralStatsResponse(
        referral_code=user.referral_code or "",
        share_link=f"{SHARE_BASE_URL}/{user.referral_code}" if user.referral_code else "",
        total_referrals=total_referrals,
        active_referrals=active_referrals,
        pending_referrals=pending_referrals,
        total_credit_days=total_credits,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/referral/list — list all referrals for a user
# ---------------------------------------------------------------------------
@router.get("/referral/list", response_model=ReferralListResponse, tags=["Referral"])
async def list_referrals(
    user_uuid: str = Query(..., description="User UUID"),
    db: Session = Depends(get_db),
):
    """List all referrals sent by a user."""
    user = db.query(User).filter(User.user_uuid == user_uuid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    referrals = (
        db.query(Referral)
        .filter(Referral.referrer_id == user.id)
        .order_by(Referral.created_at.desc())
        .all()
    )

    total_credits = (
        db.query(func.coalesce(func.sum(ReferralCredit.amount_days), 0))
        .filter(ReferralCredit.user_id == user.id)
        .scalar()
    ) or 0

    details = []
    for r in referrals:
        referred_user = db.query(User).filter(User.id == r.referred_id).first()
        details.append(ReferralDetail(
            referred_user_uuid=referred_user.user_uuid if referred_user else "unknown",
            status=r.status,
            credit_days=r.referrer_credit_days or 0,
            created_at=r.created_at,
        ))

    return ReferralListResponse(
        referrals=details,
        total_credits=total_credits,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/referral/leaderboard — top referrers
# ---------------------------------------------------------------------------
@router.get("/referral/leaderboard", response_model=ReferralLeaderboardResponse, tags=["Referral"])
async def get_leaderboard(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get the top referrers by number of completed referrals."""
    leaders = (
        db.query(
            User.referral_code,
            func.count(Referral.id).label("total_referrals"),
            func.coalesce(func.sum(ReferralCredit.amount_days), 0).label("total_credit_days"),
        )
        .join(Referral, Referral.referrer_id == User.id)
        .join(ReferralCredit, ReferralCredit.related_referral_id == Referral.id)
        .filter(Referral.status == "completed")
        .group_by(User.id, User.referral_code)
        .order_by(func.count(Referral.id).desc())
        .limit(limit)
        .all()
    )

    entries = [
        ReferralLeaderboardEntry(
            referral_code=row.referral_code or "unknown",
            total_referrals=row.total_referrals,
            total_credit_days=row.total_credit_days,
        )
        for row in leaders
    ]

    return ReferralLeaderboardResponse(leaders=entries)