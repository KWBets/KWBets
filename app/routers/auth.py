"""Auth dependency — verifies Supabase JWT from Authorization header.

Production: validates Bearer JWT against SUPABASE_JWT_SECRET.
Development (no secret set): falls back to X-User-Id header with warning log.

Creates users on first touch (create-on-first-touch pattern).
"""

import logging
import secrets
import string
from datetime import datetime, timezone

import jwt as pyjwt  # PyJWT library
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User

logger = logging.getLogger(__name__)


def _generate_referral_code() -> str:
    """Generate a readable referral code: DD-XXXXX (no confusable chars)."""
    alphabet = string.ascii_uppercase + string.digits
    alphabet = alphabet.translate(str.maketrans("", "", "0O1IL"))
    code = "".join(secrets.choice(alphabet) for _ in range(5))
    return f"DD-{code}"


def _create_user(db: Session, user_id: str, email_domain: str | None = None) -> User:
    """Create a new user with a unique referral code."""
    code = _generate_referral_code()
    while db.query(User).filter(User.referral_code == code).first():
        code = _generate_referral_code()
    user = User(
        user_id=user_id,
        referral_code=code,
        email_domain=email_domain,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user)
    db.commit()
    return user


async def get_current_user(
    authorization: str = Header(...),
    x_user_id: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Verify Supabase JWT and return the user.

    Production (SUPABASE_JWT_SECRET is set):
        - Reads Authorization: Bearer <token>
        - Validates JWT signature, expiry, audience
        - Extracts user_id from 'sub' or 'email' claim
        - Creates user on first touch

    Development (no secret configured):
        - Falls back to X-User-Id header with a warning log
        - Creates user on first touch
    """
    # ------------------------------------------------------------------
    # Production path — verify Supabase JWT
    # ------------------------------------------------------------------
    if settings.supabase_jwt_secret:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Invalid authorization header"
            )
        token = authorization[7:]
        try:
            payload = pyjwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
        except pyjwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except pyjwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

        user_id = payload.get("sub") or payload.get("email")
        if not user_id:
            raise HTTPException(
                status_code=401, detail="Token missing user identity"
            )

        email = payload.get("email", "")
        email_domain = email.split("@")[-1] if "@" in email else None

        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = _create_user(db, user_id, email_domain)
        return user

    # ------------------------------------------------------------------
    # Development fallback — X-User-Id header (no auth)
    # ------------------------------------------------------------------
    logger.warning(
        "SUPABASE_JWT_SECRET not configured — using dev fallback X-User-Id header"
    )
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(
            status_code=400,
            detail="X-User-Id header is required when SUPABASE_JWT_SECRET is not set",
        )

    user_id = x_user_id.strip()
    email_domain = user_id.split("@")[-1].lower() if "@" in user_id else None

    user = db.query(User).filter(User.user_id == user_id).first()
    if user:
        return user

    return _create_user(db, user_id, email_domain)