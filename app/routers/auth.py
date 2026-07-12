"""Auth dependency — extracts user identity from X-User-Id header.

Creates users on first touch (create-on-first-touch pattern).
"""

import secrets
import string
from datetime import datetime, timezone
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User


def _generate_referral_code() -> str:
    """Generate a readable referral code: DD-XXXXX."""
    alphabet = string.ascii_uppercase + string.digits
    # Remove confusing chars
    alphabet = alphabet.translate(str.maketrans("", "", "0O1IL"))
    code = "".join(secrets.choice(alphabet) for _ in range(5))
    return f"DD-{code}"


def get_current_user(
    x_user_id: str = Header(..., description="User identifier from frontend"),
    db: Session = Depends(get_db),
) -> User:
    """Get or create user by X-User-Id header.

    If user doesn't exist, creates them with an auto-generated referral code.
    Extracts email domain for fraud heuristic if the ID looks like an email.
    """
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(status_code=400, detail="X-User-Id header is required")

    user_id = x_user_id.strip()

    user = db.query(User).filter(User.user_id == user_id).first()
    if user:
        return user

    # Generate unique referral code
    code = _generate_referral_code()
    while db.query(User).filter(User.referral_code == code).first():
        code = _generate_referral_code()

    # Extract email domain if applicable
    email_domain = None
    if "@" in user_id:
        email_domain = user_id.split("@")[1].lower()

    user = User(
        user_id=user_id,
        referral_code=code,
        email_domain=email_domain,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user)
    db.commit()
    return user