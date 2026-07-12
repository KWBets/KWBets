"""Auth dependency — verifies Supabase JWT from Authorization header.

Uses JWKS endpoint for asymmetric key verification (ES256/RS256).
Fails closed: no dev fallback, no X-User-Id bypass.
"""

import logging
import secrets
import string
from datetime import datetime, timezone

import jwt as pyjwt
from fastapi import Depends, Header, HTTPException
from jwt import PyJWKClient
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User

logger = logging.getLogger(__name__)

# Cached JWKS client (module-level to avoid fetching on every request)
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    """Get or create the cached PyJWKClient for the Supabase JWKS endpoint."""
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(settings.supabase_jwks_url)
    return _jwks_client


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
    db: Session = Depends(get_db),
) -> User:
    """Verify Supabase JWT from the Authorization header and return the user.

    Uses the Supabase JWKS endpoint to fetch the signing key for asymmetric
    verification (ES256/RS256). Fails closed on any auth error.

    Creates users on first touch (create-on-first-touch pattern).
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header — expected 'Bearer <token>'",
        )

    token = authorization[7:]

    try:
        # Fetch the signing key from the JWKS endpoint
        jwks_client = _get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        payload = pyjwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256"],
            audience="authenticated",
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {e}")

    # Extract user identity
    user_id = payload.get("sub") or payload.get("email")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Token does not contain a user identifier (sub or email)",
        )

    # Extract email for domain-based fraud heuristic
    email = payload.get("email", "")
    email_domain = email.split("@")[-1] if "@" in email else None

    # Create-on-first-touch
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = _create_user(db, user_id, email_domain)

    return user