"""Stripe webhook receiver — handles payment events for creator cash earnings.

Endpoints:
- POST /api/v1/webhooks/stripe — Stripe webhook target

Event types handled:
- invoice.payment_succeeded -> create CreatorEarning (first payment only)
- charge.refunded -> claw back earning
- charge.dispute.created -> claw back earning
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User, CreatorEarning

logger = logging.getLogger(__name__)

router = APIRouter()


def _find_referred_user(db: Session, invoice: dict) -> User | None:
    """Locate the local User for a Stripe invoice.

    Primary: match Stripe customer id against User.stripe_customer_id.
    Fallback: match invoice customer_email against User.email (if the
    column exists), backfilling stripe_customer_id on a hit so future
    events match on the primary path.
    """
    customer_id = invoice.get("customer")
    customer_email = invoice.get("customer_email")

    user = None
    if customer_id:
        user = (
            db.query(User)
            .filter(User.stripe_customer_id == customer_id)
            .first()
        )

    if user is None and customer_email and hasattr(User, "email"):
        user = (
            db.query(User)
            .filter(User.email == customer_email)
            .first()
        )
        # Backfill so the next webhook matches on customer id directly.
        if user is not None and customer_id and not user.stripe_customer_id:
            user.stripe_customer_id = customer_id

    return user


@router.post("/webhooks/stripe", tags=["Webhooks"])
async def stripe_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    """Receive Stripe webhook events for creator cash earnings.

    Requires STRIPE_WEBHOOK_SECRET to be configured. Fails closed if
    missing. Uses stripe_event_id as idempotency key (with the unique
    constraint as a hard backstop) to prevent double-writes on Stripe
    retries.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing stripe-signature header")

    if not settings.stripe_webhook_secret:
        raise HTTPException(
            status_code=500,
            detail="Stripe webhook secret not configured",
        )

    try:
        from stripe import Webhook  # lazy import: never in the startup path

        event = Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
    except ImportError:
        logger.error("stripe package not installed — cannot verify webhook")
        raise HTTPException(status_code=500, detail="stripe package not installed")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid signature: {e}")

    event_type = event["type"]
    stripe_event_id = event["id"]

    if event_type == "invoice.payment_succeeded":
        invoice = event["data"]["object"]
        billing_reason = invoice.get("billing_reason", "")

        # First payment on a new subscription only — renewals don't earn.
        if billing_reason != "subscription_create":
            return {"status": "skipped", "reason": f"not first payment ({billing_reason})"}

        subscription_id = invoice.get("subscription")
        payment_intent = invoice.get("payment_intent")

        # Idempotency fast-path (unique constraint below is the backstop).
        if stripe_event_id:
            existing = (
                db.query(CreatorEarning)
                .filter(CreatorEarning.stripe_event_id == stripe_event_id)
                .first()
            )
            if existing:
                return {"status": "already_processed"}

        user = _find_referred_user(db, invoice)
        if not user:
            logger.info(
                "webhook: no local user for customer=%s email=%s",
                invoice.get("customer"), invoice.get("customer_email"),
            )
            return {"status": "skipped", "reason": "user not found"}

        if not user.referred_by:
            return {"status": "skipped", "reason": "user not referred"}

        referrer = db.query(User).filter(User.user_id == user.referred_by).first()
        if not referrer or not referrer.is_creator:
            return {"status": "skipped", "reason": "referrer not a creator"}

        earning = CreatorEarning(
            creator_id=referrer.user_id,
            referred_user_id=user.user_id,
            stripe_subscription_id=subscription_id,
            stripe_payment_intent_id=payment_intent,
            stripe_event_id=stripe_event_id,
            amount_cents=referrer.payout_rate_cents,
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        db.add(earning)
        try:
            db.commit()
        except IntegrityError:
            # Stripe retried and two deliveries raced — the unique
            # constraint on stripe_event_id caught it. Not an error.
            db.rollback()
            return {"status": "already_processed"}

        logger.info(
            "CreatorEarning created: creator=%s referred=%s amount_cents=%s",
            referrer.user_id, user.user_id, referrer.payout_rate_cents,
        )

        return {"status": "created", "earning_id": earning.id}

    elif event_type in ("charge.refunded", "charge.dispute.created"):
        charge = event["data"]["object"]
        payment_intent = charge.get("payment_intent")

        if payment_intent:
            earning = (
                db.query(CreatorEarning)
                .filter(
                    CreatorEarning.stripe_payment_intent_id == payment_intent,
                    CreatorEarning.status.in_(["pending", "confirmed"]),
                )
                .first()
            )
            if earning:
                earning.status = "clawed_back"
                db.commit()
                logger.info(
                    "CreatorEarning clawed back: id=%s creator=%s payment_intent=%s",
                    earning.id, earning.creator_id, payment_intent,
                )
                return {"status": "clawed_back"}

        return {"status": "skipped"}

    return {"status": "ignored", "type": event_type}
