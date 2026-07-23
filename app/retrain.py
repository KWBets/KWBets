"""Auto-Retraining Pipeline for DoubleDown AI.

Checks for new labeled outcome data, retrains the model if available,
compares ROC-AUC against current active model, and promotes only if improved.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import settings
from app.database import SessionLocal
from app.models import (
    ModelRegistry,
    PickOutcome,
    ValueBet,
)
from app.train import (
    load_training_data,
    train_model,
    save_model,
    log_model_to_registry,
    get_current_model_auc,
    run_batch_predictions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check for new data
# ---------------------------------------------------------------------------

def count_new_outcomes(db: Session, since: Optional[datetime] = None) -> int:
    """Count pick_outcomes added since a given timestamp (won/lost only)."""
    query = db.query(func.count(PickOutcome.id)).filter(
        PickOutcome.actual_outcome.in_(["won", "lost"])
    )
    if since:
        query = query.filter(PickOutcome.created_at > since)
    return query.scalar() or 0


def get_last_training_time(db: Session) -> Optional[datetime]:
    """Get the training_end timestamp of the most recent model."""
    entry = (
        db.query(ModelRegistry)
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )
    if entry and entry.training_end:
        return entry.training_end
    return None


def has_new_data(db: Session, min_new_samples: int = 10) -> tuple[bool, int]:
    """Check if there are enough new labeled outcomes for retraining.

    Args:
        db: Database session
        min_new_samples: Minimum new samples to trigger retraining

    Returns:
        (has_new_data, count_of_new_samples)
    """
    last_train = get_last_training_time(db)
    if last_train is None:
        # No model trained yet — count all available data
        total = count_new_outcomes(db)
        logger.info(f"No previous training found. Total available outcomes: {total}")
        return total >= min_new_samples, total

    new_count = count_new_outcomes(db, since=last_train)
    logger.info(f"New outcomes since {last_train}: {new_count}")
    return new_count >= min_new_samples, new_count


# ---------------------------------------------------------------------------
# Comparison and promotion
# ---------------------------------------------------------------------------

def should_promote(
    new_metrics: dict,
    current_auc: Optional[float],
    min_improvement: float = 0.005,
) -> tuple[bool, str]:
    """Decide whether to promote the new model over the current one.

    Args:
        new_metrics: Metrics dict from the newly trained model
        current_auc: ROC-AUC of the current active model (None if no model)
        min_improvement: Minimum AUC improvement to promote (default 0.005)

    Returns:
        (should_promote, reason)
    """
    new_auc = new_metrics.get("test_roc_auc", 0.0)

    if current_auc is None:
        return True, "No existing active model — promoting first model"

    improvement = new_auc - current_auc
    if improvement >= min_improvement:
        return True, (
            f"New model AUC ({new_auc:.4f}) improved over current "
            f"({current_auc:.4f}) by {improvement:.4f} (threshold: {min_improvement})"
        )
    else:
        return False, (
            f"New model AUC ({new_auc:.4f}) did not sufficiently improve over current "
            f"({current_auc:.4f}) — improvement {improvement:.4f} < threshold {min_improvement}"
        )


# ---------------------------------------------------------------------------
# Main retraining orchestrator
# ---------------------------------------------------------------------------

def run_retraining_pipeline(
    db: Session,
    test_size: float = 0.2,
    min_new_samples: int = 10,
    min_improvement: float = 0.005,
) -> dict:
    """Run the auto-retraining pipeline.

    Steps:
    1. Check for new labeled outcome data
    2. Load all training data
    3. Train new model
    4. Compare ROC-AUC with current active model
    5. Promote only if improved
    6. Generate predictions with the (possibly new) model

    Args:
        db: Database session
        test_size: Fraction for holdout set
        min_new_samples: Minimum new samples to trigger retraining
        min_improvement: Minimum AUC improvement to promote

    Returns:
        Dict with status of the retraining run
    """
    logger.info("=== Running auto-retraining pipeline ===")

    result = {
        "status": "skipped",
        "message": "",
        "new_model_version": None,
        "promoted": False,
        "new_auc": None,
        "current_auc": None,
        "new_samples": 0,
    }

    # 1. Check for new data
    has_data, new_count = has_new_data(db, min_new_samples)
    result["new_samples"] = new_count

    if not has_data:
        # Not enough data — still try generating predictions with current model
        current_active = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.is_active == True)
            .first()
        )
        if current_active:
            from app.train import load_model
            model = load_model(current_active.model_version)
            if model:
                
                import uuid
                preds = run_batch_predictions(
                    db, model, current_active.model_version, uuid.uuid4().hex
                )
                result["message"] = (
                    f"Insufficient new data ({new_count} < {min_new_samples} needed). "
                    f"Used current model for predictions ({preds} predictions stored)."
                )
                result["status"] = "partial"
                return result
        result["message"] = (
            f"Insufficient new data ({new_count} < {min_new_samples} needed). "
            f"No retraining performed."
        )
        return result

    # 2. Load training data
    X, y, _ = load_training_data(db)
    if len(X) == 0:
        result["message"] = "No valid training data could be loaded"
        return result

    # 3. Train new model
    model, metrics, train_auc, test_auc = train_model(X, y, test_size=test_size)

    current_auc = get_current_model_auc(db)
    result["current_auc"] = current_auc
    result["new_auc"] = test_auc

    # 4. Compare and decide promotion
    promote, reason = should_promote(metrics, current_auc, min_improvement)
    result["promoted"] = promote

    # 5. Always save and log, but only activate if improved
    from datetime import datetime, timezone
    import uuid as uuid_mod

    now = datetime.now(timezone.utc)
    model_version = f"v{now.strftime('%Y%m%d_%H%M%S')}_{uuid_mod.uuid4().hex[:8]}"
    model_path = save_model(model, metrics, model_version)

    hyperparams = model.get_params()
    log_model_to_registry(
        db=db,
        model_version=model_version,
        model_path=model_path,
        metrics=metrics,
        hyperparameters=hyperparams,
        set_active=promote,  # Only set active if improved
    )

    # 6. Generate predictions
    import uuid as uuid_mod2
    model_run_id = uuid_mod2.uuid4().hex
    pred_count = run_batch_predictions(db, model, model_version, model_run_id)

    result["new_model_version"] = model_version
    result["status"] = "promoted" if promote else "logged_not_promoted"

    if promote:
        result["message"] = (
            f"Model {model_version} promoted. "
            f"Test AUC: {test_auc:.4f} vs previous {current_auc or 'N/A':.4f}. "
            f"Stored {pred_count} predictions. Reason: {reason}"
        )
    else:
        result["message"] = (
            f"Model {model_version} trained but not promoted. "
            f"Test AUC: {test_auc:.4f} vs current {current_auc or 'N/A':.4f}. "
            f"Reason: {reason}. Stored {pred_count} predictions."
        )

    logger.info(f"=== Retraining pipeline complete: {result['message']} ===")
    return result


# ---------------------------------------------------------------------------
# CLI entry point (called by scheduler)
# ---------------------------------------------------------------------------

def scheduled_retrain():
    """Called by the APScheduler every 24 hours."""
    logger.info("Scheduled retrain triggered")
    db = SessionLocal()
    try:
        result = run_retraining_pipeline(db)
        logger.info(f"Retrain result: {result['status']} — {result['message']}")
        return result
    except Exception as e:
        logger.error(f"Retrain failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = SessionLocal()
    try:
        run_retraining_pipeline(db)
    finally:
        db.close()
