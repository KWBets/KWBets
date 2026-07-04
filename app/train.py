"""ML Model Training Pipeline for DoubleDown AI.

Trains an XGBoost classifier on historical features to predict game outcomes.
Evaluates on holdout set, saves model artifacts, and logs to ModelRegistry.
"""

import logging
import uuid
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, log_loss, brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib

from app.config import settings
from app.database import SessionLocal
from app.models import (
    ProcessedFeatures,
    ModelPrediction,
    ModelRegistry,
    ValueBet,
    PickOutcome,
)
from app.features import extract_feature_matrix, extract_feature_vector

logger = logging.getLogger(__name__)

# Path for saved model artifacts
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Training data loading
# ---------------------------------------------------------------------------

def load_training_data(
    db: Session,
    min_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load features and labels from the database for model training.

    Joins processed_features -> value_bets -> pick_outcomes to get
    actual results (labels) for each feature set.

    Returns:
        X: Feature matrix (n_samples x n_features)
        y: Label vector (1 = home_team_won, 0 = away_team_won)
        event_ids: List of event_ids corresponding to each sample
    """
    # Get features with outcome data
    results = (
        db.query(ProcessedFeatures, PickOutcome, ValueBet)
        .join(ValueBet, ProcessedFeatures.event_id == ValueBet.event_id)
        .join(PickOutcome, PickOutcome.value_bet_id == ValueBet.id)
        .filter(ProcessedFeatures.market_type == "h2h")
        .all()
    )

    if not results:
        logger.warning("No training data found (no features with outcomes)")
        return np.array([]), np.array([]), []

    feature_rows = []
    labels = []
    event_ids = []
    skipped = 0

    for pf, po, vb in results:
        # Determine label: 1 = home_team wins, 0 = away_team wins
        if po.actual_outcome == "won":
            if vb.team == pf.home_team:
                labels.append(1)
            elif vb.team == pf.away_team:
                labels.append(0)
            else:
                skipped += 1
                continue
        elif po.actual_outcome == "lost":
            if vb.team == pf.home_team:
                labels.append(0)
            elif vb.team == pf.away_team:
                labels.append(1)
            else:
                skipped += 1
                continue
        else:
            # push, cancelled - skip
            skipped += 1
            continue

        feature_rows.append(pf)
        event_ids.append(pf.event_id)

    if len(feature_rows) < min_samples:
        logger.warning(
            f"Only {len(feature_rows)} training samples (need {min_samples}, skipped {skipped})"
        )
        return np.array([]), np.array([]), []

    X = extract_feature_matrix(feature_rows)
    y = np.array(labels, dtype=np.int32)

    logger.info(
        f"Loaded {len(feature_rows)} training samples, "
        f"X shape: {X.shape}, y shape: {y.shape}, "
        f"home wins: {y.sum()}/{len(y)}, skipped: {skipped}"
    )

    return X, y, event_ids


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[xgb.XGBClassifier, dict, float, float]:
    """Train an XGBoost classifier with train/test split.

    Args:
        X: Feature matrix
        y: Labels (1 = home win, 0 = away win)
        test_size: Fraction of data to hold out for evaluation
        random_state: Random seed for reproducibility

    Returns:
        model: Trained XGBoost classifier
        metrics: Dict of evaluation metrics
        train_auc: ROC-AUC on training set
        test_auc: ROC-AUC on test set
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Compute scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    logger.info(
        f"Train size: {len(X_train)}, Test size: {len(X_test)}, "
        f"scale_pos_weight: {scale_pos_weight:.2f}"
    )

    # XGBoost model with hyperparameters tuned for sports prediction
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    y_test_class = model.predict(X_test)

    # Metrics
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    metrics = {
        "train_roc_auc": round(float(train_auc), 4),
        "test_roc_auc": round(float(test_auc), 4),
        "accuracy": round(float(accuracy_score(y_test, y_test_class)), 4),
        "precision": round(float(precision_score(y_test, y_test_class, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_test_class, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_test_class, zero_division=0)), 4),
        "log_loss": round(float(log_loss(y_test, y_test_pred)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_test_pred)), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "home_win_rate": float(y.mean()),
    }

    logger.info(
        f"Training complete — Train AUC: {train_auc:.4f}, "
        f"Test AUC: {test_auc:.4f}, Accuracy: {metrics['accuracy']:.4f}"
    )

    return model, metrics, train_auc, test_auc


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    model: xgb.XGBClassifier,
    metrics: dict,
    model_version: str,
) -> str:
    """Save model artifacts to disk.

    Args:
        model: Trained XGBoost model
        metrics: Evaluation metrics dict
        model_version: Version string

    Returns:
        Path to saved model file
    """
    model_path = os.path.join(MODELS_DIR, f"xgboost_{model_version}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Also save metadata
    meta_path = os.path.join(MODELS_DIR, f"xgboost_{model_version}_metrics.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return model_path


def load_model(model_version: str) -> Optional[xgb.XGBClassifier]:
    """Load a trained model by version string."""
    model_path = os.path.join(MODELS_DIR, f"xgboost_{model_version}.joblib")
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    return joblib.load(model_path)


def load_active_model(db: Session) -> Optional[xgb.XGBClassifier]:
    """Load the currently active model from the registry."""
    entry = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.is_active == True)
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )
    if not entry:
        logger.warning("No active model found in registry")
        return None
    return load_model(entry.model_version)


# ---------------------------------------------------------------------------
# Model registry logging
# ---------------------------------------------------------------------------

def log_model_to_registry(
    db: Session,
    model_version: str,
    model_path: str,
    metrics: dict,
    hyperparameters: Optional[dict] = None,
    set_active: bool = True,
) -> ModelRegistry:
    """Log a trained model to the ModelRegistry table.

    Args:
        db: Database session
        model_version: Unique version string (e.g. "v20260704_001")
        model_path: Path to saved model artifact
        metrics: Dict of evaluation metrics
        hyperparameters: Dict of model hyperparameters
        set_active: Whether to set this model as the active one

    Returns:
        ModelRegistry entry
    """
    # Deactivate current active model if setting new one
    if set_active:
        db.query(ModelRegistry).filter(
            ModelRegistry.is_active == True
        ).update({"is_active": False})

    entry = ModelRegistry(
        model_name="xgboost_h2h",
        model_version=model_version,
        model_type="xgboost",
        model_path=model_path,
        feature_set_version="v1",
        roc_auc=metrics.get("test_roc_auc"),
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        log_loss=metrics.get("log_loss"),
        brier_score=metrics.get("brier_score"),
        training_start=datetime.now(timezone.utc),  # approximate
        training_end=datetime.now(timezone.utc),
        training_samples=metrics.get("train_samples", 0) + metrics.get("test_samples", 0),
        hyperparameters=hyperparameters or {},
        is_active=set_active,
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.commit()
    logger.info(f"Model {model_version} logged to registry (active={set_active})")
    return entry


def get_current_model_auc(db: Session) -> Optional[float]:
    """Get the ROC-AUC of the current active model."""
    entry = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.is_active == True)
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )
    if entry and entry.roc_auc is not None:
        return entry.roc_auc
    return None


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_event(
    model: xgb.XGBClassifier,
    feature_row: ProcessedFeatures,
) -> dict[str, float]:
    """Run model prediction on a single feature row.

    Returns dict with 'home', 'away', 'draw' probabilities.
    """
    X = extract_feature_vector(feature_row).reshape(1, -1)
    home_prob = float(model.predict_proba(X)[0, 1])

    # For binary classification, away_prob = 1 - home_prob
    # (XGBoost is trained on home_win=1, away_win=0)
    away_prob = 1.0 - home_prob
    draw_prob = feature_row.draw_implied_prob or 0.0

    # If draw is possible, redistribute probabilities
    if draw_prob > 0.01:
        # Scale down win probs by draw probability
        total_win_prob = home_prob + away_prob
        if total_win_prob > 0:
            home_prob = home_prob / total_win_prob * (1.0 - draw_prob)
            away_prob = away_prob / total_win_prob * (1.0 - draw_prob)

    return {
        "home": round(home_prob, 4),
        "away": round(away_prob, 4),
        "draw": round(draw_prob, 4),
    }


def run_batch_predictions(
    db: Session,
    model: xgb.XGBClassifier,
    model_version: str,
    model_run_id: str,
) -> int:
    """Run predictions on all unprocessed features and store in ModelPrediction.

    Args:
        db: Database session
        model: Trained XGBoost model
        model_version: Version string
        model_run_id: Unique run identifier

    Returns:
        Number of predictions stored
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    features = db.query(ProcessedFeatures).all()

    if not features:
        logger.info("No features to predict on")
        return 0

    count = 0
    for feature in features:
        probs = predict_event(model, feature)

        prediction = ModelPrediction(
            event_id=feature.event_id,
            sport=feature.sport,
            home_team=feature.home_team,
            away_team=feature.away_team,
            commence_time=feature.commence_time,
            market_type=feature.market_type,
            home_win_probability=probs["home"],
            away_win_probability=probs["away"],
            draw_probability=probs["draw"],
            model_version=model_version,
            model_run_id=model_run_id,
            feature_timestamp=now,
            created_at=now,
        )
        db.add(prediction)
        count += 1

    db.commit()
    logger.info(f"Stored {count} predictions for model {model_version}")
    return count


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(
    db: Session,
    test_size: float = 0.2,
    set_active: bool = True,
) -> Optional[str]:
    """Run the complete training pipeline.

    Steps:
    1. Load training data (features + outcomes)
    2. Train XGBoost model
    3. Evaluate on holdout set
    4. Save model artifacts
    5. Log to ModelRegistry
    6. Generate predictions for all features

    Args:
        db: Database session
        test_size: Fraction for holdout set
        set_active: Whether to set the new model as active

    Returns:
        Model version string if successful, None otherwise
    """
    logger.info("=== Starting training pipeline ===")

    # 1. Load data
    X, y, event_ids = load_training_data(db)
    if len(X) == 0:
        logger.warning("Insufficient training data — skipping training")
        return None

    # 2. Train model
    model, metrics, train_auc, test_auc = train_model(X, y, test_size=test_size)

    # 3. Generate version string
    now = datetime.now(timezone.utc)
    model_version = f"v{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    model_run_id = uuid.uuid4().hex

    # 4. Save model
    model_path = save_model(model, metrics, model_version)

    # 5. Log to registry
    hyperparams = model.get_params()
    log_model_to_registry(
        db=db,
        model_version=model_version,
        model_path=model_path,
        metrics=metrics,
        hyperparameters=hyperparams,
        set_active=set_active,
    )

    # 6. Run batch predictions
    run_batch_predictions(db, model, model_version, model_run_id)

    logger.info(f"=== Training pipeline complete: {model_version} ===")
    return model_version


if __name__ == "__main__":
    # Standalone execution
    logging.basicConfig(level=logging.INFO)
    db = SessionLocal()
    try:
        run_training_pipeline(db)
    finally:
        db.close()
