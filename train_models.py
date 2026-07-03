from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import config
from processing.feature_engineering import build_features, prepare_training_data

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gradient_boost_v1"


def train_model(
    features: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    target_col: str = "won",
) -> dict:
    X, y = prepare_training_data(features, target_col)
    if X.empty or len(y) < config.MIN_SAMPLES_FOR_RETRAIN:
        raise ValueError(f"Insufficient training data: {len(y)} samples (need {config.MIN_SAMPLES_FOR_RETRAIN})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TRAIN_TEST_SPLIT, random_state=config.RANDOM_STATE
    )

    model = GradientBoostingClassifier(random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "log_loss": round(log_loss(y_test, y_prob), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_columns": list(X.columns),
    }

    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
    }

    path = config.MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(artifact, path)
    logger.info("Saved model to %s — metrics: %s", path, metrics)
    return metrics


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> dict | None:
    path = config.MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def predict(features: pd.DataFrame, model_name: str = DEFAULT_MODEL_NAME) -> pd.DataFrame:
    artifact = load_model(model_name)
    if artifact is None:
        raise FileNotFoundError(f"Model '{model_name}' not found in {config.MODELS_DIR}")

    model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    X = features.reindex(columns=feature_cols, fill_value=0)
    probs = model.predict_proba(X)[:, 1]

    result = features.copy()
    result["predicted_prob"] = probs
    result["model_name"] = model_name
    return result
