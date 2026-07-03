from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

REGISTRY_PATH = config.MODELS_DIR / "registry.json"


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"models": [], "active_model": None}
    return json.loads(REGISTRY_PATH.read_text())


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def register_model(model_name: str, metrics: dict) -> None:
    registry = _load_registry()
    entry = {
        "name": model_name,
        "metrics": metrics,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "path": str(config.MODELS_DIR / f"{model_name}.joblib"),
    }
    registry["models"].append(entry)
    _save_registry(registry)
    logger.info("Registered model %s", model_name)


def set_active_model(model_name: str) -> None:
    registry = _load_registry()
    names = {m["name"] for m in registry["models"]}
    if model_name not in names:
        raise ValueError(f"Model '{model_name}' not in registry")
    registry["active_model"] = model_name
    _save_registry(registry)
    logger.info("Active model set to %s", model_name)


def get_active_model() -> str | None:
    return _load_registry().get("active_model")


def list_models() -> list[dict]:
    return _load_registry().get("models", [])


def get_best_model(metric: str = "roc_auc") -> str | None:
    models = list_models()
    if not models:
        return None
    return max(models, key=lambda m: m["metrics"].get(metric, 0))["name"]
