"""Metrics computation and a tiny JSON model-registry.

`compute_metrics` gives every model an identical, comparable scorecard.
`update_registry` accumulates those scorecards into ``models/metrics.json`` which
the Flask app's "Model comparison" tab reads directly — so the web UI never shows
hand-typed numbers, only what training actually produced.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REGISTRY_PATH = MODELS_DIR / "metrics.json"


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: List[str]) -> Dict:
    """Return a JSON-serialisable scorecard (accuracy, macro/weighted F1, per-class)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def update_registry(model_name: str, metrics: Dict, extra: Dict | None = None) -> Path:
    """Merge one model's test metrics into ``models/metrics.json``."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    registry = {}
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text())
    entry = {"metrics": metrics, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if extra:
        entry.update(extra)
    registry[model_name] = entry
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    return REGISTRY_PATH


def pretty_report(y_true, y_pred, class_names) -> str:
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
