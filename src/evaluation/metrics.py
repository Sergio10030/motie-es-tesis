"""Métricas de evaluación para clasificación."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)

from src.utils.logger import setup_logger

logger = setup_logger("motie_es.metrics")


def compute_all_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, Any]:
    """Calcula todas las métricas relevantes."""
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )
    f2_per_class = 5 * (p * r) / (4 * p + r + 1e-12)
    f2_macro = float(np.mean(f2_per_class)) if len(f2_per_class) > 0 else 0.0

    recall_per_class = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )

    return {
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f2_macro": f2_macro,
        "accuracy": float(accuracy),
        "recall_por_clase": {
            lab: float(recall_per_class[i]) if i < len(recall_per_class) else 0.0
            for i, lab in enumerate(labels)
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    """F1 macro."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def f2_score(y_true: List[str], y_pred: List[str]) -> float:
    """F2 macro (recall dos veces más importante que precision)."""
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    if 4 * p + r == 0:
        return 0.0
    return float(5 * (p * r) / (4 * p + r))