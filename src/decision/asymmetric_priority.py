"""C7: Priorización con coste asimétrico."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, recall_score

from src.utils.logger import setup_logger

logger = setup_logger("motie_es.asymmetric_priority")


class AsymmetricPrioritizer:
    """Calibra umbrales de decisión con coste asimétrico.

    C = α·FN_crítica + β·FP_crítica + γ·FN_no_crítica + δ·FP_no_crítica
    con α > β y α > γ, δ.
    """

    def __init__(
        self,
        priority_levels: List[str],
        critical_level: str = "alta",
        cost_alpha: float = 5.0,
        cost_beta: float = 1.0,
        cost_gamma: float = 2.0,
        cost_delta: float = 1.0,
        confidence_threshold: float = 0.70,
        stage_discrepancy_threshold: float = 0.30,
    ) -> None:
        self.priority_levels = priority_levels
        self.critical_level = critical_level
        self.alpha = cost_alpha
        self.beta = cost_beta
        self.gamma = cost_gamma
        self.delta = cost_delta
        self.confidence_threshold = confidence_threshold
        self.stage_discrepancy_threshold = stage_discrepancy_threshold
        self.thresholds: Dict[str, float] = {}

    def compute_cost(
        self,
        y_true: List[str],
        y_pred: List[str],
    ) -> float:
        """Calcula el coste asimétrico total."""
        cost = 0.0
        for true, pred in zip(y_true, y_pred):
            true_crit = true == self.critical_level
            pred_crit = pred == self.critical_level
            if true_crit and not pred_crit:
                cost += self.alpha  # FN crítico
            elif not true_crit and pred_crit:
                cost += self.beta  # FP crítico
            elif not true_crit and not pred_crit and true != pred:
                cost += self.gamma  # FN no crítico
            elif not true_crit and not pred_crit and true != pred:
                cost += self.delta  # FP no crítico
        return cost

    def calibrate(
        self,
        y_true: List[str],
        y_proba: np.ndarray,
        classes: List[str],
    ) -> Dict[str, float]:
        """Calibra umbrales por nivel de prioridad mediante grid search."""
        best_cost = float("inf")
        best_thresholds = {level: 0.5 for level in self.priority_levels}

        grid = np.arange(0.30, 0.91, 0.05)

        for t_alta in grid:
            for t_media in grid:
                thresholds = {"alta": t_alta, "media": t_media, "baja": 0.0}
                y_pred = self._apply_thresholds(y_proba, classes, thresholds)
                cost = self.compute_cost(y_true, y_pred)
                if cost < best_cost:
                    best_cost = cost
                    best_thresholds = thresholds

        self.thresholds = best_thresholds
        logger.info("Umbrales calibrados: %s | coste=%.2f", best_thresholds, best_cost)
        return best_thresholds

    def _apply_thresholds(
        self,
        y_proba: np.ndarray,
        classes: List[str],
        thresholds: Dict[str, float],
    ) -> List[str]:
        """Aplica umbrales a las probabilidades."""
        preds = []
        for row in y_proba:
            probs = {cls: row[i] for i, cls in enumerate(classes)}
            # Priorizar clase crítica si supera umbral
            if probs.get(self.critical_level, 0) >= thresholds.get(self.critical_level, 0.5):
                preds.append(self.critical_level)
            else:
                # Elegir la clase no crítica con mayor probabilidad
                non_crit = {k: v for k, v in probs.items() if k != self.critical_level}
                if non_crit:
                    best = max(non_crit, key=non_crit.get)
                    preds.append(best)
                else:
                    preds.append(self.critical_level)
        return preds

    def evaluate(
        self,
        y_true: List[str],
        y_pred: List[str],
    ) -> Dict[str, Any]:
        """Calcula F1 macro, F2 macro y recall por clase."""
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # F2 macro
        f2_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        # Recalcular F2 por clase
        from sklearn.metrics import precision_recall_fscore_support

        p, r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        f2_per_class = 5 * (p * r) / (4 * p + r + 1e-12)
        f2_macro = float(np.mean(f2_per_class))

        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0, labels=self.priority_levels
        )

        return {
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "f2_macro": f2_macro,
            "recall_por_clase": {
                level: float(recall_per_class[i]) if i < len(recall_per_class) else 0.0
                for i, level in enumerate(self.priority_levels)
            },
            "coste_asimetrico": self.compute_cost(y_true, y_pred),
            "umbrales": self.thresholds,
        }