"""C7: Módulo de abstención y derivación a revisión humana."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("motie_es.abstention")


class AbstentionModule:
    """Detecta casos de baja confianza y los deriva a revisión humana."""

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        stage_discrepancy_threshold: float = 0.30,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.stage_discrepancy_threshold = stage_discrepancy_threshold

    def should_abstain(
        self,
        proba_max: float,
        stage_discrepancy: Optional[float] = None,
    ) -> bool:
        """Determina si un caso debe derivarse a revisión humana."""
        if proba_max < self.confidence_threshold:
            return True
        if stage_discrepancy is not None and stage_discrepancy > self.stage_discrepancy_threshold:
            return True
        return False

    def process_predictions(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Separa predicciones en alta confianza y derivadas."""
        high_confidence = []
        derived = []
        for pred in predictions:
            proba_max = pred.get("confianza", 0.0)
            discrepancy = pred.get("stage_discrepancy", None)
            if self.should_abstain(proba_max, discrepancy):
                pred["razon_derivacion"] = self._explain(proba_max, discrepancy)
                derived.append(pred)
            else:
                high_confidence.append(pred)
        return {"alta_confianza": high_confidence, "derivados": derived}

    def _explain(self, proba_max: float, discrepancy: Optional[float]) -> str:
        """Genera explicación de la incertidumbre."""
        reasons = []
        if proba_max < self.confidence_threshold:
            reasons.append(
                f"confianza baja ({proba_max:.2f} < {self.confidence_threshold})"
            )
        if discrepancy is not None and discrepancy > self.stage_discrepancy_threshold:
            reasons.append(
                f"discrepancia entre etapas ({discrepancy:.2f} > {self.stage_discrepancy_threshold})"
            )
        return "; ".join(reasons) if reasons else "caso ambiguo"