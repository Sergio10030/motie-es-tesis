"""C8: Concordancia con el criterio humano."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.utils.logger import setup_logger

logger = setup_logger("motie_es.human_agreement")


class HumanAgreementValidator:
    """Calcula kappa de Cohen, kappa ponderado y alpha de Krippendorff.

    Compara tres escenarios:
        - Brazo C solo
        - Humano solo
        - Brazo C + humano con revisión de baja confianza
    """

    def __init__(
        self,
        kappa_weighted: bool = True,
        kappa_weights: str = "quadratic",
        krippendorff_level: str = "ordinal",
    ) -> None:
        self.kappa_weighted = kappa_weighted
        self.kappa_weights = kappa_weights
        self.krippendorff_level = krippendorff_level

    @staticmethod
    def cohen_kappa(y1: List[str], y2: List[str]) -> float:
        """Calcula kappa de Cohen."""
        return float(cohen_kappa_score(y1, y2))

    def cohen_kappa_weighted(self, y1: List[str], y2: List[str]) -> float:
        """Calcula kappa de Cohen ponderado."""
        return float(
            cohen_kappa_score(y1, y2, weights=self.kappa_weights)
        )

    @staticmethod
    def krippendorff_alpha(
        data: List[List[Optional[str]]],
        level: str = "ordinal",
    ) -> float:
        """Calcula alpha de Krippendorff.

        Args:
            data: Lista de listas (una por unidad, una entrada por evaluador).
            level: 'nominal', 'ordinal', 'interval', 'ratio'.

        Returns:
            Alpha de Krippendorff.
        """
        try:
            import krippendorff

            # Convertir a formato numérico
            all_vals = sorted({v for row in data for v in row if v is not None})
            val2id = {v: i for i, v in enumerate(all_vals)}
            arr = np.array(
                [[val2id[v] if v is not None else np.nan for v in row] for row in data],
                dtype=float,
            ).T  # krippendorff espera (evaluadores, unidades)
            return float(krippendorff.alpha(reliability_data=arr, level_of_measurement=level))
        except ImportError:
            logger.warning("krippendorff no instalado; devolviendo NaN")
            return float("nan")

    def compare_scenarios(
        self,
        y_true: List[str],
        y_arm_c: List[str],
        y_human: List[str],
        derived_mask: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """Compara los tres escenarios."""
        results: Dict[str, Any] = {}

        # Escenario 1: Brazo C solo
        results["arm_c_only"] = {
            "kappa_cohen": self.cohen_kappa(y_true, y_arm_c),
            "kappa_weighted": self.cohen_kappa_weighted(y_true, y_arm_c),
        }

        # Escenario 2: Humano solo
        results["human_only"] = {
            "kappa_cohen": self.cohen_kappa(y_true, y_human),
            "kappa_weighted": self.cohen_kappa_weighted(y_true, y_human),
        }

        # Escenario 3: Brazo C + humano
        if derived_mask is not None:
            y_combined = [
                y_human[i] if derived_mask[i] else y_arm_c[i]
                for i in range(len(y_arm_c))
            ]
        else:
            y_combined = [
                y_human[i] if y_arm_c[i] != y_human[i] else y_arm_c[i]
                for i in range(len(y_arm_c))
            ]

        results["arm_c_plus_human"] = {
            "kappa_cohen": self.cohen_kappa(y_true, y_combined),
            "kappa_weighted": self.cohen_kappa_weighted(y_true, y_combined),
        }

        return results

    @staticmethod
    def classify_discrepancies(
        y_true: List[str],
        y_arm_c: List[str],
        y_human: List[str],
    ) -> Dict[str, List[int]]:
        """Clasifica discrepancias en error del modelo, error del humano y ambigüedad."""
        model_error = []
        human_error = []
        genuine_ambiguity = []

        for i, (true, arm, hum) in enumerate(zip(y_true, y_arm_c, y_human)):
            if arm == hum:
                continue
            if arm == true and hum != true:
                human_error.append(i)
            elif hum == true and arm != true:
                model_error.append(i)
            else:
                genuine_ambiguity.append(i)

        return {
            "error_modelo": model_error,
            "error_humano": human_error,
            "ambiguedad_genuina": genuine_ambiguity,
        }

    def per_class_kappa(
        self,
        y_true: List[str],
        y_pred: List[str],
        classes: List[str],
    ) -> Dict[str, float]:
        """Calcula kappa de Cohen por clase (one-vs-rest)."""
        result = {}
        for cls in classes:
            y_true_bin = [1 if y == cls else 0 for y in y_true]
            y_pred_bin = [1 if y == cls else 0 for y in y_pred]
            result[cls] = self.cohen_kappa(y_true_bin, y_pred_bin)
        return result