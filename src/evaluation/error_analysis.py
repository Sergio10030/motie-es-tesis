"""Análisis de errores y discrepancias."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("motie_es.error_analysis")


class ErrorAnalyzer:
    """Analiza patrones de error en las predicciones."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def confusion_pairs(
        self,
        y_true_col: str = "tipo_funcional",
        y_pred_col: str = "tipo_predicho",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Devuelve los pares de confusión más frecuentes."""
        errors = self.df[self.df[y_true_col] != self.df[y_pred_col]]
        if errors.empty:
            return pd.DataFrame(columns=["true", "pred", "count"])
        pairs = (
            errors.groupby([y_true_col, y_pred_col])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(top_n)
        )
        pairs.columns = ["true", "pred", "count"]
        return pairs

    def error_by_register(
        self,
        density_col: str = "densidad_jerga",
        y_true_col: str = "tipo_funcional",
        y_pred_col: str = "tipo_predicho",
    ) -> pd.DataFrame:
        """Analiza errores por nivel de densidad de registro."""
        df = self.df.copy()
        df["error"] = (df[y_true_col] != df[y_pred_col]).astype(int)
        df["register_bin"] = pd.cut(
            df[density_col], bins=[-0.01, 0.02, 0.05, 0.10, 1.0],
            labels=["muy_baja", "baja", "media", "alta"],
        )
        return df.groupby("register_bin", observed=False)["error"].agg(["mean", "count"])

    def summary(self) -> Dict[str, Any]:
        """Resumen de errores."""
        return {
            "total": len(self.df),
            "errores": int((self.df["tipo_funcional"] != self.df["tipo_predicho"]).sum())
            if "tipo_predicho" in self.df.columns else 0,
        }