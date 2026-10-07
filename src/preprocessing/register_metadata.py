"""Cálculo de metadatos de densidad de registro por reporte."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.preprocessing.cleaner import RegisterAwarePreprocessor
from src.preprocessing.lexicon_loader import LexiconLoader
from src.utils.logger import setup_logger

logger = setup_logger("motie_es.register_metadata")


class RegisterMetadataCalculator:
    """Calcula densidad de jerga y vocabulario administrativo por reporte.

    d_r = |T_r ∩ L_r| / |T_r|
    """

    def __init__(
        self,
        lexicon: LexiconLoader,
        preprocessor: RegisterAwarePreprocessor,
    ) -> None:
        self.lexicon = lexicon
        self.preprocessor = preprocessor

    def compute(self, text: str) -> Dict[str, float]:
        """Calcula los metadatos de registro para un texto."""
        tokens = self.preprocessor.tokenize(text)
        return {
            "densidad_jerga": self.lexicon.get_density(tokens, "jerga"),
            "densidad_admin": self.lexicon.get_density(tokens, "admin"),
            "n_tokens": len(tokens),
        }

    def compute_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "texto",
    ) -> pd.DataFrame:
        """Añade columnas de densidad a un DataFrame."""
        records: List[Dict[str, float]] = []
        for text in df[text_column].fillna(""):
            records.append(self.compute(text))

        meta_df = pd.DataFrame(records)
        return pd.concat([df.reset_index(drop=True), meta_df], axis=1)