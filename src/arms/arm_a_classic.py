"""C4: Brazo A - Línea base clásica (TF-IDF + Regresión Logística)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("motie_es.arm_a")


class ArmAClassic:
    """Brazo A: línea base clásica con TF-IDF y regresión logística.

    Configuración:
        - TfidfVectorizer: unigramas y bigramas, 10.000 términos.
        - LogisticRegression: solver SAGA, L2, max_iter=3000.
        - Opcionalmente SVM lineal.
        - Clasificación plana o multi-tarea.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        max_iter: int = 3000,
        solver: str = "saga",
        penalty: str = "l2",
        C: float = 1.0,
        use_svm: bool = False,
        svm_kernel: str = "linear",
        svm_C: float = 1.0,
        multi_task: bool = False,
        seed: int = 42,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_iter = max_iter
        self.solver = solver
        self.penalty = penalty
        self.C = C
        self.use_svm = use_svm
        self.svm_kernel = svm_kernel
        self.svm_C = svm_C
        self.multi_task = multi_task
        self.seed = seed
        set_seed(seed)

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model_type: Optional[LogisticRegression] = None
        self.model_priority: Optional[LogisticRegression] = None
        self.svm_model: Optional[Any] = None
        self.classes_type_: Optional[np.ndarray] = None
        self.classes_priority_: Optional[np.ndarray] = None
        self._is_fitted = False

    def _build_pipeline(self) -> Pipeline:
        """Construye el pipeline TF-IDF + clasificador."""
        clf: Any
        if self.use_svm:
            clf = LinearSVC(C=self.svm_C, max_iter=self.max_iter, random_state=self.seed)
        else:
            clf = LogisticRegression(
                solver=self.solver,
                penalty=self.penalty,
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.seed,
                n_jobs=-1,
            )
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                sublinear_tf=True,
            )),
            ("clf", clf),
        ])

    def fit(
        self,
        texts: List[str],
        y_type: List[str],
        y_priority: Optional[List[str]] = None,
    ) -> "ArmAClassic":
        """Entrena el modelo.

        Args:
            texts: Lista de textos.
            y_type: Etiquetas de tipo funcional.
            y_priority: Etiquetas de prioridad (si multi_task).

        Returns:
            self
        """
        logger.info("Entrenando Brazo A sobre %d ejemplos", len(texts))
        t0 = time.time()

        self.model_type = self._build_pipeline()
        self.model_type.fit(texts, y_type)
        self.classes_type_ = np.array(self.model_type.named_steps["clf"].classes_)

        if self.multi_task and y_priority is not None:
            self.model_priority = self._build_pipeline()
            self.model_priority.fit(texts, y_priority)
            self.classes_priority_ = np.array(
                self.model_priority.named_steps["clf"].classes_
            )

        self._is_fitted = True
        logger.info("Brazo A entrenado en %.2f s", time.time() - t0)
        return self

    def predict(self, texts: List[str]) -> Dict[str, List[str]]:
        """Predice tipo y prioridad."""
        if not self._is_fitted or self.model_type is None:
            raise RuntimeError("El modelo no está entrenado. Llame a fit() primero.")

        pred_type = self.model_type.predict(texts).tolist()
        result: Dict[str, List[str]] = {"tipo_predicho": pred_type}

        if self.multi_task and self.model_priority is not None:
            result["prioridad_predicha"] = self.model_priority.predict(texts).tolist()

        return result

    def predict_proba(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Devuelve probabilidades por clase."""
        if not self._is_fitted or self.model_type is None:
            raise RuntimeError("El modelo no está entrenado.")

        clf = self.model_type.named_steps["clf"]
        proba_type = clf.predict_proba(texts)
        result: Dict[str, np.ndarray] = {"tipo_proba": proba_type}

        if self.multi_task and self.model_priority is not None:
            clf_p = self.model_priority.named_steps["clf"]
            result["prioridad_proba"] = clf_p.predict_proba(texts)

        return result

    def cross_validate(
        self,
        texts: List[str],
        y_type: List[str],
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """Validación cruzada estratificada de 5 pliegues."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        scores: List[float] = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y_type), 1):
            X_tr = [texts[i] for i in train_idx]
            X_val = [texts[i] for i in val_idx]
            y_tr = [y_type[i] for i in train_idx]
            y_val = [y_type[i] for i in val_idx]

            pipe = self._build_pipeline()
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            scores.append(f1)
            logger.info("Pliegue %d: F1 macro = %.4f", fold, f1)

        return {
            "f1_macro_mean": float(np.mean(scores)),
            "f1_macro_std": float(np.std(scores)),
            "f1_macro_folds": scores,
        }

    def evaluate(
        self,
        texts: List[str],
        y_type: List[str],
        y_priority: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evalúa el modelo y devuelve métricas."""
        preds = self.predict(texts)
        report: Dict[str, Any] = {
            "tipo": classification_report(
                y_type, preds["tipo_predicho"], output_dict=True, zero_division=0
            ),
            "f1_macro_tipo": f1_score(
                y_type, preds["tipo_predicho"], average="macro", zero_division=0
            ),
            "f1_weighted_tipo": f1_score(
                y_type, preds["tipo_predicho"], average="weighted", zero_division=0
            ),
        }
        if self.multi_task and y_priority is not None and "prioridad_predicha" in preds:
            report["prioridad"] = classification_report(
                y_priority, preds["prioridad_predicha"], output_dict=True, zero_division=0
            )
            report["f1_macro_prioridad"] = f1_score(
                y_priority, preds["prioridad_predicha"], average="macro", zero_division=0
            )
        return report