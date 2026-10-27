"""Script de ejecución del Brazo A."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arms.arm_a_classic import ArmAClassic
from src.evaluation.metrics import compute_all_metrics
from src.utils.io import load_config, load_corpus, save_metrics
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main(config_path: str) -> None:
    config = load_config(config_path)
    logger = setup_logger(
        "motie_es.run_arm_a",
        level=config["logging"]["level"],
        log_file=config["logging"]["file"],
    )
    set_seed(config["project"]["seed"])

    logger.info("=== Ejecutando Brazo A ===")

    # Cargar corpus procesado
    corpus_path = Path(config["paths"]["data_processed"]) / "corpus_processed.csv"
    if not corpus_path.exists():
        logger.error("No existe el corpus procesado: %s", corpus_path)
        return

    df = pd.read_csv(corpus_path, encoding="utf-8")
    logger.info("Corpus cargado: %d reportes", len(df))

    # Particiones
    train_path = Path(config["paths"]["data_processed"]) / "train.csv"
    test_path = Path(config["paths"]["data_processed"]) / "test.csv"
    train_df = pd.read_csv(train_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")
    train_df["text_clean"] = train_df["text_clean"].fillna("").astype(str)
    test_df["text_clean"] = test_df["text_clean"].fillna("").astype(str)

    # Entrenar
    model = ArmAClassic(
        max_features=config["arm_a"]["tfidf_max_features"],
        ngram_range=tuple(config["arm_a"]["tfidf_ngram_range"]),
        max_iter=config["arm_a"]["logistic_max_iter"],
        solver=config["arm_a"]["logistic_solver"],
        penalty=config["arm_a"]["logistic_penalty"],
        C=config["arm_a"]["logistic_C"],
        use_svm=config["arm_a"]["use_svm"],
        multi_task=config["arm_a"]["multi_task"],
        seed=config["project"]["seed"],
    )
    model.fit(
        train_df["text_clean"].tolist(),
        train_df["tipo_funcional"].tolist(),
        train_df["prioridad"].tolist() if config["arm_a"]["multi_task"] else None,
    )

    # Evaluar
    preds = model.predict(test_df["text_clean"].tolist())
    labels = sorted(test_df["tipo_funcional"].unique().tolist())
    metrics = compute_all_metrics(
        test_df["tipo_funcional"].tolist(),
        preds["tipo_predicho"],
        labels,
    )
    if "prioridad_predicha" in preds:
        priority_labels = sorted(test_df["prioridad"].unique().tolist())
        priority_metrics = compute_all_metrics(
            test_df["prioridad"].tolist(),
            preds["prioridad_predicha"],
            priority_labels,
        )
        metrics.update({
            f"{key}_prioridad": value
            for key, value in priority_metrics.items()
        })

    save_metrics(metrics, "arm_a_metrics", config["paths"]["results_metrics"])
    logger.info("Métricas Brazo A: F1 macro = %.4f", metrics["f1_macro"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)