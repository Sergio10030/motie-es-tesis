"""Script de ejecución del Brazo B."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_all_metrics
from src.utils.io import load_config, save_metrics
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main(config_path: str) -> None:
    from src.arms.arm_b_finetuning import ArmBFinetuning

    config = load_config(config_path)
    logger = setup_logger(
        "motie_es.run_arm_b",
        level=config["logging"]["level"],
        log_file=config["logging"]["file"],
    )
    set_seed(config["project"]["seed"])

    logger.info("=== Ejecutando Brazo B ===")

    train_df = pd.read_csv(
        Path(config["paths"]["data_processed"]) / "train.csv", encoding="utf-8"
    )
    val_df = pd.read_csv(
        Path(config["paths"]["data_processed"]) / "val.csv", encoding="utf-8"
    )
    test_df = pd.read_csv(
        Path(config["paths"]["data_processed"]) / "test.csv", encoding="utf-8"
    )

    for frame in (train_df, val_df, test_df):
        frame["text_clean"] = frame["text_clean"].fillna("").astype(str)

    model_type = ArmBFinetuning(
        model_name=config["arm_b"]["default_model"],
        learning_rate=config["arm_b"]["learning_rate"],
        batch_size=config["arm_b"]["batch_size"],
        weight_decay=config["arm_b"]["weight_decay"],
        num_epochs=config["arm_b"]["num_epochs"],
        max_length=config["arm_b"]["max_length"],
        warmup_ratio=config["arm_b"]["warmup_ratio"],
        class_weighting=config["arm_b"]["class_weighting"],
        gradient_clip=config["arm_b"]["gradient_clip"],
        early_stopping_patience=config["arm_b"]["early_stopping_patience"],
        seed=config["project"]["seed"],
        checkpoint_dir=str(Path(config["paths"]["checkpoints"]) / "tipo"),
    )

    model_type.fit(
        train_df["text_clean"].tolist(),
        train_df["tipo_funcional"].tolist(),
        val_df["text_clean"].tolist(),
        val_df["tipo_funcional"].tolist(),
    )

    preds_type = model_type.predict(test_df["text_clean"].tolist())
    history_type = model_type.history
    del model_type
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    model_priority = ArmBFinetuning(
        model_name=config["arm_b"]["default_model"],
        learning_rate=config["arm_b"]["learning_rate"],
        batch_size=config["arm_b"]["batch_size"],
        weight_decay=config["arm_b"]["weight_decay"],
        num_epochs=config["arm_b"]["num_epochs"],
        max_length=config["arm_b"]["max_length"],
        warmup_ratio=config["arm_b"]["warmup_ratio"],
        class_weighting=config["arm_b"]["class_weighting"],
        gradient_clip=config["arm_b"]["gradient_clip"],
        early_stopping_patience=config["arm_b"]["early_stopping_patience"],
        seed=config["project"]["seed"],
        checkpoint_dir=str(Path(config["paths"]["checkpoints"]) / "prioridad"),
    )
    model_priority.fit(
        train_df["text_clean"].tolist(),
        train_df["prioridad"].tolist(),
        val_df["text_clean"].tolist(),
        val_df["prioridad"].tolist(),
    )
    preds_priority = model_priority.predict(test_df["text_clean"].tolist())

    labels = sorted(test_df["tipo_funcional"].unique().tolist())
    metrics = compute_all_metrics(
        test_df["tipo_funcional"].tolist(),
        preds_type["tipo_predicho"],
        labels,
    )
    priority_labels = sorted(test_df["prioridad"].unique().tolist())
    priority_metrics = compute_all_metrics(
        test_df["prioridad"].tolist(),
        preds_priority["tipo_predicho"],
        priority_labels,
    )
    metrics.update({
        f"{key}_prioridad": value
        for key, value in priority_metrics.items()
    })
    metrics["history_tipo"] = history_type
    metrics["history_prioridad"] = model_priority.history

    save_metrics(metrics, "arm_b_metrics", config["paths"]["results_metrics"])
    logger.info(
        "Métricas Brazo B: F1 macro tipo = %.4f, prioridad = %.4f",
        metrics["f1_macro"],
        metrics["f1_macro_prioridad"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)