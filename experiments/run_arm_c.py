"""Script de ejecución del Brazo C."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arms.arm_c_fewshot import ArmCFewShot, LLMClient
from src.evaluation.metrics import compute_all_metrics
from src.utils.io import load_config, save_metrics
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main(config_path: str) -> None:
    config = load_config(config_path)
    logger = setup_logger(
        "motie_es.run_arm_c",
        level=config["logging"]["level"],
        log_file=config["logging"]["file"],
    )
    set_seed(config["project"]["seed"])

    logger.info("=== Ejecutando Brazo C ===")

    train_df = pd.read_csv(
        Path(config["paths"]["data_processed"]) / "train.csv", encoding="utf-8"
    )
    test_df = pd.read_csv(
        Path(config["paths"]["data_processed"]) / "test.csv", encoding="utf-8"
    )

    # Cliente LLM
    provider_cfg = next(
        p for p in config["arm_c"]["providers"]
        if p["name"] == config["arm_c"]["default_provider"]
    )
    client = LLMClient(
        provider=provider_cfg["name"],
        model=provider_cfg["model"],
        api_key_env=provider_cfg["api_key_env"],
        base_url=provider_cfg["base_url"],
        temperature=config["arm_c"]["temperature"],
        max_tokens=config["arm_c"]["max_tokens"],
        max_retries=config["arm_c"]["max_retries"],
    )

    model = ArmCFewShot(
        client=client,
        k_shot=config["arm_c"]["default_k"],
        temperature=config["arm_c"]["temperature"],
        selection_strategy=config["arm_c"]["selection_strategy"],
        fallback_label=config["arm_c"]["fallback_label"],
        prompts_dir="config/prompts",
        seed=config["project"]["seed"],
    )

    types = sorted(train_df["tipo_funcional"].unique().tolist())
    priorities = sorted(train_df["prioridad"].unique().tolist())
    model.set_taxonomy(types, priorities)
    model.select_examples(train_df)

    results = model.classify(test_df["text_clean"].tolist())

    annotation_df = test_df[["texto", "tipo_funcional", "prioridad"]].reset_index(drop=True).copy()
    annotation_df["tipo_predicho_arm_c"] = [r["tipo_predicho"] for r in results]
    annotation_df["prioridad_predicha_arm_c"] = [
        r["prioridad_predicha"] for r in results
    ]
    annotation_df["anotacion_humana"] = ""
    annotation_df["prioridad_anotacion_humana"] = ""
    annotation_path = Path(config["paths"]["data_processed"]) / "test_with_preds.csv"
    annotation_df.to_csv(annotation_path, index=False, encoding="utf-8")
    logger.info("Plantilla para anotación humana guardada en %s", annotation_path)

    preds = [r["tipo_predicho"] for r in results]
    labels = sorted(test_df["tipo_funcional"].unique().tolist())
    metrics = compute_all_metrics(
        test_df["tipo_funcional"].tolist(), preds, labels
    )
    priority_labels = sorted(test_df["prioridad"].unique().tolist())
    priority_metrics = compute_all_metrics(
        test_df["prioridad"].tolist(),
        [r["prioridad_predicha"] for r in results],
        priority_labels,
    )
    metrics.update({
        f"{key}_prioridad": value
        for key, value in priority_metrics.items()
    })
    metrics["execution_mode"] = "api" if client.api_key else "simulado"
    metrics["invalid_responses"] = sum(
        1 for r in results if not r["tipo_valido"] or not r["prioridad_valida"]
    )

    save_metrics(metrics, "arm_c_metrics", config["paths"]["results_metrics"])
    logger.info("Métricas Brazo C: F1 macro = %.4f", metrics["f1_macro"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)