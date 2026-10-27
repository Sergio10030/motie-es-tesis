"""Script de validación de concordancia con criterio humano (C8)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, save_metrics
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.validation.human_agreement import HumanAgreementValidator


def main(config_path: str) -> int:
    config = load_config(config_path)
    logger = setup_logger(
        "motie_es.run_validation",
        level=config["logging"]["level"],
        log_file=config["logging"]["file"],
    )
    set_seed(config["project"]["seed"])

    logger.info("=== Ejecutando validación de concordancia ===")

    test_path = Path(config["paths"]["data_processed"]) / "test_with_preds.csv"
    if not test_path.exists():
        logger.error("No existe %s. Ejecute primero los brazos.", test_path)
        return 1

    df = pd.read_csv(test_path, encoding="utf-8")
    required_columns = {
        "tipo_funcional",
        "tipo_predicho_arm_c",
        "anotacion_humana",
        "prioridad",
        "prioridad_predicha_arm_c",
        "prioridad_anotacion_humana",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        logger.error("Faltan columnas requeridas para validar: %s", sorted(missing_columns))
        return 1
    annotation_columns = ["anotacion_humana", "prioridad_anotacion_humana"]
    unannotated = {
        column: df[column].isna() | df[column].astype(str).str.strip().eq("")
        for column in annotation_columns
    }
    if any(mask.any() for mask in unannotated.values()):
        logger.error(
            "Faltan anotaciones humanas: %s en %s. Complete ambas columnas antes de validar.",
            {column: int(mask.sum()) for column, mask in unannotated.items()},
            test_path,
        )
        return 1

    validator = HumanAgreementValidator(
        kappa_weighted=config["validation"]["kappa_weighted"],
        kappa_weights=config["validation"]["kappa_weights"],
        krippendorff_level=config["validation"]["krippendorff_level"],
    )

    y_true = df["tipo_funcional"].tolist()
    y_arm_c = df["tipo_predicho_arm_c"].tolist()
    y_human = df["anotacion_humana"].tolist()

    scenarios = validator.compare_scenarios(y_true, y_arm_c, y_human)
    discrepancies = validator.classify_discrepancies(y_true, y_arm_c, y_human)
    per_class = validator.per_class_kappa(y_true, y_arm_c, sorted(set(y_true)))

    # Alpha de Krippendorff
    data = [[t, a, h] for t, a, h in zip(y_true, y_arm_c, y_human)]
    alpha = validator.krippendorff_alpha(data, level=config["validation"]["krippendorff_level"])

    y_priority_true = df["prioridad"].tolist()
    y_priority_arm_c = df["prioridad_predicha_arm_c"].tolist()
    y_priority_human = df["prioridad_anotacion_humana"].tolist()
    priority_scenarios = validator.compare_scenarios(
        y_priority_true, y_priority_arm_c, y_priority_human
    )
    priority_discrepancies = validator.classify_discrepancies(
        y_priority_true, y_priority_arm_c, y_priority_human
    )
    priority_per_class = validator.per_class_kappa(
        y_priority_true, y_priority_arm_c, sorted(set(y_priority_true))
    )
    priority_data = [
        [true, arm, human]
        for true, arm, human in zip(
            y_priority_true, y_priority_arm_c, y_priority_human
        )
    ]
    priority_alpha = validator.krippendorff_alpha(
        priority_data, level=config["validation"]["krippendorff_level"]
    )

    metrics = {
        "scenarios": scenarios,
        "discrepancies": discrepancies,
        "per_class_kappa": per_class,
        "krippendorff_alpha": alpha,
        "priority_scenarios": priority_scenarios,
        "priority_discrepancies": priority_discrepancies,
        "priority_per_class_kappa": priority_per_class,
        "krippendorff_alpha_prioridad": priority_alpha,
    }

    save_metrics(metrics, "validation_metrics", config["paths"]["results_metrics"])
    logger.info(
        "Validación completada. Alpha tipo = %.4f, prioridad = %.4f",
        alpha,
        priority_alpha,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    raise SystemExit(main(args.config))