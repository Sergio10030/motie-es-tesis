"""Pruebas unitarias para concordancia humana."""
import logging

import pandas as pd
import pytest

from src.validation.human_agreement import HumanAgreementValidator


def test_cohen_kappa_perfect():
    v = HumanAgreementValidator()
    y1 = ["a", "b", "c"]
    y2 = ["a", "b", "c"]
    assert v.cohen_kappa(y1, y2) == pytest.approx(1.0)


def test_cohen_kappa_zero():
    v = HumanAgreementValidator()
    y1 = ["a", "a", "b", "b"]
    y2 = ["b", "b", "a", "a"]
    kappa = v.cohen_kappa(y1, y2)
    assert kappa <= 0.0


def test_compare_scenarios():
    v = HumanAgreementValidator()
    y_true = ["alta", "media", "baja", "alta"]
    y_arm_c = ["alta", "media", "baja", "media"]
    y_human = ["alta", "media", "baja", "alta"]
    result = v.compare_scenarios(y_true, y_arm_c, y_human)
    assert "arm_c_only" in result
    assert "human_only" in result
    assert "arm_c_plus_human" in result


def test_classify_discrepancies():
    v = HumanAgreementValidator()
    y_true = ["alta", "media", "baja"]
    y_arm_c = ["media", "media", "baja"]  # error modelo en índice 0
    y_human = ["alta", "alta", "baja"]    # error humano en índice 1
    result = v.classify_discrepancies(y_true, y_arm_c, y_human)
    assert 0 in result["error_modelo"]
    assert 1 in result["error_humano"]


def test_validation_requires_human_priority_annotation(tmp_path, monkeypatch):
    import importlib

    runner = importlib.import_module("experiments.run_validation")
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({
        "tipo_funcional": ["a", "b"],
        "tipo_predicho_arm_c": ["a", "b"],
        "anotacion_humana": ["a", "b"],
        "prioridad": ["alta", "baja"],
        "prioridad_predicha_arm_c": ["alta", "baja"],
        "prioridad_anotacion_humana": ["alta", ""],
    }).to_csv(processed / "test_with_preds.csv", index=False, encoding="utf-8")

    config = {
        "logging": {"level": "ERROR", "file": str(tmp_path / "run.log")},
        "project": {"seed": 42},
        "paths": {"data_processed": str(processed)},
    }
    monkeypatch.setattr(runner, "load_config", lambda _: config)
    monkeypatch.setattr(runner, "setup_logger", lambda *args, **kwargs: logging.getLogger("test"))

    assert runner.main("unused.yaml") == 1


def test_validation_writes_metrics_for_both_targets(tmp_path, monkeypatch):
    import importlib
    import json

    runner = importlib.import_module("experiments.run_validation")
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({
        "tipo_funcional": ["a", "b", "c", "a", "b", "c"],
        "tipo_predicho_arm_c": ["a", "b", "c", "b", "b", "c"],
        "anotacion_humana": ["a", "b", "c", "a", "b", "c"],
        "prioridad": ["alta", "media", "baja", "alta", "media", "baja"],
        "prioridad_predicha_arm_c": ["alta", "media", "baja", "media", "media", "baja"],
        "prioridad_anotacion_humana": ["alta", "media", "baja", "alta", "media", "baja"],
    }).to_csv(processed / "test_with_preds.csv", index=False, encoding="utf-8")

    metrics_dir = tmp_path / "metrics"
    config = {
        "logging": {"level": "ERROR", "file": str(tmp_path / "run.log")},
        "project": {"seed": 42},
        "paths": {"data_processed": str(processed), "results_metrics": str(metrics_dir)},
        "validation": {
            "kappa_weighted": True,
            "kappa_weights": "quadratic",
            "krippendorff_level": "ordinal",
        },
    }
    monkeypatch.setattr(runner, "load_config", lambda _: config)
    monkeypatch.setattr(runner, "setup_logger", lambda *args, **kwargs: logging.getLogger("test"))

    assert runner.main("unused.yaml") == 0
    metrics = json.loads((metrics_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    assert "scenarios" in metrics
    assert "priority_scenarios" in metrics
    assert "krippendorff_alpha_prioridad" in metrics