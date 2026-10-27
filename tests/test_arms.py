"""Pruebas unitarias para los tres brazos."""
import pandas as pd
import pytest

from src.arms.arm_a_classic import ArmAClassic
from src.arms.arm_c_fewshot import ArmCFewShot, LLMClient


@pytest.fixture
def sample_data():
    texts = [
        "estudiante golpeó a compañero en el recreo",
        "vidrio roto en el aula de clases",
        "estudiante se desmayó en educación física",
        "insultos entre estudiantes en el patio",
        "alumno no trajo tarea por tercera vez",
        "pelea en el baño con sangre",
        "daño a la puerta del laboratorio",
        "fiebre alta de estudiante en clase",
        "acoso verbal reiterado a una compañera",
        "llegó tarde a clases sin justificación",
    ]
    types = [
        "conflicto_estudiantes", "daño_infraestructura", "incidente_salud",
        "conflicto_estudiantes", "falta_disciplinaria", "acoso",
        "daño_infraestructura", "incidente_salud", "acoso", "falta_disciplinaria",
    ]
    return texts, types


def test_arm_a_fit_predict(sample_data):
    texts, types = sample_data
    model = ArmAClassic(max_features=100, max_iter=100)
    model.fit(texts, types)
    preds = model.predict(texts)
    assert "tipo_predicho" in preds
    assert len(preds["tipo_predicho"]) == len(texts)


def test_arm_a_evaluate(sample_data):
    texts, types = sample_data
    model = ArmAClassic(max_features=100, max_iter=100)
    model.fit(texts, types)
    report = model.evaluate(texts, types)
    assert "f1_macro_tipo" in report
    assert 0.0 <= report["f1_macro_tipo"] <= 1.0


def test_arm_a_multitask_predicts_priority(sample_data):
    texts, types = sample_data
    priorities = ["alta", "baja", "media", "media", "baja"] * 2
    model = ArmAClassic(max_features=100, max_iter=100, multi_task=True)
    model.fit(texts, types, priorities)
    predictions = model.predict(texts)
    assert "tipo_predicho" in predictions
    assert "prioridad_predicha" in predictions
    assert len(predictions["prioridad_predicha"]) == len(texts)


def test_arm_b_runner_trains_both_targets(tmp_path, monkeypatch):
    import importlib
    import json

    import yaml

    runner = importlib.import_module("experiments.run_arm_b")
    fit_labels = []

    class FakeArmB:
        def __init__(self, **kwargs):
            self.history = []
            self.prediction = None

        def fit(self, train_texts, train_labels, val_texts, val_labels):
            fit_labels.append(list(train_labels))
            self.prediction = train_labels[0]
            self.history = [{"epoch": 1, "loss": 0.1}]
            return self

        def predict(self, texts):
            return {"tipo_predicho": [self.prediction] * len(texts)}

    monkeypatch.setattr(runner, "ArmBFinetuning", FakeArmB)
    processed = tmp_path / "processed"
    processed.mkdir()
    data = pd.DataFrame({
        "text_clean": ["texto uno", "texto dos", "texto tres", "texto cuatro"],
        "tipo_funcional": ["tipo_a", "tipo_b", "tipo_a", "tipo_b"],
        "prioridad": ["alta", "baja", "alta", "baja"],
    })
    for split in ["train", "val", "test"]:
        data.to_csv(processed / f"{split}.csv", index=False, encoding="utf-8")

    metrics_dir = tmp_path / "metrics"
    config = {
        "logging": {"level": "ERROR", "file": str(tmp_path / "run.log")},
        "project": {"seed": 42},
        "paths": {
            "data_processed": str(processed),
            "checkpoints": str(tmp_path / "checkpoints"),
            "results_metrics": str(metrics_dir),
        },
        "arm_b": {
            "default_model": "fake/model",
            "learning_rate": 2e-5,
            "batch_size": 2,
            "weight_decay": 0.01,
            "num_epochs": 1,
            "max_length": 16,
            "warmup_ratio": 0.1,
            "class_weighting": True,
            "gradient_clip": 1.0,
            "early_stopping_patience": 1,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    runner.main(str(config_path))

    assert fit_labels == [data["tipo_funcional"].tolist(), data["prioridad"].tolist()]
    metrics = json.loads((metrics_dir / "arm_b_metrics.json").read_text(encoding="utf-8"))
    assert "f1_macro" in metrics
    assert "f1_macro_prioridad" in metrics


def test_arm_c_simulated(sample_data):
    texts, types = sample_data
    client = LLMClient(provider="openai", api_key_env="NON_EXISTENT_KEY")
    model = ArmCFewShot(client=client, k_shot=2)
    model.set_taxonomy(sorted(set(types)), ["alta", "media", "baja"])
    df = pd.DataFrame({
        "texto": texts, "tipo_funcional": types,
        "prioridad": ["media"] * len(texts),
        "densidad_jerga": [0.1] * len(texts),
        "densidad_admin": [0.1] * len(texts),
    })
    model.select_examples(df)
    result = model.classify_one(texts[0])
    assert "tipo_predicho" in result
    assert "prioridad_predicha" in result
    assert "prompt_etapa1" in result
    assert "prompt_etapa2" in result


def test_arm_c_handles_invalid():
    client = LLMClient(provider="openai", api_key_env="NON_EXISTENT_KEY")
    model = ArmCFewShot(client=client, k_shot=1)
    model.set_taxonomy(["tipo_a", "tipo_b"], ["alta", "media", "baja"])
    # Simular respuesta inválida
    label = model._parse_response("respuesta sin etiqueta válida", ["tipo_a", "tipo_b"])
    assert label == model.fallback_label