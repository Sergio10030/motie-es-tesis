"""Pruebas unitarias para priorización asimétrica y abstención."""
import numpy as np
import pytest

from src.decision.abstention import AbstentionModule
from src.decision.asymmetric_priority import AsymmetricPrioritizer


def test_cost_asymmetric():
    prioritizer = AsymmetricPrioritizer(
        priority_levels=["alta", "media", "baja"],
        critical_level="alta",
        cost_alpha=5.0, cost_beta=1.0, cost_gamma=2.0, cost_delta=1.0,
    )
    y_true = ["alta", "media", "baja"]
    y_pred = ["media", "media", "baja"]  # FN crítico
    cost = prioritizer.compute_cost(y_true, y_pred)
    assert cost >= 5.0


def test_calibrate_thresholds():
    prioritizer = AsymmetricPrioritizer(
        priority_levels=["alta", "media", "baja"],
        critical_level="alta",
    )
    y_true = ["alta", "media", "baja", "alta", "media"]
    y_proba = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.15, 0.8],
        [0.85, 0.1, 0.05],
        [0.2, 0.7, 0.1],
    ])
    classes = ["alta", "media", "baja"]
    thresholds = prioritizer.calibrate(y_true, y_proba, classes)
    assert "alta" in thresholds


def test_abstention_low_confidence():
    mod = AbstentionModule(confidence_threshold=0.70)
    assert mod.should_abstain(0.5) is True
    assert mod.should_abstain(0.9) is False


def test_abstention_discrepancy():
    mod = AbstentionModule(confidence_threshold=0.70, stage_discrepancy_threshold=0.30)
    assert mod.should_abstain(0.9, stage_discrepancy=0.5) is True
    assert mod.should_abstain(0.9, stage_discrepancy=0.1) is False