"""Fusion rules, invariants and readable failure messages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodwatch import fusion, schema

CUTOFFS = {"none": 0.0, "low": 0.05, "moderate": 0.15, "high": 0.40, "severe": 0.70}


def _thresholds() -> dict:
    return {
        "schema_version": 1,
        "fitted_on": "train",
        "modules": {
            schema.MODULE_A: {"tiers": dict(CUTOFFS)},
            schema.MODULE_B: {"tiers": dict(CUTOFFS)},
            schema.MODULE_FUSED: {"tiers": dict(CUTOFFS)},
        },
    }


def _frame(module: str, probs: list[float], drivers: str = "[]", split: str = "validation"):
    timestamps = pd.date_range("2022-08-26T00:00Z", periods=len(probs), freq="h")
    return pd.DataFrame(
        {
            "basin_id": ["swat_kalam"] * len(probs),
            "timestamp_utc": timestamps,
            "module": [module] * len(probs),
            "split": [split] * len(probs),
            "prob": probs,
            "baseline_prob": [0.1] * len(probs),
            "source": ["deep"] * len(probs),
            "top_drivers": [drivers] * len(probs),
            "model_version": [f"{module}-v1"] * len(probs),
        }
    )


def test_max_rule_and_tiers() -> None:
    a = _frame(schema.MODULE_A, [0.2, 0.9])
    b = _frame(schema.MODULE_B, [0.8, 0.1])
    fused = fusion.fuse_frames(a, b, thresholds_config=_thresholds())
    assert list(fused["prob"]) == [0.8, 0.9]
    assert list(fused["tier_a"]) == ["moderate", "severe"]
    assert list(fused["tier_b"]) == ["severe", "low"]
    assert set(fused["module"]) == {schema.MODULE_FUSED}
    schema.validate_predictions(fused)


def test_compound_requires_both_above_their_high_cutoff() -> None:
    a = _frame(schema.MODULE_A, [0.5, 0.5, 0.2])
    b = _frame(schema.MODULE_B, [0.5, 0.2, 0.5])
    fused = fusion.fuse_frames(a, b, thresholds_config=_thresholds(), rain_on_snow_escalation=False)
    assert list(fused["compound"]) == [True, False, False]


def test_rain_on_snow_escalates_a_compound_event() -> None:
    drivers = '[{"feature": "rain_on_snow", "weight": 3.0}]'
    a = _frame(schema.MODULE_A, [0.5], drivers=drivers)
    b = _frame(schema.MODULE_B, [0.5], drivers=drivers)
    fused = fusion.fuse_frames(a, b, thresholds_config=_thresholds())
    assert bool(fused.iloc[0]["compound"]) is True
    assert bool(fused.iloc[0]["escalated"]) is True
    assert fused.iloc[0]["tier"] == "severe"


def test_escalation_can_be_disabled() -> None:
    drivers = '[{"feature": "rain_on_snow", "weight": 3.0}]'
    a = _frame(schema.MODULE_A, [0.5], drivers=drivers)
    b = _frame(schema.MODULE_B, [0.5], drivers=drivers)
    fused = fusion.fuse_frames(a, b, thresholds_config=_thresholds(), rain_on_snow_escalation=False)
    assert bool(fused.iloc[0]["escalated"]) is False
    assert fused.iloc[0]["tier"] == "high"


def test_noisy_or_is_the_union_probability() -> None:
    result = fusion.noisy_or(np.array([0.2]), np.array([0.5]))
    assert result[0] == pytest.approx(0.6)


def test_drivers_are_merged_and_ranked() -> None:
    a = _frame(
        schema.MODULE_A,
        [0.5],
        drivers='[{"feature": "rain_3h", "weight": 2.0}, {"feature": "soil_moisture", "weight": 1.0}]',
    )
    b = _frame(
        schema.MODULE_B,
        [0.5],
        drivers='[{"feature": "rain_3h", "weight": 1.0}, {"feature": "snow_depth_mm", "weight": 4.0}]',
    )
    fused = fusion.fuse_frames(a, b, thresholds_config=_thresholds(), rain_on_snow_escalation=False)
    drivers = schema.parse_drivers(fused.iloc[0]["top_drivers"])
    assert drivers[0] == {"feature": "snow_depth_mm", "weight": 4.0}
    assert {"feature": "rain_3h", "weight": 3.0} in drivers


def test_mismatched_grid_raises_a_readable_diff() -> None:
    a = _frame(schema.MODULE_A, [0.5, 0.6])
    b = _frame(schema.MODULE_B, [0.5])
    with pytest.raises(fusion.FusionError, match="missing"):
        fusion.fuse_frames(a, b, thresholds_config=_thresholds())


def test_mismatched_snapshot_hash_raises() -> None:
    a = _frame(schema.MODULE_A, [0.5])
    b = _frame(schema.MODULE_B, [0.5])
    with pytest.raises(fusion.FusionError, match="data_snapshot_hash mismatch"):
        fusion.fuse_frames(
            a,
            b,
            thresholds_config=_thresholds(),
            meta_a={"data_snapshot_hash": "aaa"},
            meta_b={"data_snapshot_hash": "bbb"},
        )


def test_wrong_module_constant_raises() -> None:
    a = _frame(schema.MODULE_B, [0.5])
    b = _frame(schema.MODULE_B, [0.5])
    with pytest.raises(fusion.FusionError, match="module must be"):
        fusion.fuse_frames(a, b, thresholds_config=_thresholds())


def test_fusion_does_not_import_models_or_torch() -> None:
    source = fusion.__file__ or ""
    text = open(source, encoding="utf-8").read()
    for banned in (
        "import torch",
        "from torch",
        "import sklearn",
        "from sklearn",
        "from .models",
        "from ..models",
    ):
        assert banned not in text
