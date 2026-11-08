"""Train-only thresholds, the D-01 guard and provenance."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from floodwatch import schema, thresholds


def _frame(n: int = 400, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0, 1, n)
    label = (prob + rng.normal(0, 0.25, n) > 0.55).astype(int)
    return pd.DataFrame(
        {
            "basin_id": ["swat_kalam"] * n,
            "timestamp_utc": pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC"),
            "prob": prob,
            "label": label,
            "split": ["train"] * n,
        }
    )


def test_fit_produces_strictly_increasing_cutoffs() -> None:
    fit = thresholds.fit_tier_cutoffs(_frame())
    values = [fit["tiers"][key] for key in thresholds.ALL_TIER_KEYS]
    assert values[0] == 0.0
    assert all(a < b for a, b in zip(values, values[1:], strict=False))
    assert fit["rows"] == 400


def test_fit_ignores_validation_rows() -> None:
    train = _frame()
    poisoned = train.copy()
    poisoned["split"] = "validation"
    poisoned["label"] = 1 - poisoned["label"]
    base = thresholds.fit_tier_cutoffs(train)
    mixed = thresholds.fit_tier_cutoffs(pd.concat([train, poisoned], ignore_index=True))
    assert base["tiers"] == mixed["tiers"]


def test_empty_sample_falls_back_safely() -> None:
    fit = thresholds.fit_tier_cutoffs(_frame().iloc[0:0])
    assert fit["rows"] == 0
    assert fit["tiers"]["high"] == 0.30


def test_validation_guard_flags_drift_and_never_refits() -> None:
    train_metrics = {"pod": 0.80, "far": 0.30, "csi": 0.6, "rows": 100}
    ok = thresholds.validation_guard(train_metrics, {"pod": 0.85, "far": 0.35})
    assert ok["pass"] is True
    bad = thresholds.validation_guard(train_metrics, {"pod": 0.50, "far": 0.60})
    assert bad["pass"] is False
    assert bad["action"] == "escalate_joint_decision"
    assert bad["pod_delta"] == pytest.approx(-0.30)


def test_metrics_at_cutoff_matches_confusion_counts() -> None:
    frame = pd.DataFrame({"prob": [0.1, 0.5, 0.9], "label": [0, 0, 1]})
    block = thresholds.metrics_at_cutoff(frame, 0.5)
    assert block["pod"] == 1.0
    assert block["far"] == 0.5


def test_config_round_trip_and_hand_edit_detection(tmp_path) -> None:
    modules = {
        schema.MODULE_A: {
            "tiers": {"none": 0.0, "low": 0.1, "moderate": 0.2, "high": 0.4, "severe": 0.7}
        },
        schema.MODULE_B: {
            "tiers": {"none": 0.0, "low": 0.1, "moderate": 0.2, "high": 0.4, "severe": 0.7}
        },
        schema.MODULE_FUSED: {
            "tiers": {"none": 0.0, "low": 0.1, "moderate": 0.2, "high": 0.4, "severe": 0.7}
        },
    }
    config = thresholds.build_config(modules, data_snapshot_hash="hash")
    thresholds.validate_thresholds(config)
    path = thresholds.save_thresholds(config, tmp_path / "thresholds.yaml")
    loaded = thresholds.load_thresholds(path)
    assert loaded["provenance_fingerprint"] == config["provenance_fingerprint"]

    edited = copy.deepcopy(config)
    edited["modules"][schema.MODULE_A]["tiers"]["high"] = 0.5
    with pytest.raises(ValueError, match="provenance_fingerprint mismatch"):
        thresholds.validate_thresholds(edited)


def test_non_monotonic_cutoffs_rejected() -> None:
    modules = {
        schema.MODULE_A: {
            "tiers": {"none": 0.0, "low": 0.5, "moderate": 0.2, "high": 0.4, "severe": 0.7}
        },
    }
    config = thresholds.build_config(modules, data_snapshot_hash="hash")
    with pytest.raises(ValueError, match="not strictly increasing"):
        thresholds.validate_thresholds(config)


def test_committed_config_is_fitted_on_train_and_valid() -> None:
    config = thresholds.load_thresholds()
    thresholds.validate_thresholds(config)
    assert config["fitted_on"] == "train"
    assert config["data_snapshot_hash"]
    assert set(config["modules"]) == {schema.MODULE_A, schema.MODULE_B, schema.MODULE_FUSED}
