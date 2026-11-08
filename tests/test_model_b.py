"""Model B: frozen size limits, degree-day baseline and a small CPU train."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from floodwatch import schema
from floodwatch.features import FEATURE_COLUMNS_B
from floodwatch.models import model_b

torch = pytest.importorskip("torch")


def _features(n: int = 140, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2022-01-01T00:00Z", periods=n, freq="h")
    data = {column: rng.uniform(0, 1, n) for column in FEATURE_COLUMNS_B}
    data.update(
        {
            "basin_id": ["swat_kalam"] * n,
            "timestamp_utc": stamps,
            "label": (np.arange(n) % 7 == 0).astype(int),
            "degree_day_melt": rng.uniform(0, 10, n),
            "rain_on_snow": rng.uniform(0, 5, n),
            "snow_depth_mm": rng.uniform(0, 100, n),
            "temp_c": rng.uniform(-5, 15, n),
        }
    )
    return pd.DataFrame(data)


def _summary() -> pd.DataFrame:
    """Four hours of rolling inputs for one basin."""
    return pd.DataFrame(
        {
            "basin_id": ["swat_kalam"] * 4,
            "timestamp_utc": pd.date_range("2022-01-01T00:00Z", periods=4, freq="h"),
            "melt_3h": [0.0, 0.0, 3.0, 3.0],
            "melt_24h": [0.0, 0.0, 2.0, 2.0],
            "ros_3h": [0.0, 0.0, 1.0, 1.0],
            "snow_depth_mm": [50.0] * 4,
            "temp_c": [1.0] * 4,
        }
    )


def test_frozen_size_limits() -> None:
    assert model_b.MAX_LSTM_LAYERS == 1
    assert model_b.MIN_UNITS == 32
    assert model_b.MAX_UNITS == 64
    assert 32 <= model_b.TrainConfig().hidden_units <= 64
    with pytest.raises(ValueError, match="hidden_units"):
        model_b.TrainConfig(hidden_units=16)


def test_baseline_inputs_compute_rolling_means() -> None:
    frame = _features(6)
    summary = model_b.baseline_inputs(frame)
    melt = summary["melt_3h"].to_numpy()
    raw = frame["degree_day_melt"].to_numpy()
    assert melt[2] == pytest.approx(raw[:3].mean())


def test_fit_baseline_uses_train_rows_only() -> None:
    summary = _summary()
    mask = pd.Series([True, True, False, False])
    clean = model_b.fit_baseline(summary, train_mask=mask)
    poisoned = summary.copy()
    poisoned.loc[2:, ["melt_3h", "ros_3h", "snow_depth_mm"]] = 1e9
    dirty = model_b.fit_baseline(poisoned, train_mask=mask)
    assert clean.melt_mean == pytest.approx(dirty.melt_mean)
    assert clean.ros_mean == pytest.approx(dirty.ros_mean)


def test_baseline_probability_is_monotonic_in_melt() -> None:
    summary = _summary()
    state = model_b.BaselineState(melt_mean=0.0, melt_std=1.0, ros_mean=0.0, ros_std=1.0)
    probs = model_b.baseline_probability(summary, state)
    assert probs[0] < probs[2]
    assert np.all((probs >= 0) & (probs <= 1))


def test_probability_never_escapes_the_unit_interval() -> None:
    state = model_b.BaselineState(intercept=1e6)
    probs = model_b.baseline_probability(_summary(), state)
    assert np.all((probs >= 0) & (probs <= 1))


def _grid(features: pd.DataFrame) -> pd.DataFrame:
    stamps = pd.to_datetime(features["timestamp_utc"], utc=True)
    cutoff = stamps.max() - pd.Timedelta(hours=30)
    grid = pd.DataFrame(
        {
            "basin_id": features["basin_id"].astype(str),
            "timestamp_utc": stamps,
            "split": np.where(stamps >= cutoff, "validation", "train"),
        }
    )
    assert (grid["split"] == "train").any()
    assert (grid["split"] == "validation").any()
    return grid.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)


def test_quick_train_produces_a_valid_model_file(tmp_path: Path) -> None:
    features = _features(140)
    grid = _grid(features)
    quick = model_b.TrainConfig(seed=42, hidden_units=32)
    trained = model_b.train_model_b(features, grid, cfg=quick, quick=True)
    assert trained["train_rows"] > 0
    assert trained["validation_rows"] > 0
    assert 0.0 <= trained["best_val_pr_auc"] <= 1.0

    summary = model_b.baseline_inputs(features)
    state = model_b.fit_baseline(
        summary, train_mask=(grid["split"] == "train").reset_index(drop=True)
    )
    predictions = model_b.predict_table(
        features,
        grid,
        model=trained["model"],
        state=state,
        cfg=trained["cfg"],
        norm=trained["norm"],
        model_version="B_snowmelt_lstm-test",
        snapshot_hash="hash",
    )
    assert len(predictions) == len(grid)
    assert set(predictions["split"]) <= {"train", "validation"}
    assert predictions["prob"].between(0, 1).all()
    assert predictions["baseline_prob"].between(0, 1).all()

    path = tmp_path / "predictions_b.parquet"
    schema.publish_model_file(
        predictions,
        path,
        module=schema.MODULE_B,
        grid_index=grid,
        metadata={
            "schema_version": 1,
            "seed": 42,
            "data_snapshot_hash": "hash",
            "label_horizon_h": 6,
            "trained_at": "2026-01-01T00:00:00Z",
            "module": schema.MODULE_B,
        },
    )
    loaded = schema.validate_model_file(path, schema.MODULE_B, grid)
    assert len(loaded) == len(grid)


def test_training_is_deterministic_for_a_fixed_seed() -> None:
    features = _features(140)
    grid = _grid(features)
    cfg = model_b.TrainConfig(seed=7, hidden_units=32)
    first = model_b.train_model_b(features, grid, cfg=cfg, quick=True)
    second = model_b.train_model_b(features, grid, cfg=cfg, quick=True)
    assert first["best_val_pr_auc"] == pytest.approx(second["best_val_pr_auc"])


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    features = _features(140)
    grid = _grid(features)
    trained = model_b.train_model_b(
        features, grid, cfg=model_b.TrainConfig(seed=42, hidden_units=32), quick=True
    )
    summary = model_b.baseline_inputs(features)
    state = model_b.fit_baseline(
        summary, train_mask=(grid["split"] == "train").reset_index(drop=True)
    )
    path = tmp_path / "model_b.pt"
    model_b.save_checkpoint(
        path,
        model=trained["model"],
        state=state,
        cfg=trained["cfg"],
        norm=trained["norm"],
        metadata={"module": schema.MODULE_B},
    )
    model, loaded_state, cfg, norm, checkpoint = model_b.load_checkpoint(path)
    assert cfg.seed == 42
    assert loaded_state.melt_mean == pytest.approx(state.melt_mean)
    assert norm["mean"] == pytest.approx(trained["norm"]["mean"])
    assert checkpoint["metadata"]["module"] == schema.MODULE_B
    assert len(norm["mean"]) == len(FEATURE_COLUMNS_B)
