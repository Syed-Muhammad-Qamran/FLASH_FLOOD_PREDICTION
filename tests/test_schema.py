"""Prediction and model-output contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from floodwatch import schema


def _predictions(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "basin_id": ["swat_kalam"] * n,
            "timestamp_utc": pd.date_range("2022-08-26T00:00Z", periods=n, freq="h"),
            "module": [schema.MODULE_FUSED] * n,
            "prob": [0.1, 0.5, 0.9][:n],
            "tier": ["none", "moderate", "severe"][:n],
            "top_drivers": ["[]"] * n,
        }
    )


def test_valid_predictions_pass() -> None:
    out = schema.validate_predictions(_predictions())
    assert list(out.columns)[:6] == list(schema.PREDICTION_COLUMNS)
    assert out["timestamp_utc"].dt.tz is not None


def test_extra_columns_are_allowed() -> None:
    frame = _predictions().assign(compound=False)
    assert "compound" in schema.validate_predictions(frame).columns


def test_missing_column_raises() -> None:
    with pytest.raises(schema.SchemaError, match="missing required columns"):
        schema.validate_predictions(_predictions().drop(columns=["tier"]))


def test_unknown_tier_raises() -> None:
    frame = _predictions()
    frame.loc[0, "tier"] = "purple"
    with pytest.raises(schema.SchemaError, match="unknown tier"):
        schema.validate_predictions(frame)


def test_probability_bounds() -> None:
    frame = _predictions()
    frame.loc[0, "prob"] = 1.5
    with pytest.raises(schema.SchemaError, match="outside"):
        schema.validate_predictions(frame)


def test_null_timestamp_rejected() -> None:
    frame = _predictions()
    frame.loc[0, "timestamp_utc"] = pd.NaT
    with pytest.raises(schema.SchemaError, match="nulls or unparseable"):
        schema.validate_predictions(frame)


def test_tier_for_boundaries() -> None:
    cutoffs = {"none": 0.0, "low": 0.1, "moderate": 0.3, "high": 0.6, "severe": 0.8}
    assert schema.tier_for(0.05, cutoffs) == "none"
    assert schema.tier_for(0.1, cutoffs) == "low"
    assert schema.tier_for(0.3, cutoffs) == "moderate"
    assert schema.tier_for(0.6, cutoffs) == "high"
    assert schema.tier_for(0.8, cutoffs) == "severe"


def test_driver_parsing_tolerates_legacy_lists() -> None:
    assert schema.parse_drivers('["rain_3h"]') == [{"feature": "rain_3h", "weight": 0.0}]
    assert schema.parse_drivers([{"feature": "rain_3h", "weight": 2}]) == [
        {"feature": "rain_3h", "weight": 2.0}
    ]
    assert schema.parse_drivers("not json") == []


def test_drivers_to_json_keeps_top_three_by_magnitude() -> None:
    text = schema.drivers_to_json(
        [
            {"feature": "a", "weight": 0.1},
            {"feature": "b", "weight": -0.9},
            {"feature": "c", "weight": 0.5},
            {"feature": "d", "weight": 0.2},
        ]
    )
    assert [item["feature"] for item in json.loads(text)] == ["b", "c", "d"]


def test_model_file_round_trip(tmp_path: Path) -> None:
    grid = pd.DataFrame(
        {
            "basin_id": ["swat_kalam", "swat_kalam"],
            "timestamp_utc": pd.date_range("2022-08-26T00:00Z", periods=2, freq="h"),
            "split": ["validation", "validation"],
        }
    )
    frame = pd.DataFrame(
        {
            "basin_id": grid["basin_id"],
            "timestamp_utc": grid["timestamp_utc"],
            "module": [schema.MODULE_A] * 2,
            "split": ["validation", "validation"],
            "prob": [0.2, 0.7],
            "baseline_prob": [0.3, 0.6],
            "source": ["deep", "deep"],
            "top_drivers": ["[]", "[]"],
            "model_version": ["a-1", "a-1"],
        }
    )
    path = tmp_path / "predictions_a.parquet"
    schema.publish_model_file(
        frame,
        path,
        module=schema.MODULE_A,
        grid_index=grid,
        metadata={
            "schema_version": 1,
            "seed": 1,
            "data_snapshot_hash": "hash",
            "label_horizon_h": 6,
            "trained_at": "2026-01-01T00:00:00Z",
            "module": schema.MODULE_A,
        },
    )
    loaded = schema.validate_model_file(path, schema.MODULE_A, grid)
    assert len(loaded) == 2
    assert schema.read_parquet_metadata(path)["data_snapshot_hash"] == "hash"


def test_model_file_rejects_test_rows(tmp_path: Path) -> None:
    grid = pd.DataFrame(
        {
            "basin_id": ["swat_kalam"],
            "timestamp_utc": pd.date_range("2023-01-01T00:00Z", periods=1),
            "split": ["test"],
        }
    )
    frame = pd.DataFrame(
        {
            "basin_id": ["swat_kalam"],
            "timestamp_utc": grid["timestamp_utc"],
            "module": [schema.MODULE_A],
            "split": ["test"],
            "prob": [0.9],
            "baseline_prob": [0.9],
            "source": ["deep"],
            "top_drivers": ["[]"],
            "model_version": ["a-1"],
        }
    )
    path = tmp_path / "leak.parquet"
    schema.write_model_file(
        frame,
        path,
        module=schema.MODULE_A,
        metadata={
            "schema_version": 1,
            "seed": 1,
            "data_snapshot_hash": "hash",
            "label_horizon_h": 6,
            "trained_at": "2026-01-01T00:00:00Z",
            "module": schema.MODULE_A,
        },
    )
    with pytest.raises(schema.SchemaError, match="test-split rows"):
        schema.validate_model_file(path, schema.MODULE_A, grid)
