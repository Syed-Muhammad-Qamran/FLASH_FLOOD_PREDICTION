from __future__ import annotations

import json

import pandas as pd
import pytest

from floodwatch.schema import (
    SchemaError,
    empty_predictions,
    parse_drivers,
    tier_for,
    validate_predictions,
)


def _row(**overrides):
    row = {
        "basin_id": "swat_chakdara",
        "timestamp_utc": pd.Timestamp("2022-08-26T18:00Z"),
        "module": "A",
        "prob": 0.72,
        "tier": "high",
        "top_drivers": json.dumps(["rain_3h=44mm"]),
    }
    row.update(overrides)
    return row


def test_valid_row_passes():
    df = validate_predictions(pd.DataFrame([_row()]))
    assert str(df["timestamp_utc"].dtype).endswith("UTC]")
    assert df["prob"].iloc[0] == 0.72


def test_naive_timestamp_rejected():
    with pytest.raises(SchemaError, match="timezone-naive"):
        validate_predictions(pd.DataFrame([_row(timestamp_utc=pd.Timestamp("2022-08-26 18:00"))]))


def test_prob_out_of_range_rejected():
    with pytest.raises(SchemaError, match="outside"):
        validate_predictions(pd.DataFrame([_row(prob=1.4)]))


def test_unknown_tier_rejected():
    with pytest.raises(SchemaError, match="unknown tier"):
        validate_predictions(pd.DataFrame([_row(tier="catastrophic")]))


def test_missing_column_rejected():
    df = pd.DataFrame([_row()]).drop(columns=["top_drivers"])
    with pytest.raises(SchemaError, match="missing required"):
        validate_predictions(df)


def test_drivers_accepts_list_and_roundtrips():
    df = validate_predictions(pd.DataFrame([_row(top_drivers=["a", "b"])]))
    assert parse_drivers(df["top_drivers"].iloc[0]) == ["a", "b"]


def test_tier_for_is_monotonic():
    cut = {"low": 0.2, "moderate": 0.5, "high": 0.7, "severe": 0.9}
    assert tier_for(0.1, cut) == "none"
    assert tier_for(0.2, cut) == "low"
    assert tier_for(0.55, cut) == "moderate"
    assert tier_for(0.95, cut) == "severe"


def test_empty_predictions_has_contract_columns():
    df = empty_predictions()
    assert set(df.columns) == {
        "basin_id",
        "timestamp_utc",
        "module",
        "prob",
        "tier",
        "top_drivers",
    }
