"""Fused table -> typed domain objects, without changing probabilities."""

from __future__ import annotations

import json

import pandas as pd

from floodwatch import interop


def _fused(rows: list[dict]) -> pd.DataFrame:
    base = {
        "module": "fused",
        "split": "validation",
        "compound": False,
        "prob_a": None,
        "prob_b": None,
        "escalated": False,
    }
    out = []
    for row in rows:
        out.append({**base, **row})
    return pd.DataFrame(out)


def test_probabilities_are_not_modified() -> None:
    frame = _fused(
        [
            {
                "basin_id": "swat_kalam",
                "timestamp_utc": "2022-08-27T00:00:00Z",
                "prob": 0.6234,
                "tier": "severe",
                "top_drivers": json.dumps([{"feature": "rain_3h", "weight": 2.0}]),
                "prob_a": 0.4,
                "prob_b": 0.6234,
            }
        ]
    )
    results = interop.to_risk_results(frame)
    assert results[0].score == 0.6234
    assert results[0].prob_a == 0.4
    assert results[0].prob_b == 0.6234
    assert results[0].tier == "severe"


def test_risk_frame_sorts_worst_first() -> None:
    frame = _fused(
        [
            {
                "basin_id": "swat_kalam",
                "timestamp_utc": "2022-08-27T00:00:00Z",
                "prob": 0.1,
                "tier": "none",
                "top_drivers": "[]",
            },
            {
                "basin_id": "swat_madyan",
                "timestamp_utc": "2022-08-27T00:00:00Z",
                "prob": 0.9,
                "tier": "severe",
                "top_drivers": "[]",
            },
            {
                "basin_id": "chitral_drosh",
                "timestamp_utc": "2022-08-27T00:00:00Z",
                "prob": 0.5,
                "tier": "high",
                "top_drivers": "[]",
            },
        ]
    )
    results = interop.to_risk_results(frame)
    table = interop.risk_frame(results)
    assert list(table["area_id"])[:3] == ["swat_madyan", "chitral_drosh", "swat_kalam"]
    assert list(table["level"])[:3] == ["RED", "ORANGE", "GREEN"]
    # Unlisted catalog basins are surfaced GREEN rather than dropped.
    assert (table["level"][3:] == "GREEN").all()


def test_compound_and_escalation_reasons_are_explicit() -> None:
    frame = _fused(
        [
            {
                "basin_id": "swat_kalam",
                "timestamp_utc": "2022-08-27T00:00:00Z",
                "prob": 0.9,
                "tier": "severe",
                "compound": True,
                "escalated": True,
                "top_drivers": json.dumps([{"feature": "rain_on_snow", "weight": 3.0}]),
            }
        ]
    )
    result = interop.to_risk_results(frame)[0]
    assert result.compound is True
    assert result.escalated is True
    assert any("compound" in reason for reason in result.reasons)
    assert any("rain-on-snow" in reason for reason in result.reasons)


def test_missing_basins_are_filled_green() -> None:
    frame = _fused(
        [
            {
                "basin_id": "swat_kalam",
                "timestamp_utc": "2022-08-27T00:00:00Z",
                "prob": 0.9,
                "tier": "severe",
                "top_drivers": "[]",
            }
        ]
    )
    results = interop.to_risk_results(frame)
    assert len(results) == 8
    missing = [item for item in results if item.area_id != "swat_kalam"]
    assert all(item.tier == "none" for item in missing)


def test_latest_row_per_basin_is_used() -> None:
    frame = _fused(
        [
            {
                "basin_id": "swat_kalam",
                "timestamp_utc": "2022-08-27T00:00:00Z",
                "prob": 0.2,
                "tier": "moderate",
                "top_drivers": "[]",
            },
            {
                "basin_id": "swat_kalam",
                "timestamp_utc": "2022-08-27T01:00:00Z",
                "prob": 0.9,
                "tier": "severe",
                "top_drivers": "[]",
            },
        ]
    )
    result = interop.to_risk_results(frame)
    assert result[0].score == 0.9
    assert result[0].tier == "severe"


def test_catalog_supplies_names_and_districts() -> None:
    areas = interop.areas()
    assert areas["swat_kalam"].name
    assert areas["swat_kalam"].district
