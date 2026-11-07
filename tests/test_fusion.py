from __future__ import annotations

import json

import pandas as pd

from floodwatch.fusion import fuse

CFG = {
    "model_a": {"tiers": {"low": 0.2, "moderate": 0.5, "high": 0.7, "severe": 0.9}},
    "model_b": {"tiers": {"low": 0.2, "moderate": 0.5, "high": 0.7, "severe": 0.9}},
    "fusion": {"rule": "max", "compound_escalate": True, "model_b_max_age_hours": 24},
    "alerts": {},
}


def _preds(module, times, probs, drivers=None):
    return pd.DataFrame(
        {
            "basin_id": "swat_chakdara",
            "timestamp_utc": pd.to_datetime(times, utc=True),
            "module": module,
            "prob": probs,
            "tier": ["none"] * len(times),
            "top_drivers": drivers or [json.dumps(["x"])] * len(times),
        }
    )


def test_higher_probability_wins():
    times = pd.date_range("2022-08-26T00:00Z", periods=4, freq="3h")
    a = _preds("A", times, [0.1, 0.2, 0.3, 0.4])
    b = _preds("B", ["2022-08-26T00:00Z"], [0.8])
    fused = fuse(a, b, CFG)
    assert len(fused) == 4
    assert fused["prob"].max() == 0.8
    assert all("melt" in d for d in fused["top_drivers"])


def test_compound_escalates_one_tier():
    times = pd.date_range("2022-08-26T00:00Z", periods=2, freq="3h")
    a = _preds("A", times, [0.6, 0.6])
    b = _preds("B", ["2022-08-26T00:00Z"], [0.6])
    fused = fuse(a, b, CFG)
    assert fused["tier"].iloc[0] == "high"  # moderate escalated
    assert "compound" in fused["top_drivers"].iloc[0].lower()


def test_stale_model_b_is_not_carried_forward():
    times = pd.date_range("2022-08-26T00:00Z", periods=3, freq="3h")
    a = _preds("A", times, [0.1, 0.1, 0.1])
    b = _preds("B", ["2022-08-24T00:00Z"], [0.95])  # 48h old
    fused = fuse(a, b, CFG)
    assert fused["prob"].max() == 0.1


def test_noisy_or_rule():
    cfg = dict(CFG)
    cfg["fusion"] = {**CFG["fusion"], "rule": "noisy_or"}
    times = pd.date_range("2022-08-26T00:00Z", periods=1, freq="3h")
    a = _preds("A", times, [0.5])
    b = _preds("B", ["2022-08-26T00:00Z"], [0.5])
    fused = fuse(a, b, cfg)
    assert abs(fused["prob"].iloc[0] - 0.75) < 1e-9
