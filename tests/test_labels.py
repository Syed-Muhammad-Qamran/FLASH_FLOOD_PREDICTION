from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodwatch.labels import fit_thresholds_a, fit_thresholds_b, label_a, label_b

CFG = {
    "model_a": {
        "rain_3h_percentile": 0.95,
        "rain_3h_floor_mm": 0.0,
        "soil_moisture_percentile": 0.50,
        "tiers": {},
    },
    "model_b": {"discharge_percentile": 0.90, "tiers": {}},
}


def _table(rain_train, rain_test, discharge_train, discharge_test, soil_train=0.3, soil_test=0.9):
    n_train, n_test = len(rain_train), len(rain_test)
    ts_train = pd.date_range("2010-01-01", periods=n_train, freq="3h", tz="UTC")
    ts_test = pd.date_range("2025-01-01", periods=n_test, freq="3h", tz="UTC")
    return pd.DataFrame(
        {
            "basin_id": "b1",
            "timestamp_utc": ts_train.append(ts_test),
            "rain_mm_3h": np.concatenate([rain_train, rain_test]),
            "soil_moisture": np.concatenate(
                [np.full(n_train, soil_train), np.full(n_test, soil_test)]
            ),
            "discharge_m3s": np.concatenate([discharge_train, discharge_test]),
        }
    )


def test_thresholds_use_train_years_only():
    train = np.linspace(0, 10, 600)
    test = np.full(50, 1e6)
    table = _table(train, test, train, test)
    thr1 = fit_thresholds_a(table, CFG)

    table2 = table.copy()
    table2.loc[table2["timestamp_utc"] >= pd.Timestamp("2024-01-01", tz="UTC"), "rain_mm_3h"] = 5e6
    thr2 = fit_thresholds_a(table2, CFG)

    assert thr1["rain_mm"]["b1"] == pytest.approx(thr2["rain_mm"]["b1"])
    assert thr1["rain_mm"]["b1"] < 100


def test_label_a_compound_condition():
    table = _table(
        np.full(600, 1.0), np.array([15.0, 5.0]),
        np.full(600, 1.0), np.array([1.0, 1.0]),
        soil_train=0.2, soil_test=0.9,
    )
    thr = {
        "rain_mm": {"b1": 20.0},
        "soil_moisture": {"b1": 0.6},
        "global_rain_mm": 20.0,
        "global_soil_moisture": 0.6,
    }
    labels = label_a(table, thr)
    # 15mm is 75% of the 20mm threshold and the soil is wet -> compound positive
    assert labels.iloc[-2]
    # 5mm is too weak even on wet soil
    assert not labels.iloc[-1]


def test_label_b_uses_per_basin_percentile():
    train = np.arange(1, 601, dtype=float)
    # 90th percentile of 1..600 is ~540; test one value below and one above.
    # args: rain_train, rain_test, discharge_train, discharge_test
    table = _table(train, np.array([1.0, 1.0]), train, np.array([530.0, 560.0]))
    thr = fit_thresholds_b(table, CFG)
    labels = label_b(table, thr)
    assert not labels.iloc[-2]
    assert labels.iloc[-1]
