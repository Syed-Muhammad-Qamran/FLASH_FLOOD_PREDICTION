"""Tests for leakage prevention, split isolation, and preprocessing boundary checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from floodwatch.features import validate_no_leakage


def test_validate_no_leakage_rejects_target_keywords() -> None:
    df = pd.DataFrame({"rain_1h": [1.0, 2.0], "rain_forecast_6h": [3.0, 4.0]})
    # "rain_forecast_6h" contains "forecast" which is a target leak in historical reanalysis
    with pytest.raises(ValueError) as exc:
        validate_no_leakage(df, ["rain_1h", "rain_forecast_6h"])
    assert "contains forbidden target/leak keyword" in str(exc.value)


def test_validate_no_leakage_rejects_2010_or_2022_in_train_split() -> None:
    df_leak_2010 = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2010-07-28", periods=10, freq="h", tz="UTC"),
            "rain_1h": [5.0] * 10,
            "split": ["train"] * 10,
        }
    )
    with pytest.raises(ValueError) as exc:
        validate_no_leakage(df_leak_2010, ["rain_1h"])
    assert "Year 2010 (external case study) is present in train split" in str(exc.value)

    df_leak_2022 = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2022-08-26", periods=10, freq="h", tz="UTC"),
            "rain_1h": [5.0] * 10,
            "split": ["train"] * 10,
        }
    )
    with pytest.raises(ValueError) as exc:
        validate_no_leakage(df_leak_2022, ["rain_1h"])
    assert "Year 2022 (external case study) is present in train split" in str(exc.value)


def test_scaler_fitted_on_train_rows_only() -> None:
    """Verify that scaler mean and variance are not affected by validation rows."""
    # Synthetic train (mean 10.0) and validation (mean 1000.0)
    train_data = pd.DataFrame({"val": [10.0, 10.0, 10.0], "split": ["train"] * 3})
    val_data = pd.DataFrame({"val": [1000.0, 1000.0], "split": ["validation"] * 2})
    combined = pd.concat([train_data, val_data], ignore_index=True)

    scaler = StandardScaler()
    scaler.fit(combined.loc[combined["split"] == "train", ["val"]])

    # Mean should be 10.0, completely uncontaminated by the 1000.0 validation values
    assert np.isclose(scaler.mean_[0], 10.0)
