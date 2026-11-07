"""Feature engineering shared by the baselines and the deep models.

All rolling windows look backwards only. Labels are shifted forward separately
(see labels.forward_label), so a row at time t contains no information from
t+1 onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEGREE_DAY_MM_PER_C_PER_DAY = 2.5


def _per_basin(df: pd.DataFrame, col: str, func) -> pd.Series:
    return df.groupby("basin_id", group_keys=False)[col].transform(func)


def _forward_sum(s: pd.Series, steps: int) -> pd.Series:
    """Sum of the next `steps` values, excluding the current one.

    This is a forecast feature. In live mode the next rows are Open-Meteo
    forecast rows; in the demo they are a noisy forecast of the true series.
    """
    rev = s.astype(float).iloc[::-1]
    rolled = rev.rolling(steps, min_periods=1).sum().iloc[::-1]
    return rolled.shift(-1)


def melt_index(df: pd.DataFrame) -> pd.Series:
    """Degree-day snowmelt in mm per 3h, scaled by basin snow fraction."""
    temp = df["temp_c"].clip(lower=0.0)
    steps_per_day = 8.0
    return (
        temp
        * DEGREE_DAY_MM_PER_C_PER_DAY
        / steps_per_day
        * df["snow_fraction"].astype(float)
    )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    out = df.copy()
    doy = ts.dt.dayofyear.to_numpy()
    hour = ts.dt.hour.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def build_features_a(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Cloudburst features: short-window rainfall, wetness, seasonality."""
    out = add_time_features(df)
    g = out.groupby("basin_id", group_keys=False)

    rain = out["rain_mm_3h"]
    out["rain_6h"] = g["rain_mm_3h"].transform(lambda s: s.rolling(2, min_periods=1).sum())
    out["rain_12h"] = g["rain_mm_3h"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    out["rain_24h"] = g["rain_mm_3h"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    out["rain_72h"] = g["rain_mm_3h"].transform(lambda s: s.rolling(24, min_periods=1).sum())
    out["rain_30d_mean"] = g["rain_mm_3h"].transform(
        lambda s: s.rolling(240, min_periods=24).mean()
    )
    out["rain_anomaly"] = rain / (out["rain_30d_mean"] + 0.05)
    out["soil_lag_9h"] = g["soil_moisture"].transform(lambda s: s.shift(3))
    out["soil_trend"] = out["soil_moisture"] - out["soil_lag_9h"]
    out["rain_x_soil"] = out["rain_24h"] * out["soil_moisture"]

    # Forecast features. Without these the model can only nowcast and lead
    # time collapses to zero.
    fcst_rain = out["rain_fcst_mm_3h"] if "rain_fcst_mm_3h" in out else out["rain_mm_3h"]
    out["rain_next_3h"] = fcst_rain.groupby(out["basin_id"]).transform(lambda s: s.shift(-1))
    out["rain_next_12h"] = fcst_rain.groupby(out["basin_id"]).transform(
        lambda s: _forward_sum(s, 4)
    )
    out["forecast_excess"] = out["rain_next_3h"] / (out["rain_30d_mean"] + 0.05)

    cols = [
        "rain_mm_3h",
        "rain_6h",
        "rain_12h",
        "rain_24h",
        "rain_72h",
        "rain_anomaly",
        "rain_next_3h",
        "rain_next_12h",
        "forecast_excess",
        "soil_moisture",
        "soil_trend",
        "rain_x_soil",
        "temp_c",
        "doy_sin",
        "doy_cos",
        "hour_sin",
        "hour_cos",
    ]
    return out, cols


def build_features_b(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Melt features: degree-day accumulation, recent flow, seasonality."""
    out = add_time_features(df)
    out["melt_index"] = melt_index(out)
    g = out.groupby("basin_id", group_keys=False)

    out["melt_3d"] = g["melt_index"].transform(lambda s: s.rolling(24, min_periods=1).sum())
    out["melt_7d"] = g["melt_index"].transform(lambda s: s.rolling(56, min_periods=1).sum())
    out["temp_7d_mean"] = g["temp_c"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["rain_7d"] = g["rain_mm_3h"].transform(lambda s: s.rolling(56, min_periods=1).sum())
    out["discharge_lag1"] = g["discharge_m3s"].transform(lambda s: s.shift(1))
    out["discharge_lag2"] = g["discharge_m3s"].transform(lambda s: s.shift(2))
    out["discharge_lag3"] = g["discharge_m3s"].transform(lambda s: s.shift(3))
    out["discharge_7d_mean"] = g["discharge_m3s"].transform(
        lambda s: s.rolling(56, min_periods=7).mean()
    )
    out["discharge_ratio"] = out["discharge_m3s"] / (out["discharge_7d_mean"] + 1e-6)

    # Forecast melt over the next 24h, from forecast temperature.
    temp_fcst = out["temp_fcst_c"] if "temp_fcst_c" in out else out["temp_c"]
    melt_fcst = temp_fcst.clip(lower=0.0) * 2.5 / 8.0 * out["snow_fraction"].astype(float)
    out["melt_next_24h"] = melt_fcst.groupby(out["basin_id"]).transform(
        lambda s: _forward_sum(s, 8)
    )
    out["temp_next_max"] = temp_fcst.groupby(out["basin_id"]).transform(
        lambda s: _forward_sum(s.clip(lower=0.0), 8)
    )

    cols = [
        "melt_index",
        "melt_3d",
        "melt_7d",
        "melt_next_24h",
        "temp_next_max",
        "temp_7d_mean",
        "rain_7d",
        "soil_moisture",
        "discharge_lag1",
        "discharge_lag2",
        "discharge_lag3",
        "discharge_7d_mean",
        "discharge_ratio",
        "doy_sin",
        "doy_cos",
    ]
    return out, cols


def design_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    basins: list[str] | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Numeric features plus one-hot basin identity. Returns (X, row_index)."""
    basins = basins or sorted(df["basin_id"].unique().tolist())
    X = df[feature_cols].astype(float).copy()
    dummies = pd.get_dummies(df["basin_id"].astype(str), prefix="basin", dtype=float)
    for b in basins:
        col = f"basin_{b}"
        if col not in dummies.columns:
            dummies[col] = 0.0
    dummies = dummies[[f"basin_{b}" for b in basins]]
    X = pd.concat([X, dummies], axis=1)

    if dropna:
        keep = X.notna().all(axis=1)
        X = X.loc[keep]
    return X, X.index
