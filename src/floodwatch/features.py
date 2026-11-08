"""Feature engineering for the two mechanisms.

Scaffolding for the person-1 lane (M1): a single feature table, one
implementation, no forks. Columns are the contract both models read.

No feature uses data beyond the forecast horizon. Labels are computed from the
future and are the only column allowed to look forward.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASINS = ROOT / "configs" / "basins.yaml"

RAW_COLUMNS: tuple[str, ...] = (
    "basin_id",
    "timestamp_utc",
    "rain_mm",
    "rain_fcst_mm_3h",
    "temp_c",
    "temp_fcst_c",
    "freezing_level_m",
    "soil_moisture",
    "snow_depth_mm",
    "snowmelt_mm",
    "discharge_m3s",
    "snow_fraction",
    "elevation_mean_m",
    "terrain_susceptibility",
)

# The temporal sequence Model B reads (snowmelt emphasis).
FEATURE_COLUMNS_B: tuple[str, ...] = (
    "rain_1h",
    "rain_3h",
    "rain_24h",
    "rain_72h",
    "rain_fcst_mm_3h",
    "temp_c",
    "temp_fcst_c",
    "freezing_level_m",
    "soil_moisture",
    "snow_depth_mm",
    "snowmelt_mm",
    "degree_day_melt",
    "rain_on_snow",
    "terrain_susceptibility",
)

# Model A's temporal branch (cloudburst emphasis). The spatial branch consumes
# the 3x3 patch built by the ingest lane.
FEATURE_COLUMNS_A: tuple[str, ...] = (
    "rain_1h",
    "rain_3h",
    "rain_24h",
    "rain_72h",
    "rain_fcst_mm_3h",
    "temp_c",
    "freezing_level_m",
    "soil_moisture",
    "terrain_susceptibility",
)

DEFAULT_FEATURE_COLUMNS = FEATURE_COLUMNS_B


def load_basins(path: str | Path | None = None) -> list[dict]:
    data = yaml.safe_load(Path(path or DEFAULT_BASINS).read_text(encoding="utf-8"))
    return list(data["basins"])


def basin_ids(path: str | Path | None = None) -> list[str]:
    return [b["basin_id"] for b in load_basins(path)]


def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add rolling rainfall aggregates and interaction terms."""
    df = raw.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)

    grouped = df.groupby("basin_id", group_keys=False, sort=False)
    for hours, name in ((1, "rain_1h"), (3, "rain_3h"), (24, "rain_24h"), (72, "rain_72h")):
        df[name] = grouped["rain_mm"].transform(
            lambda series, window=hours: series.rolling(window, min_periods=1).sum()
        )

    # Rain-on-snow: liquid water on an existing snowpack drives the melt mechanism.
    snow_present = (df["snow_depth_mm"].astype(float) > 5.0).astype(float)
    df["rain_on_snow"] = df["rain_3h"].astype(float) * snow_present * df["snow_fraction"]

    # Degree-day melt, explicit so the baseline and the model share it.
    df["degree_day_melt"] = (
        np.clip(df["temp_c"].astype(float), 0.0, None) * 0.06 * df["snow_fraction"]
    )

    return df


def add_labels(
    features: pd.DataFrame,
    *,
    horizon_h: int = 6,
    train_mask: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Label = a flood proxy anywhere in the next ``horizon_h`` hours.

    Thresholds are estimated on the train split only so the label itself does
    not leak validation quantiles.
    """
    df = features.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)

    if train_mask is None:
        train_mask = pd.Series(True, index=df.index)
    train_mask = pd.Series(train_mask, index=df.index).fillna(False).astype(bool)

    thresholds: dict[str, dict[str, float]] = {}
    for basin_id, group in df.groupby("basin_id", sort=True):
        train_group = group[train_mask.loc[group.index]]
        if train_group.empty:
            train_group = group
        thresholds[basin_id] = {
            "rain": float(train_group["rain_3h"].quantile(0.975)),
            "melt": float(train_group["degree_day_melt"].quantile(0.90)),
        }

    labels = pd.Series(0, index=df.index, dtype="int8")
    for basin_id, group in df.groupby("basin_id", sort=True):
        idx = group.index
        rain = group["rain_3h"].astype(float)
        melt = group["degree_day_melt"].astype(float)
        # max over t+1 .. t+horizon_h
        future_rain = rain.rolling(horizon_h, min_periods=1).max().shift(-horizon_h)
        future_melt = melt.rolling(horizon_h, min_periods=1).max().shift(-horizon_h)
        hit = (future_rain >= thresholds[basin_id]["rain"]) | (
            future_melt >= thresholds[basin_id]["melt"]
        )
        labels.loc[idx] = hit.fillna(False).astype("int8")

    df["label"] = labels
    df.attrs["label_thresholds"] = thresholds
    return df


def feature_table(raw: pd.DataFrame, *, train_mask=None, horizon_h: int = 6) -> pd.DataFrame:
    return add_labels(add_features(raw), horizon_h=horizon_h, train_mask=train_mask)


def write_features(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def read_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df
