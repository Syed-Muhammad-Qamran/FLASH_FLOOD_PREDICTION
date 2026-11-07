"""Proxy labels.

Neither module has observed flood labels. Both therefore use a *proxy* derived
from training years only:

  Model A (cloudburst): 3-hour rainfall above the train-period 95th percentile
  (with a physical floor) OR a weaker rainfall excess on already-wet soils.

  Model B (melt): GloFAS discharge above the train-period 90th percentile for
  that basin.

These are proxies, not truth. docs/label_rules.md states the consequences and
the review checklist. The one hard rule: thresholds are fitted on TRAIN rows
only, never on validation or test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import thresholds_config
from .splits import mask as split_mask

RAIN_COL = "rain_mm_3h"
SOIL_COL = "soil_moisture"
DISCHARGE_COL = "discharge_m3s"

MIN_TRAIN_ROWS = 500


def _train(table: pd.DataFrame) -> pd.DataFrame:
    return table.loc[split_mask(table["timestamp_utc"], "train")]


def fit_thresholds_a(
    table: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or thresholds_config()
    block = cfg["model_a"]
    q = float(block["rain_3h_percentile"])
    floor_mm = float(block["rain_3h_floor_mm"])
    sq = float(block["soil_moisture_percentile"])

    train = _train(table)
    if train.empty:
        raise ValueError("no training rows; check configs/split.yaml against the data")

    global_rain = float(max(train[RAIN_COL].quantile(q), floor_mm))
    global_soil = float(train[SOIL_COL].quantile(sq))

    rain: dict[str, float] = {}
    soil: dict[str, float] = {}
    for basin_id, grp in train.groupby("basin_id"):
        if len(grp) < MIN_TRAIN_ROWS:
            rain[str(basin_id)] = global_rain
            soil[str(basin_id)] = global_soil
            continue
        rain[str(basin_id)] = float(max(grp[RAIN_COL].quantile(q), floor_mm))
        soil[str(basin_id)] = float(grp[SOIL_COL].quantile(sq))

    return {
        "rain_mm": rain,
        "soil_moisture": soil,
        "global_rain_mm": global_rain,
        "global_soil_moisture": global_soil,
        "rain_percentile": q,
        "soil_percentile": sq,
        "floor_mm": floor_mm,
        "provenance": "train-only",
    }


def label_a(table: pd.DataFrame, thresholds: dict[str, Any]) -> pd.Series:
    """Cloudburst proxy: intense 3h rain, or moderate rain on saturated soil."""
    rain_thr = table["basin_id"].map(thresholds["rain_mm"]).astype(float)
    rain_thr = rain_thr.fillna(thresholds["global_rain_mm"])
    soil_thr = table["basin_id"].map(thresholds["soil_moisture"]).astype(float)
    soil_thr = soil_thr.fillna(thresholds["global_soil_moisture"])

    intense = table[RAIN_COL] >= rain_thr
    wet_soil_excess = (table[RAIN_COL] >= 0.7 * rain_thr) & (table[SOIL_COL] >= soil_thr)
    return (intense | wet_soil_excess).fillna(False).astype(bool)


def fit_thresholds_b(
    table: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or thresholds_config()
    block = cfg["model_b"]
    q = float(block["discharge_percentile"])

    train = _train(table)
    if train.empty:
        raise ValueError("no training rows; check configs/split.yaml against the data")

    global_q = float(train[DISCHARGE_COL].quantile(q))
    per_basin: dict[str, float] = {}
    for basin_id, grp in train.groupby("basin_id"):
        if len(grp) < MIN_TRAIN_ROWS:
            per_basin[str(basin_id)] = global_q
            continue
        per_basin[str(basin_id)] = float(grp[DISCHARGE_COL].quantile(q))

    return {
        "discharge_m3s": per_basin,
        "global_discharge_m3s": global_q,
        "percentile": q,
        "provenance": "train-only",
    }


def label_b(table: pd.DataFrame, thresholds: dict[str, Any]) -> pd.Series:
    thr = table["basin_id"].map(thresholds["discharge_m3s"]).astype(float)
    thr = thr.fillna(thresholds["global_discharge_m3s"])
    return (table[DISCHARGE_COL] >= thr).fillna(False).astype(bool)


def event_ids(labels: pd.Series, merge_gap: int = 1) -> pd.Series:
    """Integer event id per row, -1 outside any event.

    Consecutive positives separated by <= merge_gap negatives share an id.
    """
    from .metrics import find_events

    values = np.full(len(labels), -1, dtype=int)
    for event_no, (start, end) in enumerate(find_events(labels, merge_gap=merge_gap)):
        values[start : end + 1] = event_no
    return pd.Series(values, index=labels.index, name="event_id")


def build_labels(
    table: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach label_a, label_b and event ids to a feature table."""
    cfg = cfg or thresholds_config()
    thr_a = fit_thresholds_a(table, cfg)
    thr_b = fit_thresholds_b(table, cfg)

    out = table.copy()
    out["label_a"] = label_a(out, thr_a)
    out["label_b"] = label_b(out, thr_b)
    out["event_a"] = event_ids(out["label_a"])
    out["event_b"] = event_ids(out["label_b"])

    meta = {
        "model_a": thr_a,
        "model_b": thr_b,
        "base_rate_a_train": float(out.loc[split_mask(out["timestamp_utc"], "train"), "label_a"].mean()),
        "base_rate_b_train": float(out.loc[split_mask(out["timestamp_utc"], "train"), "label_b"].mean()),
        "base_rate_a_test": float(out.loc[split_mask(out["timestamp_utc"], "test"), "label_a"].mean()),
        "base_rate_b_test": float(out.loc[split_mask(out["timestamp_utc"], "test"), "label_b"].mean()),
    }
    return out, meta


def forward_label(label: pd.Series, horizon_steps: int) -> pd.Series:
    """Shift a label so that row t answers 'does an event occur in t+1..t+h?'."""
    if horizon_steps <= 0:
        return label.astype(bool)

    def _fwd(s: pd.Series) -> pd.Series:
        rev = s.astype(float).iloc[::-1]
        rolled = rev.rolling(horizon_steps, min_periods=1).max().iloc[::-1]
        return rolled.shift(-1)

    return _fwd(label).fillna(0).astype(bool)
