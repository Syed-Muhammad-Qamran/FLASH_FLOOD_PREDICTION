"""Temporal windowing and the canonical grid index.

Scaffolding for the person-1 lane (M1). Contract file: joint PR to change.

- 72 h lookback, 1 h stride.
- No window crosses a split boundary (a 30-day buffer is not scored).
- ``grid_index.parquet`` lists every ``(basin_id, timestamp_utc, split)`` both
  models must score. A missing row is a bug, not a zero.
- Scalers/imputers must be fit on train rows only; the helpers here expose the
  train mask so callers cannot forget.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "split.yaml"

LOOKBACK_H = 72
STRIDE_H = 1


def load_split_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def split_bounds(cfg: dict | None = None) -> dict[str, tuple[pd.Timestamp, pd.Timestamp | None]]:
    cfg = cfg or load_split_config()
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp | None]] = {}
    for name, block in cfg["splits"].items():
        start = pd.to_datetime(block["start"], utc=True)
        end = pd.to_datetime(block["end"], utc=True) if block.get("end") else None
        bounds[name] = (start, end)
    return bounds


def assign_split(timestamps: pd.Series, cfg: dict | None = None) -> pd.Series:
    """Return split label per timestamp; NaT gaps (buffers) become None."""
    ts = pd.to_datetime(timestamps, utc=True)
    out = pd.Series([None] * len(ts), index=ts.index, dtype="object")
    # Later splits overwrite earlier ones at boundaries; the ranges are disjoint.
    for name, (start, end) in split_bounds(cfg).items():
        mask = ts >= start
        if end is not None:
            mask &= ts < end
        out = out.mask(mask, name)
    return out


def train_mask(features: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    splits = assign_split(features["timestamp_utc"], cfg)
    return (splits == "train").fillna(False)


def build_grid_index(
    features: pd.DataFrame,
    *,
    cfg: dict | None = None,
    include_test: bool = True,
) -> pd.DataFrame:
    """Every scored ``(basin_id, timestamp_utc)`` with its split and label."""
    df = features.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["split"] = assign_split(df["timestamp_utc"], cfg)
    df = df[df["split"].notna()].copy()
    if not include_test:
        df = df[df["split"] != "test"]
    grid = (
        df[["basin_id", "timestamp_utc", "split"]]
        .drop_duplicates(["basin_id", "timestamp_utc"])
        .sort_values(["basin_id", "timestamp_utc"])
        .reset_index(drop=True)
    )
    grid["split"] = grid["split"].astype("string")
    grid["timestamp_utc"] = pd.to_datetime(grid["timestamp_utc"], utc=True)
    return grid


def write_grid_index(grid: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(p, index=False)
    return p


def read_grid_index(path: str | Path) -> pd.DataFrame:
    grid = pd.read_parquet(path)
    grid["timestamp_utc"] = pd.to_datetime(grid["timestamp_utc"], utc=True)
    return grid


def _basin_arrays(
    features: pd.DataFrame, feature_columns: tuple[str, ...] | list[str]
) -> dict[str, tuple[np.ndarray, pd.DatetimeIndex]]:
    arrays: dict[str, tuple[np.ndarray, pd.DatetimeIndex]] = {}
    for basin_id, group in features.groupby("basin_id", sort=False):
        group = group.sort_values("timestamp_utc")
        index = pd.DatetimeIndex(pd.to_datetime(group["timestamp_utc"], utc=True))
        full = pd.date_range(index.min(), index.max(), freq="h", tz="UTC")
        frame = group.set_index(index).reindex(full)
        values = frame[list(feature_columns)].astype("float32").fillna(0.0).to_numpy()
        arrays[str(basin_id)] = (values, full)
    return arrays


def iter_sequences(
    features: pd.DataFrame,
    keys: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    *,
    lookback_h: int = LOOKBACK_H,
    batch_size: int = 2048,
    pad_short: bool = False,
) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    """Yield ``(key_chunk, X)`` with ``X`` shaped ``(n, lookback_h, n_features)``.

    Training uses ``pad_short=False``: rows whose lookback is incomplete are
    dropped and never silently zero-filled into a valid sample.

    Scoring must cover the whole canonical grid, so ``pad_short=True`` is used
    there: the short history at the very start of a record is edge-padded
    (earliest observed value), which keeps every grid row scorable without
    inventing zeros.
    """
    arrays = _basin_arrays(features, feature_columns)
    keys = keys.reset_index(drop=True)
    n_features = len(feature_columns)

    for start in range(0, len(keys), batch_size):
        chunk = keys.iloc[start : start + batch_size].reset_index(drop=True)
        X = np.zeros((len(chunk), lookback_h, n_features), dtype=np.float32)
        keep = np.zeros(len(chunk), dtype=bool)
        for basin_id, positions in chunk.groupby("basin_id", sort=False).groups.items():
            arr, index = arrays[str(basin_id)]
            pos = index.get_indexer(pd.DatetimeIndex(chunk.loc[positions, "timestamp_utc"]))
            for local, p in zip(positions, pos, strict=False):
                if p < lookback_h - 1:
                    if not pad_short:
                        continue
                    # Edge-pad: repeat the earliest available hour.
                    start = max(0, p - lookback_h + 1)
                    block = arr[start : p + 1]
                    pad = lookback_h - len(block)
                    X[int(local)] = np.concatenate(
                        [np.repeat(block[:1], pad, axis=0), block], axis=0
                    )
                    keep[int(local)] = True
                    continue
                X[int(local)] = arr[p - lookback_h + 1 : p + 1]
                keep[int(local)] = True
        yield chunk.loc[keep].reset_index(drop=True), X[keep]


def latest_keys(grid_index: pd.DataFrame) -> pd.DataFrame:
    """Newest row per basin (used by the adapter and the dashboard)."""
    grid = grid_index.sort_values("timestamp_utc")
    return grid.groupby("basin_id", as_index=False).tail(1).reset_index(drop=True)
