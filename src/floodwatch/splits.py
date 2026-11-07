"""Chronological splits with a guard band.

The single most common way this kind of project produces impressive-looking
but meaningless numbers is leakage across the split. Everything here exists to
make leakage a hard error instead of a code review comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import split_config as _load_split_config

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class Window:
    name: str
    start: pd.Timestamp  # inclusive, UTC
    end: pd.Timestamp  # exclusive, UTC

    def contains(self, ts: pd.Series) -> pd.Series:
        return (ts >= self.start) & (ts < self.end)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_windows(cfg: dict[str, Any] | None = None) -> dict[str, Window]:
    """Build the split windows, applying the gap to the start of each later split."""
    cfg = cfg or _load_split_config()
    gap = pd.Timedelta(days=int(cfg.get("gap_days", 0)))

    windows: dict[str, Window] = {}
    for name in SPLIT_NAMES:
        block = cfg[name]
        start = _utc(block["start"])
        end = _utc(block["end"]) + pd.Timedelta(days=1)  # make end inclusive
        if name != "train":
            start = start + gap
        windows[name] = Window(name=name, start=start, end=end)

    assert_no_overlap(windows)
    return windows


def assert_no_overlap(windows: dict[str, Window]) -> None:
    ordered = [windows[n] for n in SPLIT_NAMES]
    for a, b in zip(ordered, ordered[1:]):
        if a.end > b.start:
            raise ValueError(
                f"split windows overlap or the gap is too small: "
                f"{a.name} ends {a.end} but {b.name} starts {b.start}"
            )


def assign_split(timestamps: pd.Series, cfg: dict[str, Any] | None = None) -> pd.Series:
    """Label each timestamp as train / validation / test / gap / outside."""
    ts = pd.to_datetime(timestamps, utc=True)
    windows = load_windows(cfg)

    conditions = [windows[n].contains(ts) for n in SPLIT_NAMES]
    # Anything between windows is "gap"; anything beyond is "outside".
    choices = list(SPLIT_NAMES)
    out = pd.Series(
        np.select(conditions, choices, default="outside"),
        index=ts.index,
        dtype="object",
    )

    between_gap = pd.Series(False, index=ts.index)
    for a, b in zip(SPLIT_NAMES, SPLIT_NAMES[1:]):
        between_gap |= (ts >= windows[a].end) & (ts < windows[b].start)
    out[between_gap] = "gap"
    return out.astype("string")


def mask(timestamps: pd.Series, name: str, cfg: dict[str, Any] | None = None) -> pd.Series:
    windows = load_windows(cfg)
    if name not in windows:
        raise KeyError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    return windows[name].contains(pd.to_datetime(timestamps, utc=True))


def describe(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    for name, w in load_windows(cfg).items():
        rows.append(
            {
                "split": name,
                "start_utc": w.start,
                "end_utc": w.end - pd.Timedelta(days=1),
                "days": (w.end - w.start).days,
            }
        )
    return pd.DataFrame(rows)
