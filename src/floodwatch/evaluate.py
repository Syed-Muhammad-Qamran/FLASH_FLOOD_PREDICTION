"""Evaluation entry point used by scripts and by the app's report tab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics
from .labels import forward_label
from .splits import mask as split_mask


def align_to_table(table: pd.DataFrame, pred: pd.DataFrame) -> pd.Series:
    """Reindex a (basin_id, timestamp_utc, prob) frame onto `table`'s rows."""
    idx = pd.MultiIndex.from_frame(
        table[["basin_id", "timestamp_utc"]].assign(
            timestamp_utc=pd.to_datetime(table["timestamp_utc"], utc=True)
        )
    )
    series = pred.assign(
        timestamp_utc=pd.to_datetime(pred["timestamp_utc"], utc=True)
    ).set_index(["basin_id", "timestamp_utc"])["prob"]
    return pd.Series(series.reindex(idx).to_numpy(), index=table.index, name="prob")


def evaluate_series(
    table: pd.DataFrame,
    prob: pd.Series,
    label_col: str,
    split: str,
    threshold: float,
    *,
    horizon_steps: int = 0,
    n_boot: int = 500,
) -> dict[str, Any]:
    """Score one module on one split.

    `prob` must be aligned to `table.index`. Rows with no prediction are dropped
    (that is honest: the model did not issue a forecast there).
    """
    if horizon_steps > 0:
        y = forward_label(table[label_col], horizon_steps)
    else:
        y = table[label_col]

    in_split = split_mask(table["timestamp_utc"], split)
    valid = in_split & prob.notna()

    sub = table.loc[valid]
    y_true = y.loc[valid].to_numpy(dtype=bool)
    y_score = prob.loc[valid].to_numpy(dtype=float)

    if len(sub) == 0:
        raise ValueError(f"no scored rows for split {split!r}")

    # Block bootstrap by event when events exist, else by basin.
    if "event_a" in sub.columns and label_col == "label_a":
        groups = sub["event_a"].to_numpy()
    elif "event_b" in sub.columns and label_col == "label_b":
        groups = sub["event_b"].to_numpy()
    else:
        groups = sub["basin_id"].to_numpy()
    if np.all(groups < 0):
        groups = sub["basin_id"].to_numpy()

    result = metrics.summary(
        sub["timestamp_utc"],
        y_true,
        y_score,
        threshold,
        groups=groups,
        event_labels=table.loc[valid, label_col].to_numpy(dtype=bool),
        n_boot=n_boot,
    )
    result["split"] = split
    result["label"] = label_col
    result["rows_scored"] = int(len(sub))
    result["basins"] = int(sub["basin_id"].nunique())
    return result


def compare(
    table: pd.DataFrame,
    candidates: dict[str, pd.Series],
    label_col: str,
    split: str,
    thresholds: dict[str, float],
    *,
    horizon_steps: int = 0,
    n_boot: int = 500,
) -> pd.DataFrame:
    rows = []
    for name, prob in candidates.items():
        res = evaluate_series(
            table,
            prob,
            label_col,
            split,
            thresholds.get(name, 0.5),
            horizon_steps=horizon_steps,
            n_boot=n_boot,
        )
        res["candidate"] = name
        rows.append(res)
    cols = ["candidate", "split", "threshold", "pod", "far", "csi", "pr_auc", "event_pod", "event_far", "event_csi", "median_lead_hours", "brier", "rows_scored"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def write_report(results: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return p
