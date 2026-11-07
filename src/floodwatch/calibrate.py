"""Threshold and tier calibration on VALIDATION years only.

This script proposes numbers. A human reviews the proposal, commits it into
configs/thresholds.yaml under `calibrated:`, and records the commit in
docs/log.md. That review step is the point: it makes threshold changes visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import metrics
from .labels import forward_label
from .splits import mask as split_mask


def pick_operating_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    min_alert_rate: float = 0.002,
    grid_size: int = 300,
) -> dict[str, float]:
    """Threshold that maximises CSI, subject to a minimum alert rate."""
    yt = np.asarray(y_true).astype(bool)
    ys = np.asarray(y_score, dtype=float)
    if len(np.unique(ys)) < 5:
        return {"threshold": 0.5, "csi": float("nan"), "pod": float("nan"), "far": float("nan"), "alert_rate": float("nan")}

    candidates = np.unique(np.quantile(ys, np.linspace(0.5, 0.9995, grid_size)))
    best: dict[str, float] | None = None
    for thr in candidates:
        pred = ys >= thr
        alert_rate = float(pred.mean())
        if alert_rate < min_alert_rate:
            continue
        s = metrics.point_scores(yt, pred)
        if not np.isfinite(s["csi"]):
            continue
        if best is None or (s["csi"], s["pod"]) > (best["csi"], best["pod"]):
            best = {
                "threshold": float(thr),
                "csi": float(s["csi"]),
                "pod": float(s["pod"]),
                "far": float(s["far"]),
                "alert_rate": alert_rate,
            }
    return best or {"threshold": 0.5, "csi": float("nan"), "pod": float("nan"), "far": float("nan"), "alert_rate": float("nan")}


def tier_cutoffs(y_score: np.ndarray, high_threshold: float) -> dict[str, float]:
    """Bracket the operating point so tiers stay interpretable.

    The high tier is the CSI-optimal alert threshold. Lower tiers are fractions
    of it, and severe is the upper half of the remaining probability range. This
    is stable even when the score distribution is bimodal.
    """
    ys = np.asarray(y_score, dtype=float)
    q = lambda p: float(np.quantile(ys, p))  # noqa: E731
    high = float(np.clip(high_threshold, 0.02, 0.95))
    low = float(np.clip(0.35 * high, 0.005, high - 2e-3))
    moderate = float(np.clip(0.65 * high, low + 1e-3, high - 1e-3))
    severe = float(np.clip(max(high + 0.5 * (1.0 - high), q(0.995)), high + 1e-3, 0.9999))
    return {"low": low, "moderate": moderate, "high": high, "severe": severe}


def calibrate_module(
    table: pd.DataFrame,
    prob: pd.Series,
    label_col: str,
    *,
    horizon_steps: int,
    split: str = "validation",
    min_alert_rate: float = 0.002,
) -> dict[str, Any]:
    y = forward_label(table[label_col], horizon_steps)
    in_split = split_mask(table["timestamp_utc"], split) & prob.notna()
    if in_split.sum() == 0:
        raise ValueError(f"no rows to calibrate on split {split!r}")

    y_true = y.loc[in_split].to_numpy(dtype=bool)
    y_score = prob.loc[in_split].to_numpy(dtype=float)

    operating = pick_operating_threshold(y_true, y_score, min_alert_rate=min_alert_rate)
    cutoffs = tier_cutoffs(y_score, operating["threshold"])
    point = metrics.point_scores(y_true, y_score >= operating["threshold"])

    return {
        "split": split,
        "rows": int(in_split.sum()),
        "positives": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "operating": operating,
        "tiers": cutoffs,
        "validation_point_scores": {k: float(v) for k, v in point.items()},
    }


def to_yaml_block(model_key: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        model_key: {
            "calibrated": {
                "tiers": {k: round(float(v), 6) for k, v in result["tiers"].items()},
                "operating_threshold": round(float(result["operating"]["threshold"]), 6),
                "calibrated_on": result["split"],
                "rows": result["rows"],
                "positives": result["positives"],
            }
        }
    }


def write_proposal(results: dict[str, dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    payload: dict[str, Any] = {
        "# PROPOSAL ONLY - review, then copy values into configs/thresholds.yaml": None
    }
    for module, res in results.items():
        payload.update(to_yaml_block(f"model_{module.lower()}", res))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return p
