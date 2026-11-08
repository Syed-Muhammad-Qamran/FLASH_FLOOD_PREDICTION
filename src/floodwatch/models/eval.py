"""Standard evaluation harness shared by both mechanisms.

Contract file: joint `contract` PR to change.

All functions are pure and numpy/pandas only. They do not import torch or
sklearn so the harness can run in the lightweight app image.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def confusion_counts(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) at a probability threshold."""
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_score) >= float(threshold)
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))
    return tp, fp, fn, tn


def pod(y_true, y_score, threshold: float) -> float:
    tp, _, fn, _ = confusion_counts(y_true, y_score, threshold)
    return _safe_div(tp, tp + fn)


def far(y_true, y_score, threshold: float) -> float:
    tp, fp, _, _ = confusion_counts(y_true, y_score, threshold)
    return _safe_div(fp, tp + fp)


def csi(y_true, y_score, threshold: float) -> float:
    tp, fp, fn, _ = confusion_counts(y_true, y_score, threshold)
    return _safe_div(tp, tp + fp + fn)


def precision(y_true, y_score, threshold: float) -> float:
    tp, fp, _, _ = confusion_counts(y_true, y_score, threshold)
    return _safe_div(tp, tp + fp)


def recall(y_true, y_score, threshold: float) -> float:
    return pod(y_true, y_score, threshold)


def average_precision(y_true, y_score) -> float:
    """Rank-based PR-AUC (average precision). Zero when there are no positives."""
    truth = np.asarray(y_true).astype(bool)
    score = np.asarray(y_score, dtype=float)
    if truth.size == 0 or not truth.any():
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    truth = truth[order]
    score = score[order]
    # Collapse ties so equal scores are treated as one operating point.
    tp = np.cumsum(truth)
    fp = np.cumsum(~truth)
    keep = np.r_[score[1:] != score[:-1], True]
    tp, fp = tp[keep], fp[keep]
    precision_curve = tp / np.maximum(tp + fp, 1)
    recall_curve = tp / np.maximum(tp[-1], 1)
    ap = 0.0
    prev_recall = 0.0
    for prec, rec in zip(precision_curve, recall_curve, strict=False):
        ap += prec * (rec - prev_recall)
        prev_recall = rec
    return float(ap)


def _event_runs(truth: pd.DataFrame, *, min_gap_h: float = 6.0) -> list[dict[str, Any]]:
    """Group contiguous positive labels into events per basin."""
    events: list[dict[str, Any]] = []
    frame = truth.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    for basin_id, group in frame.sort_values("timestamp_utc").groupby("basin_id", sort=True):
        positives = group[group["label"].astype(int) == 1]
        if positives.empty:
            continue
        times = positives["timestamp_utc"].tolist()
        start = prev = times[0]
        for ts in times[1:]:
            if (ts - prev) > pd.Timedelta(hours=min_gap_h):
                events.append({"basin_id": basin_id, "start": start, "end": prev})
                start = ts
            prev = ts
        events.append({"basin_id": basin_id, "start": start, "end": prev})
    return events


def event_metrics(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
    *,
    lead_window_h: float = 24.0,
    min_gap_h: float = 6.0,
) -> dict[str, float]:
    """Event POD and median lead time.

    ``truth`` needs ``basin_id, timestamp_utc, label``; ``predictions`` needs
    ``basin_id, timestamp_utc, prob``. An event counts as detected if any alert
    is raised in ``[start - lead_window_h, end]``.
    """
    events = _event_runs(truth, min_gap_h=min_gap_h)
    if not events:
        return {"event_pod": 0.0, "median_lead_time_h": 0.0, "events": 0.0}

    preds = predictions.copy()
    preds["timestamp_utc"] = pd.to_datetime(preds["timestamp_utc"], utc=True)
    alerts = preds[preds["prob"] >= float(threshold)]

    detected = 0
    leads: list[float] = []
    for event in events:
        basin_alerts = alerts[alerts["basin_id"] == event["basin_id"]]
        window_start = event["start"] - pd.Timedelta(hours=lead_window_h)
        # The event end plus the label horizon is the useful detection window.
        window_end = event["end"] + pd.Timedelta(hours=6)
        hits = basin_alerts[
            (basin_alerts["timestamp_utc"] >= window_start)
            & (basin_alerts["timestamp_utc"] <= window_end)
        ]
        if not hits.empty:
            detected += 1
            first = hits["timestamp_utc"].min()
            leads.append(max(0.0, (event["start"] - first).total_seconds() / 3600.0))

    return {
        "event_pod": _safe_div(detected, len(events)),
        "median_lead_time_h": float(np.median(leads)) if leads else 0.0,
        "events": float(len(events)),
    }


def evaluate(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
    *,
    include_events: bool = True,
) -> dict[str, float]:
    """Full metric block for one mechanism or fusion variant."""
    merged = truth[["basin_id", "timestamp_utc", "label"]].merge(
        predictions[["basin_id", "timestamp_utc", "prob"]],
        on=["basin_id", "timestamp_utc"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return {
            "rows": 0.0,
            "threshold": float(threshold),
            "pod": 0.0,
            "far": 0.0,
            "csi": 0.0,
            "precision": 0.0,
            "pr_auc": 0.0,
            "event_pod": 0.0,
            "median_lead_time_h": 0.0,
        }
    y_true = merged["label"].astype(int)
    y_score = merged["prob"].astype(float)
    metrics: dict[str, float] = {
        "rows": float(len(merged)),
        "threshold": float(threshold),
        "pod": pod(y_true, y_score, threshold),
        "far": far(y_true, y_score, threshold),
        "csi": csi(y_true, y_score, threshold),
        "precision": precision(y_true, y_score, threshold),
        "pr_auc": average_precision(y_true, y_score),
        "positive_rate": float(y_true.mean()),
    }
    if include_events:
        metrics.update(event_metrics(truth, predictions, threshold))
    return metrics


def sweep_thresholds(
    y_true,
    y_score,
    *,
    steps: int = 99,
) -> list[dict[str, float]]:
    """Operating points from 0.01 to 0.99 for threshold selection."""
    out = []
    for i in range(1, steps + 1):
        threshold = i / (steps + 1)
        out.append(
            {
                "threshold": threshold,
                "pod": pod(y_true, y_score, threshold),
                "far": far(y_true, y_score, threshold),
                "csi": csi(y_true, y_score, threshold),
            }
        )
    return out
