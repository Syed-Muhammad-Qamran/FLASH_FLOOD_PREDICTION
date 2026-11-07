"""Scoring for a rare-event warning system.

Pointwise scores alone are misleading for floods: a model can have POD 0.9 and
still be useless if it alerts every day. So every report carries:

  * pointwise contingency (POD, FAR, CSI, precision, F1)
  * base rate and alert rate, so FAR is interpretable
  * PR-AUC (imbalanced-data honest) and Brier
  * event-level POD/FAR/CSI, because "did we warn on the flood" is the question
  * lead time, because a warning at the flood peak is not a warning
  * bootstrap confidence intervals, grouped by event when available
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # scikit-learn is a core dependency, but keep the module importable without it
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


def _as_bool(a: Sequence | np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(a).astype(bool)


def contingency(y_true, y_pred) -> dict[str, int]:
    yt = _as_bool(y_true)
    yp = _as_bool(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same length")
    return {
        "tp": int(np.sum(yt & yp)),
        "fp": int(np.sum(~yt & yp)),
        "fn": int(np.sum(yt & ~yp)),
        "tn": int(np.sum(~yt & ~yp)),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def point_scores(y_true, y_pred) -> dict[str, float]:
    c = contingency(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    pod = _safe_div(tp, tp + fn)
    far = _safe_div(fp, tp + fp)
    csi = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    return {
        **c,
        "n": n,
        "pod": pod,
        "far": far,
        "csi": csi,
        "precision": precision,
        "recall": pod,
        "f1": _safe_div(2 * precision * pod, precision + pod),
        "base_rate": _safe_div(tp + fn, n),
        "alert_rate": _safe_div(tp + fp, n),
        "specificity": _safe_div(tn, tn + fp),
    }


def pr_auc(y_true, y_score) -> float:
    yt = _as_bool(y_true)
    if average_precision_score is None or yt.sum() == 0 or yt.sum() == len(yt):
        return float("nan")
    return float(average_precision_score(yt, np.asarray(y_score, dtype=float)))


def roc_auc(y_true, y_score) -> float:
    yt = _as_bool(y_true)
    if roc_auc_score is None or yt.sum() == 0 or yt.sum() == len(yt):
        return float("nan")
    return float(roc_auc_score(yt, np.asarray(y_score, dtype=float)))


def brier(y_true, y_score) -> float:
    yt = _as_bool(y_true).astype(float)
    return float(np.mean((np.asarray(y_score, dtype=float) - yt) ** 2))


def find_events(flags, merge_gap: int = 1) -> list[tuple[int, int]]:
    """Return inclusive (start, end) index pairs for runs of True.

    Runs separated by <= merge_gap False steps are merged into one event.
    """
    f = _as_bool(flags)
    events: list[tuple[int, int]] = []
    i, n = 0, len(f)
    while i < n:
        if not f[i]:
            i += 1
            continue
        start = i
        while i < n and f[i]:
            i += 1
        end = i - 1
        if events and start - events[-1][1] - 1 <= merge_gap:
            events[-1] = (events[-1][0], end)
        else:
            events.append((start, end))
    return events


def _step_hours(times: pd.DatetimeIndex) -> float:
    if len(times) < 2:
        return 3.0
    # Do not use .view("int64"): the integer unit depends on the pandas version.
    diffs = pd.Series(times).diff().dt.total_seconds().to_numpy()
    positive = diffs[diffs > 0]
    return float(np.median(positive)) / 3600.0 if len(positive) else 3.0


def event_scores(
    times: Iterable,
    y_true,
    y_pred,
    *,
    max_lead_steps: int = 8,
    merge_gap: int = 1,
) -> dict[str, float]:
    """Event-based POD/FAR/CSI plus lead time.

    An event counts as detected if an alert occurs inside it *or* up to
    `max_lead_steps` before it: an early warning is the point, so it must not be
    scored as a false alarm. An alert episode outside every event's warning
    window counts as one false alarm.
    """
    times = pd.DatetimeIndex(pd.to_datetime(list(times), utc=True))
    yt = _as_bool(y_true)
    yp = _as_bool(y_pred)
    if len(times) != len(yt):
        raise ValueError("times and labels must have the same length")

    events = find_events(yt, merge_gap=merge_gap)
    if not events:
        return {
            "n_events": 0,
            "event_pod": float("nan"),
            "event_far": float("nan"),
            "event_csi": float("nan"),
            "median_lead_hours": float("nan"),
            "share_detected_with_lead": float("nan"),
            "n_false_alarm_episodes": 0,
        }

    step_h = _step_hours(times)
    covered = np.zeros(len(yt), dtype=bool)
    hits = 0
    leads: list[float] = []

    for start, end in events:
        window_start = max(0, start - max_lead_steps)
        # Alerts inside the event or in its warning window are not false alarms.
        covered[window_start : end + 1] = True
        if yp[window_start : end + 1].any():
            hits += 1
            j = start
            while j - 1 >= 0 and yp[j - 1] and (start - (j - 1)) <= max_lead_steps:
                j -= 1
            leads.append((start - j) * step_h)

    fa_mask = yp & ~covered
    fa_episodes = find_events(fa_mask, merge_gap=merge_gap)

    n_events = len(events)
    n_fa = len(fa_episodes)
    return {
        "n_events": n_events,
        "event_pod": _safe_div(hits, n_events),
        "event_far": _safe_div(n_fa, hits + n_fa),
        "event_csi": _safe_div(hits, n_events + n_fa),
        "median_lead_hours": float(np.median(leads)) if leads else float("nan"),
        "share_detected_with_lead": (
            float(np.mean([x >= step_h for x in leads])) if leads else float("nan")
        ),
        "n_false_alarm_episodes": n_fa,
    }


def bootstrap_ci(
    y_true,
    y_score,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    groups: Sequence | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap. Pass `groups` (e.g. event ids) for a block bootstrap.

    Resampling whole events is the right default here: timesteps inside a storm
    are strongly dependent, so a timestep bootstrap understates uncertainty.
    """
    yt = np.asarray(y_true)
    ys = np.asarray(y_score)
    rng = np.random.default_rng(seed)

    if groups is None:
        n = len(yt)
        draws = (rng.integers(0, n, n) for _ in range(n_boot))
        draw_list = list(draws)
    else:
        g = np.asarray(groups)
        uniq, inv = np.unique(g, return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        draw_list = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            draw_list.append(np.concatenate([members[p] for p in pick]))

    values: list[float] = []
    for idx in draw_list:
        try:
            v = float(metric_fn(yt[idx], ys[idx]))
        except Exception:
            continue
        if np.isfinite(v):
            values.append(v)

    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(arr, alpha / 2)),
        "hi": float(np.quantile(arr, 1 - alpha / 2)),
        "n": int(len(arr)),
    }


def calibration_table(y_true, y_score, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": _as_bool(y_true).astype(int), "p": np.asarray(y_score, float)})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    out = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    out["bin"] = out["bin"].astype(str)
    return out


def summary(
    times: Iterable,
    y_true,
    y_score,
    threshold: float,
    *,
    groups: Sequence | None = None,
    event_labels: Sequence | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """One call that produces the numbers the report and the app need.

    `y_true` should be the forward-shifted label used for training (classification
    quality). `event_labels` should be the *raw* label, so event boundaries and
    lead time are measured against when the hazard actually occurs.
    """
    yt = _as_bool(y_true)
    ys = np.asarray(y_score, dtype=float)
    yp = ys >= threshold
    ev = yt if event_labels is None else _as_bool(event_labels)

    out: dict[str, float] = {}
    out.update(point_scores(yt, yp))
    out["pr_auc"] = pr_auc(yt, ys)
    out["roc_auc"] = roc_auc(yt, ys)
    out["brier"] = brier(yt, ys)
    out["threshold"] = float(threshold)
    out.update(event_scores(times, ev, yp))

    for name, fn in (
        ("pod", lambda t, s: point_scores(t, s >= threshold)["pod"]),
        ("far", lambda t, s: point_scores(t, s >= threshold)["far"]),
        ("csi", lambda t, s: point_scores(t, s >= threshold)["csi"]),
    ):
        ci = bootstrap_ci(yt, ys, fn, groups=groups, n_boot=n_boot, seed=seed)
        out[f"{name}_ci_lo"] = ci["lo"]
        out[f"{name}_ci_hi"] = ci["hi"]
    return out
