from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodwatch import metrics


def test_contingency_and_scores_known_values():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0]
    c = metrics.contingency(y_true, y_pred)
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    s = metrics.point_scores(y_true, y_pred)
    assert s["pod"] == pytest.approx(2 / 3)
    assert s["far"] == pytest.approx(1 / 3)
    assert s["csi"] == pytest.approx(2 / 4)
    assert s["alert_rate"] == pytest.approx(3 / 6)
    assert s["base_rate"] == pytest.approx(3 / 6)


def test_find_events_merges_short_gaps():
    flags = [0, 1, 1, 0, 1, 0, 0, 1]
    assert metrics.find_events(flags, merge_gap=1) == [(1, 4), (7, 7)]
    assert metrics.find_events(flags, merge_gap=0) == [(1, 2), (4, 4), (7, 7)]


def test_event_scores_lead_time():
    times = pd.date_range("2024-01-01", periods=12, freq="3h", tz="UTC")
    y_true = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    r = metrics.event_scores(times, y_true, y_pred)
    assert r["n_events"] == 1
    assert r["event_pod"] == 1.0
    assert r["n_false_alarm_episodes"] == 0
    # alert starts 2 steps (6h) before the event
    assert r["median_lead_hours"] == pytest.approx(6.0)


def test_false_alarm_episode_counted():
    times = pd.date_range("2024-01-01", periods=10, freq="3h", tz="UTC")
    y_true = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    r = metrics.event_scores(times, y_true, y_pred, max_lead_steps=1)
    assert r["event_pod"] == 1.0
    assert r["n_false_alarm_episodes"] == 2


def test_bootstrap_ci_is_ordered():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = np.clip(y * 0.5 + rng.normal(0.25, 0.15, 400), 0, 1)
    ci = metrics.bootstrap_ci(
        y, s, lambda t, p: metrics.point_scores(t, p >= 0.5)["csi"], n_boot=200, seed=1
    )
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["n"] > 0


def test_empty_events_returns_nan_not_crash():
    times = pd.date_range("2024-01-01", periods=5, freq="3h", tz="UTC")
    r = metrics.event_scores(times, [0] * 5, [0] * 5)
    assert r["n_events"] == 0
    assert np.isnan(r["event_pod"])
