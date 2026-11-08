"""The explainer is pure, exact at every boundary and never says 'higher' at severe."""

from __future__ import annotations

import pytest

from floodwatch import explain
from floodwatch.interop import RiskResult
from floodwatch.tiers import Level

THRESHOLDS = {"none": 0.0, "low": 0.1, "moderate": 0.3, "high": 0.6, "severe": 0.8}


def _result(tier: str, score: float) -> RiskResult:
    level = {
        "none": Level.GREEN,
        "low": Level.GREEN,
        "moderate": Level.YELLOW,
        "high": Level.ORANGE,
        "severe": Level.RED,
    }[tier]
    return RiskResult(
        area_id="swat_kalam",
        score=score,
        level=level,
        reasons=["rain_3h (8.2)"],
        computed_at="2022-08-27T00:00:00Z",
        tier=tier,
    )


@pytest.mark.parametrize(
    ("tier", "score", "next_tier", "gap"),
    [
        ("none", 0.05, "low", 0.05),
        ("low", 0.2, "moderate", 0.1),
        ("moderate", 0.45, "high", 0.15),
        ("high", 0.7, "severe", 0.1),
    ],
)
def test_gap_to_next_boundary(tier: str, score: float, next_tier: str, gap: float) -> None:
    out = explain.explain(_result(tier, score), THRESHOLDS)
    assert out.next_tier == next_tier
    assert out.gap_to_next == pytest.approx(gap)


def test_severe_has_no_higher_level() -> None:
    out = explain.explain(_result("severe", 0.95), THRESHOLDS)
    assert out.next_tier == "none"
    assert out.gap_to_next == 0.0
    assert "maximum tier" in out.binding_constraint


def test_binding_constraint_uses_live_conditions() -> None:
    conditions = {"rain_next_3h": 8.2, "rain_3h": 12.0, "soil_moisture": 0.4}
    text, feature = explain.binding_constraint(
        "moderate", conditions=conditions, thresholds=THRESHOLDS, prob=0.45
    )
    assert feature == "rain_next_3h"
    assert "8.2" in text
    assert "high needs" in text


def test_binding_constraint_falls_back_to_score_gap() -> None:
    text, feature = explain.binding_constraint(
        "moderate", conditions=None, reasons=None, thresholds=THRESHOLDS, prob=0.45
    )
    assert feature is None
    assert "0.45" in text and "0.60" in text


def test_explain_all_handles_conditions_by_area() -> None:
    results = [_result("moderate", 0.45)]
    out = explain.explain_all(
        results, THRESHOLDS, conditions_by_area={"swat_kalam": {"rain_3h": 8.0}}
    )
    assert out[0].driver_feature == "rain_3h"


def test_explainer_has_no_io_or_model_imports() -> None:
    source = explain.__file__ or ""
    text = open(source, encoding="utf-8").read()
    for banned in ("import torch", "import sklearn", "requests", "open("):
        assert banned not in text
