"""'Why not higher?' explainability.

Pure functions only: no I/O, no network, no pandas. Given a ``RiskResult`` and
the fitted tier cutoffs, report the next tier, the probability gap and the
meteorological/hydrological constraint that binds first.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .tiers import TIER_RANK, TIERS, next_tier, tier_rank

# Illustrative requirements for the *next* tier. They are display guidance for
# the demo, not calibrated physical thresholds; the probability cutoffs remain
# the authority for the tier itself.
FEATURE_REQUIREMENTS: dict[str, dict[str, float]] = {
    "rain_3h": {"moderate": 10.0, "high": 25.0, "severe": 45.0},
    "rain_24h": {"moderate": 30.0, "high": 70.0, "severe": 120.0},
    "rain_72h": {"moderate": 45.0, "high": 110.0, "severe": 170.0},
    "rain_next_3h": {"moderate": 10.0, "high": 25.0, "severe": 45.0},
    "soil_moisture": {"moderate": 0.50, "high": 0.70, "severe": 0.85},
    "degree_day_melt": {"moderate": 0.30, "high": 0.70, "severe": 1.20},
    "rain_on_snow": {"moderate": 5.0, "high": 15.0, "severe": 30.0},
    "temp_c": {"moderate": 10.0, "high": 18.0, "severe": 25.0},
    "snow_depth_mm": {"moderate": 50.0, "high": 100.0, "severe": 150.0},
    "freezing_level_m": {"moderate": 3000.0, "high": 3800.0, "severe": 4500.0},
}

UNITS = {
    "rain_3h": "mm",
    "rain_24h": "mm",
    "rain_72h": "mm",
    "rain_next_3h": "mm",
    "rain_on_snow": "mm",
    "snow_depth_mm": "mm",
    "degree_day_melt": "mm/h",
    "soil_moisture": "",
    "temp_c": "C",
    "freezing_level_m": "m",
}

PRIORITY = (
    "rain_next_3h",
    "rain_3h",
    "rain_24h",
    "rain_72h",
    "rain_on_snow",
    "degree_day_melt",
    "soil_moisture",
    "snow_depth_mm",
    "temp_c",
    "freezing_level_m",
)


@dataclass(frozen=True)
class Explanation:
    area_id: str
    tier: str
    next_tier: str
    gap_to_next: float
    binding_constraint: str
    driver_feature: str | None = None


def next_cutoff(tier: str, thresholds: Mapping[str, float]) -> float | None:
    following = next_tier(tier)
    if tier_rank(tier) >= TIER_RANK["severe"] or following == "none":
        return None
    value = thresholds.get(following)
    return float(value) if value is not None else None


def gap_to_next(prob: float, tier: str, thresholds: Mapping[str, float]) -> float:
    cutoff = next_cutoff(tier, thresholds)
    if cutoff is None:
        return 0.0
    return round(max(0.0, cutoff - float(prob)), 6)


def _constraint_for(
    feature: str,
    value: float,
    target_tier: str,
) -> str:
    unit = UNITS.get(feature, "")
    target = FEATURE_REQUIREMENTS.get(feature, {}).get(target_tier)
    if target is None:
        suffix = f" ~{target:g} {unit}".rstrip() if target is not None else ""
        return f"{feature} ({value:g} {unit}){suffix}"
    suffix = f"{target_tier} needs ~{target:g} {unit}".strip()
    return f"{feature} ({value:g} {unit}; {suffix})"


def binding_constraint(
    tier: str,
    *,
    conditions: Mapping[str, float] | None,
    reasons: list[str] | None = None,
    thresholds: Mapping[str, float] | None = None,
    prob: float = 0.0,
) -> tuple[str, str | None]:
    """Return ``(human text, driver feature)`` for the binding constraint."""
    target_tier = next_tier(tier)
    conditions = conditions or {}
    if target_tier == "none":
        return ("maximum tier reached; no higher level exists", None)

    # Prefer live conditions, in meteorological priority order.
    candidates: list[tuple[float, str, float]] = []
    for feature in PRIORITY:
        if feature not in conditions:
            continue
        value = float(conditions[feature])
        target = FEATURE_REQUIREMENTS.get(feature, {}).get(target_tier)
        if target is None:
            continue
        shortfall = max(0.0, target - value) / max(abs(target), 1e-9)
        candidates.append((shortfall, feature, value))
    if candidates:
        _, feature, value = max(candidates, key=lambda item: item[0])
        return (_constraint_for(feature, value, target_tier), feature)

    # No live conditions: fall back to the strongest reported driver.
    if reasons:
        first = str(reasons[0])
        feature = first.split(" (")[0]
        target = FEATURE_REQUIREMENTS.get(feature, {}).get(target_tier)
        if target is not None:
            return (_constraint_for(feature, float("nan"), target_tier), feature)
        return (first, feature)

    cutoff = (thresholds or {}).get(target_tier)
    if cutoff is not None:
        return (
            f"risk score {float(prob):.2f}; {target_tier} needs {float(cutoff):.2f}",
            None,
        )
    return ("no binding constraint available", None)


def explain(
    result: Any,
    thresholds: Mapping[str, float],
    *,
    conditions: Mapping[str, float] | None = None,
) -> Explanation:
    """Explain one basin evaluation without a second click."""
    tier = str(getattr(result, "tier", "none"))
    area_id = str(getattr(result, "area_id", None) or getattr(result, "basin_id", ""))
    prob = float(getattr(result, "prob", getattr(result, "score", 0.0)))
    reasons = getattr(result, "reasons", None)
    if reasons is None:
        reasons = getattr(result, "top_drivers", None)
    reason_strings = [str(item) for item in reasons] if isinstance(reasons, (list, tuple)) else []

    if tier not in TIERS:
        tier = "none"
    following = next_tier(tier)
    gap = gap_to_next(prob, tier, thresholds)
    if tier_rank(tier) >= TIER_RANK["severe"]:
        text, feature = ("maximum tier reached; no higher level exists", None)
        following = "none"
        gap = 0.0
    else:
        text, feature = binding_constraint(
            tier,
            conditions=conditions,
            reasons=reason_strings,
            thresholds=thresholds,
            prob=prob,
        )
    return Explanation(
        area_id=area_id,
        tier=tier,
        next_tier=following,
        gap_to_next=gap,
        binding_constraint=text,
        driver_feature=feature,
    )


def explain_all(
    results: list[Any],
    thresholds: Mapping[str, float],
    *,
    conditions_by_area: Mapping[str, Mapping[str, float]] | None = None,
) -> list[Explanation]:
    conditions_by_area = conditions_by_area or {}
    return [
        explain(result, thresholds, conditions=conditions_by_area.get(result.area_id))
        for result in results
    ]


__all__ = [
    "Explanation",
    "FEATURE_REQUIREMENTS",
    "binding_constraint",
    "explain",
    "explain_all",
    "gap_to_next",
    "next_cutoff",
]
