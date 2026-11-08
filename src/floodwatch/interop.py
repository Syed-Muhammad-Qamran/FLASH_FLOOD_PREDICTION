"""Interoperability adapter: fused prediction rows -> typed domain objects.

The only module that knows both the tabular vocabulary and the UI vocabulary.
It never changes a probability, never calls the network and never writes a file.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from . import schema
from .tiers import Level, tier_to_level

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "configs" / "basins.yaml"

COMPOUND_REASON = "compound: cloudburst and snowmelt both active"


@dataclass(frozen=True)
class Area:
    id: str
    name: str
    district: str
    lat: float
    lon: float
    terrain_susceptibility: float = 0.5
    glacial_lake_nearby: bool = False
    snow_cover_frac: float = 0.0


@dataclass(frozen=True)
class RiskResult:
    area_id: str
    score: float
    level: Level
    reasons: list[str] = field(default_factory=list)
    computed_at: str = ""
    tier: str = "none"
    compound: bool = False
    prob_a: float | None = None
    prob_b: float | None = None
    escalated: bool = False
    provisional: bool = False
    prob_std: float | None = None
    tier_stable: bool | None = None


def _load_catalog(path: str | Path | None = None) -> dict[str, Area]:
    data = yaml.safe_load(Path(path or DEFAULT_CATALOG).read_text(encoding="utf-8"))
    out: dict[str, Area] = {}
    for basin in data.get("basins", []):
        area_id = str(basin["basin_id"])
        out[area_id] = Area(
            id=area_id,
            name=str(basin.get("name", area_id)),
            district=str(basin.get("district", "")),
            lat=float(basin["lat"]),
            lon=float(basin["lon"]),
            terrain_susceptibility=float(basin.get("terrain_susceptibility", 0.5)),
            glacial_lake_nearby=bool(basin.get("glacial_lake_nearby", False)),
            snow_cover_frac=float(basin.get("snow_fraction", 0.0)),
        )
    return out


def areas(catalog: dict[str, Area] | None = None) -> dict[str, Area]:
    return dict(catalog or _load_catalog())


def _reason_strings(top_drivers: object) -> list[str]:
    reasons = []
    for driver in schema.parse_drivers(top_drivers):
        feature = driver["feature"]
        weight = float(driver["weight"])
        reasons.append(f"{feature} ({weight:g})" if weight else feature)
    return reasons


def _lookup(
    frame: pd.DataFrame | None, area_id: str, when: pd.Timestamp, column: str
) -> float | None:
    if frame is None or frame.empty:
        return None
    subset = frame[
        (frame["basin_id"].astype(str) == area_id)
        & (pd.to_datetime(frame["timestamp_utc"], utc=True) == when)
    ]
    if subset.empty or column not in subset:
        return None
    return float(subset.iloc[0][column])


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN check


def to_risk_results(
    fused: pd.DataFrame,
    *,
    catalog: dict[str, Area] | None = None,
    pred_a: pd.DataFrame | None = None,
    pred_b: pd.DataFrame | None = None,
    computed_at: str | None = None,
) -> list[RiskResult]:
    """One ``RiskResult`` per monitored basin at its newest fused timestep."""
    if fused is None or fused.empty:
        return []
    frame = schema.validate_predictions(fused)
    frame = frame.sort_values("timestamp_utc")
    latest = frame.groupby("basin_id", as_index=False).tail(1)
    catalog = areas(catalog)

    if computed_at is None:
        computed_at = pd.Timestamp(latest["timestamp_utc"].max()).isoformat()

    has_prob_a = "prob_a" in latest.columns
    has_prob_b = "prob_b" in latest.columns
    results: list[RiskResult] = []
    for row in latest.itertuples(index=False):
        area_id = str(row.basin_id)
        compound = bool(getattr(row, "compound", False))
        escalated = bool(getattr(row, "escalated", False))
        reasons = _reason_strings(row.top_drivers)
        if compound:
            reasons = [COMPOUND_REASON] + reasons
        if escalated:
            reasons = (
                reasons[:1] + ["rain-on-snow escalation: liquid water on a snowpack"] + reasons[1:]
            )
        if not reasons:
            reasons = ["no dominant driver above threshold"]
        when = pd.Timestamp(row.timestamp_utc)
        prob_a = (
            _float_or_none(getattr(row, "prob_a", None))
            if has_prob_a
            else _lookup(pred_a, area_id, when, "prob")
        )
        prob_b = (
            _float_or_none(getattr(row, "prob_b", None))
            if has_prob_b
            else _lookup(pred_b, area_id, when, "prob")
        )
        results.append(
            RiskResult(
                area_id=area_id,
                score=float(row.prob),
                level=tier_to_level(str(row.tier)),
                reasons=reasons,
                computed_at=str(computed_at),
                tier=str(row.tier),
                compound=compound,
                prob_a=prob_a,
                prob_b=prob_b,
                escalated=bool(getattr(row, "escalated", False)),
                provisional=bool(getattr(row, "provisional", False)),
                prob_std=_float_or_none(getattr(row, "prob_std", None)),
                tier_stable=(
                    None if getattr(row, "tier_stable", None) is None else bool(row.tier_stable)
                ),
            )
        )
    results.sort(key=lambda r: r.area_id)
    missing = sorted(set(catalog) - {r.area_id for r in results})
    if missing:
        # The app may run before every basin has a row; surface rather than crash.
        for area_id in missing:
            results.append(
                RiskResult(
                    area_id=area_id,
                    score=0.0,
                    level=Level.GREEN,
                    reasons=["no prediction for this basin in the selected run"],
                    computed_at=str(computed_at),
                    tier="none",
                )
            )
    return results


def risk_frame(
    results: Iterable[RiskResult], catalog: dict[str, Area] | None = None
) -> pd.DataFrame:
    catalog = areas(catalog)
    rows = []
    for result in results:
        area = catalog.get(result.area_id)
        rows.append(
            {
                "area_id": result.area_id,
                "name": area.name if area else result.area_id,
                "district": area.district if area else "",
                "level": result.level.name,
                "level_value": int(result.level),
                "score": result.score,
                "tier": result.tier,
                "compound": result.compound,
                "prob_a": result.prob_a,
                "prob_b": result.prob_b,
                "prob_std": result.prob_std,
                "tier_stable": result.tier_stable,
                "provisional": result.provisional,
                "reasons": "; ".join(result.reasons),
                "computed_at": result.computed_at,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["level_value", "score"], ascending=[False, False]).reset_index(
        drop=True
    )


def levels_from(results: Iterable[RiskResult]) -> dict[str, Level]:
    return {result.area_id: result.level for result in results}


def reason_text(result: RiskResult) -> str:
    return "; ".join(result.reasons)


__all__ = [
    "Area",
    "RiskResult",
    "areas",
    "levels_from",
    "reason_text",
    "risk_frame",
    "to_risk_results",
]
