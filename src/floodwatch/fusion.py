"""Two-mechanism fusion over validated model-output files.

This module never imports ``model_a``/``model_b`` and never retrains. It reads
only the §3.7 parquet contract, asserts that both files describe exactly the
same grid and data snapshot, then applies the agreed rules:

* ``prob_fused = max(prob_a, prob_b)``
* ``compound`` when both mechanisms are at or above their own alert cutoffs
* optional rain-on-snow escalation when a compound event carries a
  rain-on-snow driver
* top drivers merged and ranked by ``abs(weight)``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import schema
from .thresholds import module_tiers
from .tiers import TIER_RANK, TIERS, escalate, max_tier

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A = ROOT / "artifacts" / "a" / "predictions_a.parquet"
DEFAULT_B = ROOT / "artifacts" / "b" / "predictions_b.parquet"

RAIN_ON_SNOW_FEATURES = frozenset(
    {"rain_on_snow", "snowmelt_mm", "degree_day_melt", "snow_depth_mm"}
)


class FusionError(ValueError):
    """Raised when the two model files cannot be fused safely."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def noisy_or(prob_a: np.ndarray | pd.Series, prob_b: np.ndarray | pd.Series) -> np.ndarray:
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    return 1.0 - (1.0 - a) * (1.0 - b)


def _merge_drivers(
    drivers_a: list[dict[str, Any]], drivers_b: list[dict[str, Any]], *, limit: int = 3
) -> list[dict[str, float]]:
    merged: dict[str, float] = {}
    for item in list(drivers_a) + list(drivers_b):
        feature = str(item.get("feature", ""))
        if not feature:
            continue
        merged[feature] = merged.get(feature, 0.0) + float(item.get("weight", 0.0))
    ordered = sorted(merged.items(), key=lambda kv: abs(kv[1]), reverse=True)[:limit]
    return [{"feature": feature, "weight": round(weight, 4)} for feature, weight in ordered]


def has_rain_on_snow(drivers: list[dict[str, Any]]) -> bool:
    return any(
        str(item.get("feature", "")) in RAIN_ON_SNOW_FEATURES
        and abs(float(item.get("weight", 0.0))) > 0.0
        for item in drivers
    )


def key_set(df: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    return set(
        zip(df["basin_id"].astype(str), pd.to_datetime(df["timestamp_utc"], utc=True), strict=False)
    )


def _describe_diff(
    expected: set[tuple[str, pd.Timestamp]], actual: set[tuple[str, pd.Timestamp]], label: str
) -> str:
    missing = sorted(expected - actual)[:3]
    extra = sorted(actual - expected)[:3]
    parts = []
    if missing:
        parts.append(f"{len(expected - actual)} missing e.g. {missing}")
    if extra:
        parts.append(f"{len(actual - expected)} extra e.g. {extra}")
    return f"{label}: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# compatibility assertions
# ---------------------------------------------------------------------------
def assert_compatible(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    *,
    meta_a: dict[str, str] | None = None,
    meta_b: dict[str, str] | None = None,
) -> None:
    problems: list[str] = []

    bad_a = set(pred_a["module"].astype(str)) - {schema.MODULE_A}
    bad_b = set(pred_b["module"].astype(str)) - {schema.MODULE_B}
    if bad_a:
        problems.append(f"A file module must be {schema.MODULE_A!r}; found {sorted(bad_a)}")
    if bad_b:
        problems.append(f"B file module must be {schema.MODULE_B!r}; found {sorted(bad_b)}")

    keys_a = key_set(pred_a)
    keys_b = key_set(pred_b)
    if keys_a != keys_b:
        problems.append(_describe_diff(keys_a, keys_b, "A vs B grid mismatch"))

    if meta_a is not None and meta_b is not None:
        hash_a = meta_a.get("data_snapshot_hash")
        hash_b = meta_b.get("data_snapshot_hash")
        if not hash_a or not hash_b:
            problems.append("both model files must declare data_snapshot_hash metadata")
        elif hash_a != hash_b:
            problems.append(
                f"data_snapshot_hash mismatch: A={hash_a!r} B={hash_b!r}; retrain both models"
            )
    if problems:
        raise FusionError("cannot fuse model outputs:\n  - " + "\n  - ".join(problems))


def load_model_file(
    path: str | Path, module: str, grid_index: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    path = Path(path)
    df = schema.validate_model_file(path, module, grid_index)
    meta = schema.read_parquet_metadata(path)
    return df, meta


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------
def fuse_frames(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    *,
    thresholds_config: dict[str, Any],
    meta_a: dict[str, str] | None = None,
    meta_b: dict[str, str] | None = None,
    rain_on_snow_escalation: bool = True,
    include_tiers: tuple[str, ...] = ("train", "validation"),
) -> pd.DataFrame:
    """Fuse two validated model frames into the six-column prediction contract."""
    a = pred_a.copy()
    b = pred_b.copy()
    a["timestamp_utc"] = pd.to_datetime(a["timestamp_utc"], utc=True)
    b["timestamp_utc"] = pd.to_datetime(b["timestamp_utc"], utc=True)

    if include_tiers is not None:
        a = a[a["split"].astype(str).isin(include_tiers)].copy()
        b = b[b["split"].astype(str).isin(include_tiers)].copy()

    assert_compatible(a, b, meta_a=meta_a, meta_b=meta_b)

    cut_a = module_tiers(thresholds_config, schema.MODULE_A)
    cut_b = module_tiers(thresholds_config, schema.MODULE_B)
    cut_fused = module_tiers(thresholds_config, schema.MODULE_FUSED)

    merged = a.merge(
        b,
        on=["basin_id", "timestamp_utc"],
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if merged.empty:
        return schema.empty_predictions().assign(
            split=pd.Series(dtype="string"),
            compound=pd.Series(dtype="bool"),
            prob_a=pd.Series(dtype="float64"),
            prob_b=pd.Series(dtype="float64"),
            tier_a=pd.Series(dtype="string"),
            tier_b=pd.Series(dtype="string"),
            escalated=pd.Series(dtype="bool"),
            data_snapshot_hash=pd.Series(dtype="string"),
        )

    prob_a = merged["prob_a"].astype(float)
    prob_b = merged["prob_b"].astype(float)
    tier_a = [schema.tier_for(value, cut_a) for value in prob_a]
    tier_b = [schema.tier_for(value, cut_b) for value in prob_b]

    fused_prob = np.maximum(prob_a.to_numpy(), prob_b.to_numpy())
    compound = (prob_a.to_numpy() >= cut_a["high"]) & (prob_b.to_numpy() >= cut_b["high"])

    drivers_a = merged["top_drivers_a"].map(schema.parse_drivers).tolist()
    drivers_b = merged["top_drivers_b"].map(schema.parse_drivers).tolist()

    rows: list[dict[str, Any]] = []
    for i in range(len(merged)):
        base_tier = schema.tier_for(float(fused_prob[i]), cut_fused)
        # Never let fusion downgrade the stronger mechanism's tier.
        base_tier = max_tier(base_tier, tier_a[i], tier_b[i])
        is_compound = bool(compound[i])
        reason_drivers = _merge_drivers(drivers_a[i], drivers_b[i])
        escalated = False
        if (
            is_compound
            and rain_on_snow_escalation
            and (has_rain_on_snow(drivers_a[i]) or has_rain_on_snow(drivers_b[i]))
        ):
            base_tier = escalate(base_tier, 1)
            escalated = True
        rows.append(
            {
                "basin_id": str(merged.at[i, "basin_id"]),
                "timestamp_utc": merged.at[i, "timestamp_utc"],
                "module": schema.MODULE_FUSED,
                "split": str(merged.at[i, "split_a"]),
                "prob": round(float(fused_prob[i]), 6),
                "tier": base_tier,
                "top_drivers": schema.drivers_to_json(reason_drivers[:3]),
                "compound": is_compound,
                "prob_a": round(float(prob_a.iloc[i]), 6),
                "prob_b": round(float(prob_b.iloc[i]), 6),
                "tier_a": tier_a[i],
                "tier_b": tier_b[i],
                "escalated": escalated,
                "data_snapshot_hash": (meta_a or {}).get("data_snapshot_hash", ""),
            }
        )

    out = pd.DataFrame(rows)
    return schema.validate_predictions(out)


def fuse(
    path_a: str | Path = DEFAULT_A,
    path_b: str | Path = DEFAULT_B,
    *,
    grid_index: pd.DataFrame,
    thresholds_config: dict[str, Any],
    rain_on_snow_escalation: bool = True,
) -> pd.DataFrame:
    """Read both §3.7 files, validate the interface and return fused rows."""
    pred_a, meta_a = load_model_file(path_a, schema.MODULE_A, grid_index)
    pred_b, meta_b = load_model_file(path_b, schema.MODULE_B, grid_index)
    return fuse_frames(
        pred_a,
        pred_b,
        thresholds_config=thresholds_config,
        meta_a=meta_a,
        meta_b=meta_b,
        rain_on_snow_escalation=rain_on_snow_escalation,
    )


def tier_index(tier: str) -> int:
    return TIER_RANK.get(tier, 0)


def describe_fusion_rules(thresholds_config: dict[str, Any]) -> str:
    cut_a = module_tiers(thresholds_config, schema.MODULE_A)
    cut_b = module_tiers(thresholds_config, schema.MODULE_B)
    return (
        "max(prob_a, prob_b); compound when "
        f"A>={cut_a['high']:.3f} and B>={cut_b['high']:.3f}; "
        "rain-on-snow escalates a compound event one tier"
    )


__all__ = [
    "FusionError",
    "assert_compatible",
    "fuse",
    "fuse_frames",
    "has_rain_on_snow",
    "load_model_file",
    "noisy_or",
    "tier_index",
    "TIERS",
]
