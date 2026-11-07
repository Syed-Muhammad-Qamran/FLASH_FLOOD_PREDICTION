"""Fusion of the cloudburst (A) and melt (B) modules.

Rule (from the agreed design):
  * the higher of the two probabilities wins
  * the fused row carries the winning module's drivers
  * if both modules are at or above their `moderate` tier, the event is
    compound and the tier is escalated one level
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import thresholds_config
from .schema import TIER_RANK, parse_drivers, tier_for, validate_predictions


def _tier_cutoffs(block: dict[str, Any]) -> dict[str, float]:
    tiers = block.get("tiers", {}) or {}
    return {k: float(v) for k, v in tiers.items()}


def _escalate(tier: str, steps: int = 1) -> str:
    order = ["none", "low", "moderate", "high", "severe"]
    idx = min(order.index(tier) + steps, len(order) - 1) if tier in order else 0
    return order[idx]


def fuse(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = cfg or thresholds_config()
    fusion_cfg = cfg["fusion"]
    cutoffs_a = _tier_cutoffs(cfg["model_a"])
    cutoffs_b = _tier_cutoffs(cfg["model_b"])
    cutoffs_fused = {
        k: max(cutoffs_a.get(k, 0.0), cutoffs_b.get(k, 0.0)) for k in cutoffs_a
    }

    a = validate_predictions(pred_a).copy()
    b = validate_predictions(pred_b).copy()
    if a.empty:
        raise ValueError("pred_a is empty; nothing to fuse")
    if (a["module"] != "A").any():
        raise ValueError("pred_a must contain module == 'A'")
    if (b["module"] != "B").any():
        raise ValueError("pred_b must contain module == 'B'")

    a = a.rename(columns={"prob": "prob_a", "tier": "tier_a", "top_drivers": "drivers_a"})
    b = b.rename(columns={"prob": "prob_b", "tier": "tier_b", "top_drivers": "drivers_b"})
    a = a.sort_values("timestamp_utc")
    b = b.sort_values("timestamp_utc")

    max_age = pd.Timedelta(hours=float(fusion_cfg.get("model_b_max_age_hours", 24)))
    merged = pd.merge_asof(
        a[["basin_id", "timestamp_utc", "prob_a", "tier_a", "drivers_a"]],
        b[["basin_id", "timestamp_utc", "prob_b", "tier_b", "drivers_b"]],
        on="timestamp_utc",
        by="basin_id",
        direction="backward",
        tolerance=max_age,
    )

    rule = str(fusion_cfg.get("rule", "max")).lower()
    if rule == "noisy_or":
        fused = 1.0 - (1.0 - merged["prob_a"]) * (1.0 - merged["prob_b"].fillna(0.0))
    else:
        fused = merged[["prob_a", "prob_b"]].max(axis=1, skipna=True)
    merged["prob"] = fused.clip(0.0, 1.0)

    merged["tier"] = merged["prob"].map(lambda p: tier_for(float(p), cutoffs_fused))

    both_moderate = (
        merged["prob_a"].map(lambda p: TIER_RANK[tier_for(float(p), cutoffs_a)])
        >= TIER_RANK["moderate"]
    ) & (
        merged["prob_b"].fillna(0.0).map(lambda p: TIER_RANK[tier_for(float(p), cutoffs_b)])
        >= TIER_RANK["moderate"]
    )
    merged["compound"] = both_moderate
    if bool(fusion_cfg.get("compound_escalate", True)):
        merged.loc[both_moderate, "tier"] = merged.loc[both_moderate, "tier"].map(
            lambda t: _escalate(str(t), 1)
        )

    drivers: list[str] = []
    for row in merged.itertuples(index=False):
        a_drivers = parse_drivers(row.drivers_a)
        b_drivers = parse_drivers(row.drivers_b)
        pa = float(row.prob_a) if pd.notna(row.prob_a) else 0.0
        pb = float(row.prob_b) if pd.notna(row.prob_b) else -1.0
        if pb > pa:
            chosen = [f"melt: {d}" for d in b_drivers]
        else:
            chosen = [f"cloudburst: {d}" for d in a_drivers]
        if bool(row.compound):
            chosen = ["compound: cloudburst + melt"] + chosen
        drivers.append(json.dumps(chosen, ensure_ascii=False))

    out = pd.DataFrame(
        {
            "basin_id": merged["basin_id"],
            "timestamp_utc": merged["timestamp_utc"],
            "module": "fused",
            "prob": merged["prob"],
            "tier": merged["tier"],
            "top_drivers": drivers,
        }
    )
    out.attrs["compound_count"] = int(merged["compound"].sum())
    return validate_predictions(out)


def fuse_summary(pred_fused: pd.DataFrame) -> pd.DataFrame:
    """One row per basin: latest fused risk plus its cause."""
    df = pred_fused.sort_values("timestamp_utc")
    latest = df.groupby("basin_id", as_index=False).tail(1)
    return latest.sort_values("prob", ascending=False).reset_index(drop=True)
