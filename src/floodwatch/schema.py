"""The prediction contract.

Every module writes exactly this table. If a script cannot produce it, the
script is wrong, not the contract.

    basin_id      str
    timestamp_utc datetime64[ns, UTC]
    module        {"A", "B", "fused"}
    prob          float in [0, 1]
    tier          {"none", "low", "moderate", "high", "severe"}
    top_drivers   JSON-encoded list[str]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "basin_id",
    "timestamp_utc",
    "module",
    "prob",
    "tier",
    "top_drivers",
)

MODULES: tuple[str, ...] = ("A", "B", "fused")
TIERS: tuple[str, ...] = ("none", "low", "moderate", "high", "severe")
TIER_RANK = {t: i for i, t in enumerate(TIERS)}


class SchemaError(ValueError):
    """Raised when a predictions table violates the shared contract."""


def empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "basin_id": pd.Series(dtype="string"),
            "timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "module": pd.Series(dtype="string"),
            "prob": pd.Series(dtype="float64"),
            "tier": pd.Series(dtype="string"),
            "top_drivers": pd.Series(dtype="string"),
        }
    )


def validate_predictions(
    df: pd.DataFrame,
    *,
    allow_extra: bool = True,
    require_utc: bool = True,
) -> pd.DataFrame:
    """Validate and normalise a predictions table. Returns a copy."""
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    if not allow_extra:
        extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
        if extra:
            problems.append(f"unexpected columns: {extra}")

    out = df.copy()

    # timestamp_utc must be tz-aware UTC. Naive timestamps are rejected on
    # purpose: silently localising them is how a 5-hour bug reaches an alert.
    ts = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=False)
    if require_utc:
        if getattr(ts.dt, "tz", None) is None:
            problems.append(
                "timestamp_utc is timezone-naive; attach UTC explicitly "
                "(pd.to_datetime(..., utc=True))"
            )
        else:
            ts = ts.dt.tz_convert("UTC")
    out["timestamp_utc"] = ts

    out["basin_id"] = out["basin_id"].astype("string")
    out["module"] = out["module"].astype("string")
    out["tier"] = out["tier"].astype("string")

    if out["basin_id"].isna().any():
        problems.append("basin_id contains nulls")
    if out["timestamp_utc"].isna().any():
        problems.append("timestamp_utc contains nulls or unparseable values")

    bad_module = set(out["module"].dropna().unique()) - set(MODULES)
    if bad_module:
        problems.append(f"unknown module values: {sorted(bad_module)}")
    bad_tier = set(out["tier"].dropna().unique()) - set(TIERS)
    if bad_tier:
        problems.append(f"unknown tier values: {sorted(bad_tier)}")

    prob = pd.to_numeric(out["prob"], errors="coerce")
    if prob.isna().any():
        problems.append("prob contains nulls or non-numeric values")
    if ((prob < 0) | (prob > 1)).any():
        problems.append("prob outside [0, 1]")
    out["prob"] = prob.astype("float64")

    def _check_drivers(value: object) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value), ensure_ascii=False)
        if not isinstance(value, str):
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list):
            return None
        return json.dumps(parsed, ensure_ascii=False)

    drivers = out["top_drivers"].map(_check_drivers)
    if drivers.isna().any():
        problems.append("top_drivers must be a JSON list or list-like")
    out["top_drivers"] = drivers.astype("string")

    if problems:
        raise SchemaError("invalid predictions table:\n  - " + "\n  - ".join(problems))

    return out.sort_values(["timestamp_utc", "basin_id", "module"]).reset_index(drop=True)


def write_predictions(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    validate_predictions(df).to_parquet(p, index=False)
    return p


def read_predictions(path: str | Path) -> pd.DataFrame:
    return validate_predictions(pd.read_parquet(path))


def tier_for(prob: float, cutoffs: dict[str, float]) -> str:
    """Map a probability to a tier using calibrated cutoffs."""
    tier = "none"
    for name in ("low", "moderate", "high", "severe"):
        cutoff = cutoffs.get(name)
        if cutoff is not None and prob >= float(cutoff):
            tier = name
    return tier


def tier_rank(tier: str) -> int:
    return TIER_RANK.get(tier, 0)


def parse_drivers(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def concat(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [p for p in parts if p is not None and not p.empty]
    if not frames:
        return empty_predictions()
    return pd.concat(frames, ignore_index=True)
