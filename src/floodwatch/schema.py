"""The prediction and model-output contracts.

Two related tables live here:

1. ``validate_predictions``: the six-column prediction table used by the fused
   output and the app (``basin_id, timestamp_utc, module, prob, tier,
   top_drivers``). Extra columns are additive and allowed.

2. ``validate_model_file``: the nine-column parquet each model publishes
   (``... split, baseline_prob, source, model_version``). This is the ONLY
   interface between the two model owners; fusion asserts it before joining.

A missing grid row is a bug, not a zero.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .tiers import TIER_RANK, TIERS

SCHEMA_VERSION = 1
LABEL_HORIZON_H = 6

MODULE_A = "A_cloudburst_cnn_lstm"
MODULE_B = "B_snowmelt_lstm"
MODULE_FUSED = "fused"
MODULES: tuple[str, ...] = (MODULE_A, MODULE_B, MODULE_FUSED)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "basin_id",
    "timestamp_utc",
    "module",
    "prob",
    "tier",
    "top_drivers",
)

MODEL_FILE_COLUMNS: tuple[str, ...] = (
    "basin_id",
    "timestamp_utc",
    "module",
    "split",
    "prob",
    "baseline_prob",
    "source",
    "top_drivers",
    "model_version",
)

MODEL_FILE_METADATA_KEYS: tuple[str, ...] = (
    "schema_version",
    "seed",
    "data_snapshot_hash",
    "label_horizon_h",
    "trained_at",
    "module",
)

SPLITS: tuple[str, ...] = ("train", "validation", "test")


class SchemaError(ValueError):
    """Raised when a table violates the shared contract."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def require_columns(df: pd.DataFrame, columns: Iterable[str], *, what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{what} is missing required columns: {missing}")


def tier_for(prob: float, cutoffs: dict[str, float]) -> str:
    """Map a probability to a tier using calibrated cutoffs.

    ``cutoffs`` must contain at least ``moderate``, ``high`` and ``severe``.
    Missing cutoffs are treated as unreachable.
    """
    tier = "none"
    for name in ("low", "moderate", "high", "severe"):
        cutoff = cutoffs.get(name)
        if cutoff is not None and prob >= float(cutoff):
            tier = name
    return tier


def tier_rank(tier: str) -> int:
    return TIER_RANK.get(tier, 0)


def parse_drivers(value: object) -> list[dict[str, Any]]:
    """Parse a JSON driver list into objects; tolerate legacy list[str]."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        try:
            raw = json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(
                {"feature": str(item.get("feature", "")), "weight": float(item.get("weight", 0.0))}
            )
        else:
            out.append({"feature": str(item), "weight": 0.0})
    return out


def drivers_to_json(drivers: Iterable[dict[str, Any]]) -> str:
    ordered = sorted(drivers, key=lambda d: abs(float(d.get("weight", 0.0))), reverse=True)[:3]
    return json.dumps(
        [{"feature": str(d["feature"]), "weight": float(d["weight"])} for d in ordered],
        ensure_ascii=False,
    )


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


# ---------------------------------------------------------------------------
# prediction table (six-column contract)
# ---------------------------------------------------------------------------
def validate_predictions(
    df: pd.DataFrame,
    *,
    allow_extra: bool = True,
    require_utc: bool = True,
) -> pd.DataFrame:
    """Validate and normalise a prediction table. Returns a sorted copy."""
    require_columns(df, PREDICTION_COLUMNS, what="predictions table")
    problems: list[str] = []
    out = df.copy()

    if not allow_extra:
        extra = [c for c in out.columns if c not in PREDICTION_COLUMNS]
        if extra:
            problems.append(f"unexpected columns: {extra}")

    ts = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=require_utc)
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


# ---------------------------------------------------------------------------
# model-output file (nine-column contract, §3.7)
# ---------------------------------------------------------------------------
def read_parquet_metadata(path: str | Path) -> dict[str, str]:
    meta = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {k.decode("utf-8"): v.decode("utf-8") for k, v in meta.items()}


def write_model_file(
    df: pd.DataFrame,
    path: str | Path,
    *,
    module: str,
    metadata: dict[str, Any],
) -> Path:
    """Write a model file with the required parquet metadata.

    The caller is responsible for validating first; ``publish_model_file`` does
    both in the right order.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = dict(table.schema.metadata or {})
    for key in MODEL_FILE_METADATA_KEYS:
        if key not in metadata:
            raise SchemaError(f"model-file metadata is missing required key {key!r}")
    for key, value in metadata.items():
        existing[key.encode("utf-8")] = str(value).encode("utf-8")
    existing[b"module"] = str(module).encode("utf-8")
    table = table.replace_schema_metadata(existing)
    pq.write_table(table, p)
    return p


def validate_model_file(
    path: str | Path,
    module: str,
    grid_index: pd.DataFrame,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Enforce the §3.7 model-output contract. Returns a sorted copy."""
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"model file does not exist: {path}")
    df = pd.read_parquet(path)
    require_columns(df, MODEL_FILE_COLUMNS, what=f"model file {path.name}")
    problems: list[str] = []

    if module not in MODULES:
        raise SchemaError(f"unknown module {module!r}")
    if (df["module"].astype(str) != module).any():
        bad = sorted(set(df["module"].astype(str)) - {module})
        problems.append(f"module must be exactly {module!r}; found {bad}")

    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    if ts.isna().any():
        problems.append("timestamp_utc contains nulls or unparseable values")
    elif (ts.dt.minute != 0).any() or (ts.dt.second != 0).any():
        problems.append("timestamp_utc must be floored to the hour")
    df["timestamp_utc"] = ts

    for column in ("prob", "baseline_prob"):
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            problems.append(f"{column} contains nulls")
        if ((values < 0) | (values > 1)).any():
            problems.append(f"{column} outside [0, 1]")
        df[column] = values.astype("float64")

    bad_source = set(df["source"].astype(str)) - {"deep", "baseline"}
    if bad_source:
        problems.append(f"source must be 'deep' or 'baseline'; found {sorted(bad_source)}")

    bad_split = set(df["split"].astype(str)) - set(SPLITS)
    if bad_split:
        problems.append(f"unknown split values: {sorted(bad_split)}")

    if df.duplicated(["basin_id", "timestamp_utc"]).any():
        dup = int(df.duplicated(["basin_id", "timestamp_utc"]).sum())
        problems.append(f"{dup} duplicate (basin_id, timestamp_utc) keys")

    sorted_df = df.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)
    if not df.reset_index(drop=True).equals(sorted_df):
        problems.append("rows are not sorted by (basin_id, timestamp_utc)")

    # Grid alignment: exact key set, minus test rows unless explicitly allowed.
    require_columns(grid_index, ("basin_id", "timestamp_utc", "split"), what="grid_index")
    grid = grid_index.copy()
    grid["timestamp_utc"] = pd.to_datetime(grid["timestamp_utc"], utc=True)
    if not allow_test:
        grid = grid[grid["split"].astype(str) != "test"]
    expected = set(zip(grid["basin_id"].astype(str), grid["timestamp_utc"], strict=False))
    actual = set(zip(df["basin_id"].astype(str), df["timestamp_utc"], strict=False))
    missing = expected - actual
    extra = actual - expected
    if missing:
        sample = sorted(missing)[:3]
        problems.append(f"{len(missing)} missing grid rows, e.g. {sample}")
    if extra:
        sample = sorted(extra)[:3]
        problems.append(f"{len(extra)} rows outside the allowed grid, e.g. {sample}")

    if not allow_test and (df["split"].astype(str) == "test").any():
        problems.append("test-split rows present before M7")

    meta = read_parquet_metadata(path)
    missing_meta = [k for k in MODEL_FILE_METADATA_KEYS if k not in meta]
    if missing_meta:
        problems.append(f"parquet metadata missing keys: {missing_meta}")
    if meta.get("module") and meta["module"] != module:
        problems.append(f"parquet metadata module {meta['module']!r} != {module!r}")

    if problems:
        raise SchemaError(f"invalid model file {path}:\n  - " + "\n  - ".join(problems))

    return sorted_df


def publish_model_file(
    df: pd.DataFrame,
    path: str | Path,
    *,
    module: str,
    grid_index: pd.DataFrame,
    metadata: dict[str, Any],
) -> Path:
    """Write then validate a model file. Any failure raises before publishing."""
    write_model_file(df, path, module=module, metadata=metadata)
    validate_model_file(path, module, grid_index)
    return Path(path)
