#!/usr/bin/env python
"""Build the self-contained golden replay fixtures.

A clean clone has no ``data/processed`` grid and no model parquets (both are
gitignored), so the fixture pair must carry everything the replay needs. This
script:

1. slices the August 2022 validation window out of both published model files;
2. writes the two fixture parquets with full §3.7 metadata;
3. fuses them with the committed ``configs/thresholds.yaml``;
4. records the expected probability, tier and compound flag for every row.

Any later threshold edit, fusion change or weight tweak makes
``tests/test_golden_replay.py`` fail with a row-level diff.

Run: python scripts/09_build_golden.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from floodwatch import fusion, schema, thresholds  # noqa: E402

ART_A = ROOT / "artifacts" / "a" / "predictions_a.parquet"
ART_B = ROOT / "artifacts" / "b" / "predictions_b.parquet"
FEATURES_PATH = ROOT / "data" / "demo" / "features_demo.parquet"
SNAPSHOT_PATH = ROOT / "data" / "processed" / "SNAPSHOT.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "golden"
FIXTURE_A = FIXTURE_DIR / "predictions_a.parquet"
FIXTURE_B = FIXTURE_DIR / "predictions_b.parquet"
LABELS_PATH = FIXTURE_DIR / "labels.parquet"
EXPECTED_PATH = FIXTURE_DIR / "golden_expected.json"

WINDOW_START = "2022-08-24T00:00:00Z"
WINDOW_END = "2022-08-29T00:00:00Z"
MAX_FIXTURE_BYTES = 500_000


def _window(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    start = pd.Timestamp(WINDOW_START)
    end = pd.Timestamp(WINDOW_END)
    return frame[(frame["timestamp_utc"] >= start) & (frame["timestamp_utc"] < end)].copy()


def _publish(frame: pd.DataFrame, path: Path, module: str, snapshot_hash: str, seed: int) -> None:
    schema.write_model_file(
        frame.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True),
        path,
        module=module,
        metadata={
            "schema_version": schema.SCHEMA_VERSION,
            "seed": seed,
            "data_snapshot_hash": snapshot_hash,
            "label_horizon_h": schema.LABEL_HORIZON_H,
            "trained_at": datetime.now(UTC).isoformat(),
            "module": module,
            "fixture": "golden_replay",
        },
    )


def main() -> int:
    global WINDOW_START, WINDOW_END
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=WINDOW_START)
    parser.add_argument("--end", default=WINDOW_END)
    args = parser.parse_args()

    WINDOW_START, WINDOW_END = args.start, args.end

    if not ART_A.exists() or not ART_B.exists() or not SNAPSHOT_PATH.exists():
        print(
            "missing model artifacts or snapshot; run 00_build_demo.py and 04_train_b.py",
            file=sys.stderr,
        )
        return 2

    snapshot_hash = str(json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["data_snapshot_hash"])
    frame_a = _window(ART_A)
    frame_b = _window(ART_B)
    if frame_a.empty or frame_b.empty:
        print("golden window has no rows; choose a validation window", file=sys.stderr)
        return 2
    if set(frame_a["split"]) != {"validation"} or set(frame_b["split"]) != {"validation"}:
        print("golden window must lie strictly inside the validation split", file=sys.stderr)
        return 2

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _publish(frame_a, FIXTURE_A, schema.MODULE_A, snapshot_hash, seed=0)
    _publish(frame_b, FIXTURE_B, schema.MODULE_B, snapshot_hash, seed=42)

    grid = (
        pd.concat(
            [
                frame_a[["basin_id", "timestamp_utc", "split"]],
                frame_b[["basin_id", "timestamp_utc", "split"]],
            ]
        )
        .drop_duplicates(["basin_id", "timestamp_utc"])
        .reset_index(drop=True)
    )
    validated_a = schema.validate_model_file(FIXTURE_A, schema.MODULE_A, grid)
    validated_b = schema.validate_model_file(FIXTURE_B, schema.MODULE_B, grid)
    meta_a = schema.read_parquet_metadata(FIXTURE_A)
    meta_b = schema.read_parquet_metadata(FIXTURE_B)

    # Ablation needs the truth labels for the same 960 validation rows. Keeping
    # them in the golden directory makes the suite self-contained in CI.
    features = pd.read_parquet(FEATURES_PATH)
    features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True)
    labels = features[
        (features["timestamp_utc"] >= pd.Timestamp(WINDOW_START))
        & (features["timestamp_utc"] < pd.Timestamp(WINDOW_END))
    ][["basin_id", "timestamp_utc", "label"]].copy()
    labels["basin_id"] = labels["basin_id"].astype(str)
    labels = labels.merge(
        grid.astype({"basin_id": str, "split": str, "timestamp_utc": "datetime64[ns, UTC]"}),
        on=["basin_id", "timestamp_utc"],
        how="inner",
        validate="one_to_one",
    )
    labels = (
        labels[["basin_id", "timestamp_utc", "split", "label"]]
        .sort_values(["basin_id", "timestamp_utc"])
        .reset_index(drop=True)
    )
    labels.to_parquet(LABELS_PATH, index=False)

    config = thresholds.load_thresholds()
    thresholds.validate_thresholds(config)
    fused = fusion.fuse_frames(
        validated_a,
        validated_b,
        thresholds_config=config,
        meta_a=meta_a,
        meta_b=meta_b,
    )

    rows = [
        {
            "basin_id": str(row.basin_id),
            "timestamp_utc": pd.Timestamp(row.timestamp_utc).isoformat(),
            "prob": round(float(row.prob), 6),
            "tier": str(row.tier),
            "compound": bool(row.compound),
            "escalated": bool(row.escalated),
        }
        for row in fused.itertuples(index=False)
    ]
    expected = {
        "schema_version": 1,
        "window": {"start_utc": WINDOW_START, "end_utc": WINDOW_END},
        "data_snapshot_hash": snapshot_hash,
        "thresholds_path": "configs/thresholds.yaml",
        "thresholds_fingerprint": config["provenance_fingerprint"],
        "module_a_version": str(validated_a["model_version"].iloc[0]),
        "module_b_version": str(validated_b["model_version"].iloc[0]),
        "rows": rows,
    }
    EXPECTED_PATH.write_text(
        json.dumps(expected, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    sizes = {
        path.name: path.stat().st_size
        for path in (FIXTURE_A, FIXTURE_B, LABELS_PATH, EXPECTED_PATH)
    }
    print(f"golden fixtures written: {len(rows)} rows")
    for name, size in sizes.items():
        print(f"  {name}: {size} bytes")
    total = sum(sizes.values())
    print(f"  total: {total} bytes (limit {MAX_FIXTURE_BYTES})")
    if total > MAX_FIXTURE_BYTES:
        print("fixtures exceed the 500 KB budget; shrink the window", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
