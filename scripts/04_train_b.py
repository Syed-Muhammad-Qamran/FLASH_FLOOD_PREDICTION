#!/usr/bin/env python
"""Train Model B (snowmelt LSTM) and publish the §3.7 model output.

Outputs (all under the Model B owner directory):
  artifacts/b/model_b.pt
  artifacts/b/predictions_b.parquet
  artifacts/b/metrics_b.json
  artifacts/b/manifest.json
  tests/fixtures/golden/predictions_b.parquet

Run: python scripts/04_train_b.py [--quick] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from floodwatch import schema  # noqa: E402
from floodwatch.features import read_features  # noqa: E402
from floodwatch.models import model_b, windows  # noqa: E402

ART = ROOT / "artifacts" / "b"
PRED_PATH = ART / "predictions_b.parquet"
METRICS_PATH = ART / "metrics_b.json"
MANIFEST_PATH = ART / "manifest.json"
CHECKPOINT_PATH = ART / "model_b.pt"
SNAPSHOT_PATH = ROOT / "data" / "processed" / "SNAPSHOT.json"
GRID_PATH = ROOT / "data" / "processed" / "grid_index.parquet"
FEATURES_PATH = ROOT / "data" / "demo" / "features_demo.parquet"
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "golden" / "predictions_b.parquet"
GOLDEN_START = "2022-08-24T00:00:00Z"
GOLDEN_END = "2022-08-29T00:00:00Z"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "nogit00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_golden(predictions: pd.DataFrame) -> None:
    """Small August 2022 validation window from the frozen validation years."""
    frame = predictions.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    window = frame[
        (frame["timestamp_utc"] >= pd.Timestamp(GOLDEN_START))
        & (frame["timestamp_utc"] < pd.Timestamp(GOLDEN_END))
    ]
    if window.empty:
        print("warning: no golden window rows; fixture not written")
        return
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    window[
        [
            "basin_id",
            "timestamp_utc",
            "module",
            "split",
            "prob",
            "baseline_prob",
            "source",
            "top_drivers",
            "model_version",
        ]
    ].to_parquet(GOLDEN_PATH, index=False)
    print(f"golden fixture: {len(window)} rows -> {GOLDEN_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer epochs for a local smoke run")
    parser.add_argument("--seed", type=int, default=model_b.DEFAULT_SEED)
    parser.add_argument("--hidden-units", type=int, default=48)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not FEATURES_PATH.exists() or not GRID_PATH.exists():
        print("missing demo data; run: python scripts/00_build_demo.py --force", file=sys.stderr)
        return 2

    features = read_features(FEATURES_PATH)
    grid = windows.read_grid_index(GRID_PATH)
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    snapshot_hash = str(snapshot["data_snapshot_hash"])

    cfg = model_b.TrainConfig(seed=args.seed, hidden_units=args.hidden_units)
    result = model_b.train_model_b(features, grid, cfg=cfg, quick=args.quick, verbose=args.verbose)

    # Fit the degree-day baseline on train summaries only.
    summary = model_b.baseline_inputs(features)
    summary = summary.merge(
        grid[["basin_id", "timestamp_utc", "split"]],
        on=["basin_id", "timestamp_utc"],
        how="left",
    )
    train_mask = summary["split"].fillna("").eq("train")
    state = model_b.fit_baseline(summary, train_mask=train_mask)

    model_version = f"{schema.MODULE_B}-{git_sha()}-{snapshot_hash[:8]}"
    deep_prob_table = model_b.predict_table(
        features,
        grid,
        model=result["model"],
        state=state,
        cfg=result["cfg"],
        norm=result["norm"],
        model_version=model_version,
        snapshot_hash=snapshot_hash,
        source="deep",
    )
    truth = features[["basin_id", "timestamp_utc", "label"]]
    comparison = model_b.validation_metrics(deep_prob_table, truth)

    # Baseline wins -> publish baseline probabilities, state it in the manifest.
    source = "deep" if comparison["deep_wins"] else "baseline"
    predictions = deep_prob_table.copy()
    predictions["source"] = source
    if source == "baseline":
        predictions["prob"] = predictions["baseline_prob"]

    ART.mkdir(parents=True, exist_ok=True)
    schema.publish_model_file(
        predictions,
        PRED_PATH,
        module=schema.MODULE_B,
        grid_index=grid,
        metadata={
            "schema_version": schema.SCHEMA_VERSION,
            "seed": cfg.seed,
            "data_snapshot_hash": snapshot_hash,
            "label_horizon_h": schema.LABEL_HORIZON_H,
            "trained_at": datetime.now(UTC).isoformat(),
            "module": schema.MODULE_B,
        },
    )

    model_b.save_checkpoint(
        CHECKPOINT_PATH,
        model=result["model"],
        state=state,
        cfg=result["cfg"],
        norm=result["norm"],
        metadata={
            "model_version": model_version,
            "data_snapshot_hash": snapshot_hash,
            "seed": cfg.seed,
            "source": source,
        },
    )

    metrics = {
        "schema_version": 1,
        "module": schema.MODULE_B,
        "model_version": model_version,
        "data_snapshot_hash": snapshot_hash,
        "seed": cfg.seed,
        "trained_on": "train",
        "evaluated_on": "validation",
        "display_threshold": 0.2,
        "deep": comparison["deep"],
        "baseline": comparison["baseline"],
        "baseline_wins": not comparison["deep_wins"],
        "published_source": source,
        "best_epoch": result["best_epoch"],
        "train_rows": result["train_rows"],
        "validation_rows": comparison["rows"],
        "synthetic_data": True,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "module": schema.MODULE_B,
                "model_version": model_version,
                "data_snapshot_hash": snapshot_hash,
                "seed": cfg.seed,
                "rows": int(len(predictions)),
                "sha256": sha256(PRED_PATH),
                "validated_at": datetime.now(UTC).isoformat(),
                "validation_metrics_path": "artifacts/b/metrics_b.json",
                "baseline_wins": not comparison["deep_wins"],
                "published_source": source,
                "architecture": {
                    "lstm_layers": model_b.MAX_LSTM_LAYERS,
                    "hidden_units": cfg.hidden_units,
                    "dropout": cfg.dropout,
                    "lookback_h": cfg.lookback_h,
                    "features": list(cfg.features),
                },
                "synthetic_data": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_golden(predictions)
    print(
        f"Model B published: source={source} rows={len(predictions)} "
        f"val_pr_auc={comparison['deep'].get('pr_auc')} "
        f"baseline_pr_auc={comparison['baseline'].get('pr_auc')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
