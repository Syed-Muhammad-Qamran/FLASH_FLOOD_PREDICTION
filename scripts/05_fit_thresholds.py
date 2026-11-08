#!/usr/bin/env python
"""Fit risk-tier thresholds on train rows only and write configs/thresholds.yaml.

D-01: validation rows are used for the drift check only. A breach is recorded
and escalated; this script never refits silently on validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from floodwatch import fusion, schema, thresholds  # noqa: E402
from floodwatch.features import read_features  # noqa: E402
from floodwatch.models import windows  # noqa: E402

GRID_PATH = ROOT / "data" / "processed" / "grid_index.parquet"
FEATURES_PATH = ROOT / "data" / "demo" / "features_demo.parquet"
SNAPSHOT_PATH = ROOT / "data" / "processed" / "SNAPSHOT.json"


def labelled(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    truth = features[["basin_id", "timestamp_utc", "label"]].copy()
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True)
    frame = predictions.merge(
        truth, on=["basin_id", "timestamp_utc"], how="left", validate="one_to_one"
    )
    if frame["label"].isna().any():
        raise ValueError("model output rows are missing labels; grid and features disagree")
    frame["label"] = frame["label"].astype(int)
    return frame


def fit_module(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    target_far: float,
    min_pod: float,
) -> dict:
    frame = labelled(predictions, features)
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    fit = thresholds.fit_tier_cutoffs(train, target_far=target_far, min_pod=min_pod)
    cutoff = fit["tiers"]["high"]
    train_metrics = thresholds.metrics_at_cutoff(train, cutoff)
    validation_metrics = thresholds.metrics_at_cutoff(validation, cutoff)
    drift = thresholds.validation_guard(train_metrics, validation_metrics)
    return {
        "tiers": fit["tiers"],
        "operating_point": fit["operating_point"],
        "fit_rows": fit["rows"],
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "drift": drift,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-far", type=float, default=thresholds.DEFAULT_TARGET_FAR)
    parser.add_argument("--min-pod", type=float, default=thresholds.DEFAULT_MIN_POD)
    parser.add_argument("--out", default=str(thresholds.DEFAULT_PATH))
    args = parser.parse_args()

    grid = windows.read_grid_index(GRID_PATH)
    features = read_features(FEATURES_PATH)
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    snapshot_hash = str(snapshot["data_snapshot_hash"])

    pred_a, meta_a = fusion.load_model_file(
        ROOT / "artifacts" / "a" / "predictions_a.parquet", schema.MODULE_A, grid
    )
    pred_b, meta_b = fusion.load_model_file(
        ROOT / "artifacts" / "b" / "predictions_b.parquet", schema.MODULE_B, grid
    )
    fusion.assert_compatible(pred_a, pred_b, meta_a=meta_a, meta_b=meta_b)

    modules = {
        schema.MODULE_A: fit_module(
            pred_a, features, target_far=args.target_far, min_pod=args.min_pod
        ),
        schema.MODULE_B: fit_module(
            pred_b, features, target_far=args.target_far, min_pod=args.min_pod
        ),
    }

    # Seed the fused block from the max-probability rule, then refit on the
    # actual fused rows (compound/escalation can only raise a tier).
    seed_frame = pred_a[["basin_id", "timestamp_utc", "split", "prob"]].merge(
        pred_b[["basin_id", "timestamp_utc", "prob"]],
        on=["basin_id", "timestamp_utc"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    seed_frame["prob"] = np.maximum(seed_frame["prob_a"], seed_frame["prob_b"])
    seed = thresholds.fit_tier_cutoffs(
        labelled(seed_frame[["basin_id", "timestamp_utc", "split", "prob"]], features).loc[
            lambda d: d["split"] == "train"
        ],
        target_far=args.target_far,
        min_pod=args.min_pod,
    )
    modules[schema.MODULE_FUSED] = {
        "tiers": seed["tiers"],
        "operating_point": seed["operating_point"],
        "fit_rows": seed["rows"],
        "train_metrics": thresholds.metrics_at_cutoff(
            labelled(seed_frame, features).query("split == 'train'"), seed["tiers"]["high"]
        ),
        "validation_metrics": thresholds.metrics_at_cutoff(
            labelled(seed_frame, features).query("split == 'validation'"), seed["tiers"]["high"]
        ),
        "drift": {"pass": True, "action": "ok", "note": "seeded from max-probability rule"},
    }
    provisional = thresholds.build_config(
        modules,
        data_snapshot_hash=snapshot_hash,
        target_far=args.target_far,
        min_pod=args.min_pod,
    )

    fused = fusion.fuse_frames(
        pred_a,
        pred_b,
        thresholds_config=provisional,
        meta_a=meta_a,
        meta_b=meta_b,
    )
    modules[schema.MODULE_FUSED] = fit_module(
        fused[fused["split"] != "test"], features, target_far=args.target_far, min_pod=args.min_pod
    )

    config = thresholds.build_config(
        modules,
        data_snapshot_hash=snapshot_hash,
        target_far=args.target_far,
        min_pod=args.min_pod,
    )
    thresholds.validate_thresholds(config)
    path = thresholds.save_thresholds(config, args.out)

    breaches = thresholds.drift_breaches(config)
    print(f"thresholds written: {path}")
    for name, block in modules.items():
        print(
            f"  {name}: high={block['tiers']['high']:.4f} "
            f"drift_pass={block['drift']['pass']} "
            f"pod_train={block['drift'].get('pod_train')} "
            f"pod_val={block['drift'].get('pod_validation')}"
        )
    if breaches:
        print(f"  D-01 drift breach (escalate as a joint decision): {breaches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
