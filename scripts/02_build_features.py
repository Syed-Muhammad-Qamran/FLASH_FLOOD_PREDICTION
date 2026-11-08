"""
CLI Script: 02_build_features.py
================================
Transforms raw features into rolling windows and canonical grid index.
Usage:
    python scripts/02_build_features.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from floodwatch.features import build_feature_table
from floodwatch.models.windows import (
    CORE_TEMPORAL_FEATURES,
    create_basin_windows,
    export_canonical_grid,
    fit_train_scaler,
)


def assign_splits(df: pd.DataFrame, split_cfg_path: str = "configs/split.yaml") -> pd.DataFrame:
    """Assigns train/val/test splits based on split.yaml timestamps."""
    with open(split_cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = cfg.get("splits", {})
    train_end = pd.to_datetime(splits["train"]["end_utc"])
    val_start = pd.to_datetime(splits["validation"]["start_utc"])
    val_end = pd.to_datetime(splits["validation"]["end_utc"])
    test_start = pd.to_datetime(splits["test"]["start_utc"])

    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    conds = [(ts <= train_end), (ts >= val_start) & (ts <= val_end), (ts >= test_start)]
    choices = ["train", "validation", "test"]
    df["split"] = np.select(conds, choices, default="buffer")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build features and windows.")
    parser.add_argument(
        "--input", default="data/processed/raw_features.parquet", help="Raw features input"
    )
    parser.add_argument("--out-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    in_file = Path(args.input)
    if not in_file.exists():
        print(f"Input file {in_file} not found. Running fetch first...")
        import importlib

        fetch_mod = importlib.import_module("scripts.01_fetch_openmeteo")
        fetch_mod.main()

    print("Loading raw features...")
    df = pd.read_parquet(in_file)
    df = assign_splits(df)

    processed_basins = []
    for b_id, grp in df.groupby("basin_id"):
        print(f"  Engineering features for {b_id}...")
        feat_df = build_feature_table(grp)
        processed_basins.append(feat_df)

    full_features = pd.concat(processed_basins, ignore_index=True)

    # Fit scaler strictly on train split
    train_subset = full_features[full_features["split"] == "train"]
    print("Fitting normalization scalers strictly on train split...")
    scaler = fit_train_scaler(train_subset, CORE_TEMPORAL_FEATURES)
    full_scaled = scaler.transform(full_features, CORE_TEMPORAL_FEATURES)

    # Construct windows & canonical grid
    meta_records = []
    for _b_id, grp in full_scaled.groupby("basin_id"):
        _, _, _, meta = create_basin_windows(grp)
        meta_records.append(meta)

    all_meta = pd.concat(meta_records, ignore_index=True)
    grid = export_canonical_grid(all_meta, Path(args.out_dir) / "grid_index.parquet")
    print(f"Canonical grid index written: {len(grid):,} rows.")
    print(f"  Train rows: {(grid['split'] == 'train').sum():,}")
    print(f"  Validation rows: {(grid['split'] == 'validation').sum():,}")
    print(f"  Test rows: {(grid['split'] == 'test').sum():,}")


if __name__ == "__main__":
    main()
