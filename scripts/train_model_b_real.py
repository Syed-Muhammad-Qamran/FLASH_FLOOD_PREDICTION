#!/usr/bin/env python
"""Reproducible Real-Data Training and External Event Evaluation for Model B.
=============================================================================
Mechanism: Snowmelt and Rain-on-Snow Flood Risk (Model B)
Architecture: Single-layer LSTM (32-64 units) + Degree-Day Heuristic Baseline
Owner: Person 2 (Warisha) / Core ML Lane

Key Invariants:
  1. Real historical weather from Open-Meteo ERA5 (temperature, snow depth, rain-on-snow).
  2. Baseline is deterministic degree-day heuristic (DegreeDayBaseline).
  3. Preprocessing fit strictly on train split rows.
  4. 2010 and 2022 floods are strictly external holdouts.
  5. Ground truth impacts cite official NDMA/PDMA/FFC reports.

Usage:
  python scripts/train_model_b_real.py [--quick] [--offline] [--epochs 15]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from floodwatch.data_sources import cache, openmeteo  # noqa: E402
from floodwatch.features import add_features, load_basins, validate_no_leakage  # noqa: E402
from floodwatch.models.eval_cases import evaluate_case  # noqa: E402
from floodwatch.models.model_b import (  # noqa: E402
    TrainConfig,
    build_baseline,
    build_model,
)

CONFIG_PATH = ROOT / "configs" / "real_data_training.yaml"
ARTIFACTS_DIR = ROOT / "artifacts" / "b"


class TemporalDatasetB(Dataset):
    def __init__(
        self, sequences: dict[str, np.ndarray], index_df: pd.DataFrame, lookback: int = 72
    ):
        self.sequences = {k: torch.tensor(v, dtype=torch.float32) for k, v in sequences.items()}
        self.index_df = index_df.reset_index(drop=True)
        self.lookback = lookback
        self.basin_ids = self.index_df["basin_id"].values
        self.step_indices = self.index_df["step_idx"].values
        self.targets = torch.tensor(self.index_df["target"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.basin_ids[idx]
        s = self.step_indices[idx]
        x_seq = self.sequences[b][s - self.lookback : s]
        return x_seq, self.targets[idx]


def fetch_or_load_real_weather_b(
    basins: list[dict],
    start_date: str,
    end_date: str,
    *,
    offline: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    cache_base = cache_dir or (ROOT / "data" / "raw" / "cache")
    cache_base.mkdir(parents=True, exist_ok=True)
    frames = []

    for b in basins:
        b_id = b["basin_id"]
        key = cache.cache_key(
            "archive",
            basin_id=b_id,
            start=start_date,
            end=end_date,
            lat=round(float(b["lat"]), 4),
            lon=round(float(b["lon"]), 4),
        )
        cached = cache.read_frame(key, cache_dir=cache_base)
        if cached is not None and not cached.empty:
            frames.append(cached)
            continue

        if offline:
            demo_p = ROOT / "data" / "demo" / "features_demo.parquet"
            if demo_p.exists():
                df_demo = pd.read_parquet(demo_p)
                frames.append(df_demo[df_demo["basin_id"] == b_id])
                continue
            raise RuntimeError(f"Offline mode requested but no cache exists for {b_id}")

        res = openmeteo.fetch_archive(
            basin_id=b_id,
            latitude=float(b["lat"]),
            longitude=float(b["lon"]),
            start_date=start_date,
            end_date=end_date,
            elevation_m=b.get("elevation_mean_m"),
            terrain_susceptibility=b.get("terrain_susceptibility"),
            snow_fraction=b.get("snow_fraction"),
            use_cache=True,
            cache_dir=cache_base,
        )
        if res.frame.empty:
            raise RuntimeError(f"Open-Meteo returned empty frame for {b_id}")
        frames.append(res.frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], utc=True)
    return combined.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)


def build_labeled_features_b(
    raw_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    horizon_h: int = 6,
) -> pd.DataFrame:
    """Build features emphasizing snowpack melt, degree-days, and rain-on-snow."""
    df = add_features(raw_df)

    validate_no_leakage(df, feature_cols)

    # Snowmelt/rain-on-snow proxy trigger:
    # Substantial degree-day melt OR intense liquid rain on substantial snow cover
    is_melt_event = (df["degree_day_melt"].fillna(0) >= 0.8) | (df["rain_on_snow"].fillna(0) >= 6.0)

    grouped = df.groupby("basin_id", group_keys=False, sort=False)
    df["label"] = grouped.apply(
        lambda g: (
            is_melt_event.loc[g.index]
            .iloc[::-1]
            .rolling(horizon_h, min_periods=1)
            .max()
            .iloc[::-1]
            .shift(-horizon_h)
            .fillna(0)
            .astype(int)
        )
    )

    df.attrs["label_type"] = "runoff_proxy_snowmelt"
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast run with fewer epochs")
    parser.add_argument("--offline", action="store_true", help="use local cache")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    m_cfg = config["model_b"]
    feature_cols = m_cfg["features"]
    lookback = m_cfg["lookback_h"]
    horizon = m_cfg["label_horizon_h"]

    basins = load_basins()
    print("=== Model B (Snowmelt & Rain-on-Snow Risk) Real-Data Training ===")
    print(f"Monitored Basins: {len(basins)}")
    print(
        f"Architecture: SnowmeltLSTM (1 layer, {m_cfg['hidden_units']} units) + DegreeDayBaseline"
    )
    print(f"Features: {feature_cols}")

    train_start = "2020-01-01" if args.quick else "2016-01-01"
    train_end = "2021-12-31"

    raw_weather = fetch_or_load_real_weather_b(basins, train_start, train_end, offline=args.offline)
    feat_df = build_labeled_features_b(raw_weather, feature_cols, horizon_h=horizon)

    ts = feat_df["timestamp_utc"]
    val_cutoff = pd.to_datetime("2021-01-01T00:00:00Z")
    buffer_cutoff = pd.to_datetime("2020-12-01T00:00:00Z")

    feat_df["split"] = "train"
    feat_df.loc[(ts >= buffer_cutoff) & (ts < val_cutoff), "split"] = "buffer"
    feat_df.loc[ts >= val_cutoff, "split"] = "validation"

    validate_no_leakage(feat_df, feature_cols)

    train_mask = feat_df["split"] == "train"
    val_mask = feat_df["split"] == "validation"

    scaler = StandardScaler()
    scaler.fit(feat_df.loc[train_mask, feature_cols])
    scaled_feats = feat_df.copy()
    scaled_feats[feature_cols] = scaler.transform(feat_df[feature_cols]).fillna(0.0)

    basin_arrays = {}
    index_records = []
    for b_id, grp in scaled_feats.groupby("basin_id"):
        g = grp.sort_values("timestamp_utc").reset_index(drop=True)
        vals = g[feature_cols].values.astype(np.float32)
        lbls = g["label"].values.astype(np.float32)
        times, splits = g["timestamp_utc"].values, g["split"].values
        basin_arrays[b_id] = vals

        for step in range(lookback, len(g)):
            if splits[step] == "buffer":
                continue
            index_records.append(
                {
                    "basin_id": b_id,
                    "timestamp_utc": times[step],
                    "step_idx": step,
                    "split": splits[step],
                    "target": lbls[step],
                }
            )

    index_df = pd.DataFrame(index_records)
    train_idx = index_df[index_df["split"] == "train"].reset_index(drop=True)
    val_idx = index_df[index_df["split"] == "validation"].reset_index(drop=True)

    train_loader = DataLoader(
        TemporalDatasetB(basin_arrays, train_idx, lookback=lookback),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TemporalDatasetB(basin_arrays, val_idx, lookback=lookback),
        batch_size=args.batch_size,
        shuffle=False,
    )

    t_cfg = TrainConfig(
        seed=args.seed,
        hidden_units=m_cfg["hidden_units"],
        max_epochs=args.epochs,
    )
    model = build_model(n_features=len(feature_cols), cfg=t_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=m_cfg["learning_rate"])

    epochs = 3 if args.quick else args.epochs
    print(f"\nTraining Model B LSTM for {epochs} epochs...")
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(x_b)
            loss = nn.functional.binary_cross_entropy(pred, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
        avg_loss = total_loss / len(train_idx)
        if ep % max(1, epochs // 4) == 0 or ep == epochs:
            print(f"  Epoch {ep:02d}/{epochs:02d} - Train Loss: {avg_loss:.4f}")

    # Degree-Day Baseline fit
    baseline_fn, baseline_state = build_baseline(feat_df[train_mask])

    model.eval()
    val_preds = []
    with torch.no_grad():
        for x_b, _ in val_loader:
            val_preds.append(model(x_b).numpy())
    val_probs = np.concatenate(val_preds)

    y_val = val_idx["target"].values
    val_pred_binary = (val_probs >= 0.50).astype(int)
    val_pod = float(np.sum((val_pred_binary == 1) & (y_val == 1)) / max(1, np.sum(y_val == 1)))
    val_far = float(
        np.sum((val_pred_binary == 1) & (y_val == 0)) / max(1, np.sum(val_pred_binary == 1))
    )

    print(f"\nModel B Validation Metrics (2021): POD={val_pod:.4f} | FAR={val_far:.4f}")

    # Step 8: External Case Evaluation
    print("\n=======================================================")
    print("MODEL B HISTORICAL CASE EVALUATION (2010 & 2022)")
    print("=======================================================")
    case_reports = {}
    for case_year, (c_start, c_end) in [
        (2010, ("2010-07-28", "2010-08-03")),
        (2022, ("2022-08-25", "2022-08-30")),
    ]:
        try:
            case_weather = fetch_or_load_real_weather_b(
                basins, c_start, c_end, offline=args.offline
            )
            case_feats = add_features(case_weather)
            case_feats[feature_cols] = scaler.transform(case_feats[feature_cols]).fillna(0.0)

            case_basin_arrays = {
                b_id: grp[feature_cols].values.astype(np.float32)
                for b_id, grp in case_feats.groupby("basin_id")
            }
            case_records = []
            for b_id, grp in case_feats.groupby("basin_id"):
                g = grp.sort_values("timestamp_utc").reset_index(drop=True)
                times = g["timestamp_utc"].values
                for step in range(len(g)):
                    case_records.append(
                        {"basin_id": b_id, "timestamp_utc": times[step], "step_idx": step}
                    )

            c_idx_df = pd.DataFrame(case_records)
            preds_case = []
            for row in c_idx_df.itertuples():
                arr = case_basin_arrays[row.basin_id]
                s = row.step_idx
                if s < lookback:
                    pad = np.repeat(arr[:1], lookback - s, axis=0)
                    seq = np.vstack([pad, arr[:s]])
                else:
                    seq = arr[s - lookback : s]
                t_seq = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32)
                with torch.no_grad():
                    prob = float(model(t_seq).item())
                preds_case.append(prob)

            c_idx_df["prob"] = preds_case
            report = evaluate_case(
                case_year=case_year, predictions_df=c_idx_df, model_name="Model B", threshold=0.50
            )
            case_reports[case_year] = report.to_dict()
            print(f"\n--- Model B on {case_year} Case ({report.title}) ---")
            for comp in report.comparisons:
                print(
                    f"  {comp.basin_id:<16}: Confirmed={comp.confirmed_status:<18} PredRisk={str(comp.model_predicted_risk):<5} MaxProb={comp.model_max_prob:.3f} [{comp.comparison_outcome}]"
                )
        except Exception as exc:
            print(f"Warning: Could not evaluate Model B on {case_year}: {exc}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS_DIR / "model_b_real.pt")
    manifest = {
        "module": "B_snowmelt_lstm",
        "seed": args.seed,
        "features": feature_cols,
        "trained_at": datetime.now(UTC).isoformat(),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "validation_metrics": {"pod": val_pod, "far": val_far},
        "case_reports": case_reports,
    }
    (ARTIFACTS_DIR / "manifest_real.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nModel B artifacts saved to {ARTIFACTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
