"""Synthetic, labelled demo dataset.

The repository must run end to end before any credential exists. This module
generates a physically plausible hourly table for the eight monitored basins
and a deterministic placeholder for Model A (until the person-1 lane publishes
M2a). It is a pipeline test, never a scientific result.

Every artefact written here is labelled synthetic in its manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as feat
from . import schema
from .models import windows

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "demo" / "features_demo.parquet"
GRID_PATH = ROOT / "data" / "processed" / "grid_index.parquet"
SNAPSHOT_PATH = ROOT / "data" / "processed" / "SNAPSHOT.json"
HEALTH_PATH = ROOT / "data" / "processed" / "source_health.json"
PRED_A_PATH = ROOT / "artifacts" / "a" / "predictions_a.parquet"
MANIFEST_A_PATH = ROOT / "artifacts" / "a" / "manifest.json"

START = "2021-09-01T00:00:00Z"
END = "2023-02-01T00:00:00Z"

# Fixed events so the August 2022 replay window always has something to show.
KNOWN_EVENTS: tuple[tuple[str, str, float], ...] = (
    ("swat_kalam", "2022-08-26T18:00:00Z", 70.0),
    ("swat_madyan", "2022-08-26T19:00:00Z", 55.0),
    ("swat_chakdara", "2022-08-27T00:00:00Z", 45.0),
    ("panjkora_dir", "2022-08-26T21:00:00Z", 62.0),
    ("panjkora_talash", "2022-08-27T02:00:00Z", 40.0),
    ("chitral_chitral", "2022-07-15T12:00:00Z", 30.0),
    ("chitral_mastuj", "2022-07-20T15:00:00Z", 18.0),
    ("chitral_drosh", "2022-07-15T14:00:00Z", 26.0),
)


def _add_burst(burst: np.ndarray, index: int, amplitude: float, width: int) -> None:
    n = len(burst)
    for k in range(width):
        if 0 <= index + k < n:
            burst[index + k] += amplitude * np.exp(-k / 1.6)


def generate_raw(start: str = START, end: str = END, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    n = len(t)
    doy = t.dayofyear.to_numpy().astype(float)
    hour = t.hour.to_numpy().astype(float)

    frames: list[pd.DataFrame] = []
    for basin in feat.load_basins():
        basin_id = str(basin["basin_id"])
        elev = float(basin["elevation_mean_m"])
        snowf = float(basin["snow_fraction"])
        terrain = float(basin["terrain_susceptibility"])

        monsoon = 0.10 + 1.65 * np.exp(-0.5 * ((doy - 212) / 34) ** 2)
        winter = 0.22 * np.exp(-0.5 * ((doy - 70) / 28) ** 2)
        base = rng.gamma(shape=0.55, scale=0.70, size=n) * (monsoon + winter)

        burst = np.zeros(n)
        n_bursts = int(rng.poisson(10 * (n / 8000)))
        monsoon_idx = np.flatnonzero((doy >= 170) & (doy <= 275))
        for _ in range(n_bursts):
            i = (
                int(rng.choice(monsoon_idx))
                if len(monsoon_idx) and rng.random() < 0.85
                else int(rng.integers(0, n))
            )
            _add_burst(burst, i, float(rng.uniform(18, 55)), int(rng.integers(2, 6)))
        rain = np.clip(base + burst, 0.0, None)

        seasonal = 23.0 - 12.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        lapse = 6.5 * elev / 1000.0
        diurnal = 4.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
        temp = seasonal - lapse + diurnal + rng.normal(0.0, 1.5, n)

        snow_avail = np.clip(np.sin(2 * np.pi * (doy - 60) / 365.25), 0.0, 1.0)
        snow_depth = snow_avail * snowf * 140.0
        snowmelt = np.clip(temp, 0.0, None) * 0.10 * snowf * (snow_depth > 5.0)
        freezing_level = np.clip(4200.0 + 120.0 * temp, 2500.0, 6500.0)

        soil = np.empty(n)
        s = 0.30
        for i in range(n):
            s = 0.965 * s + 0.010 + 0.011 * rain[i]
            soil[i] = min(s, 0.95)

        discharge = np.empty(n)
        q = 40.0 + 380.0 * snowf
        for i in range(n):
            q = 0.82 * q + 1.5 * rain[i] + 22.0 * snowmelt[i] + 3.0
            discharge[i] = q
        discharge = np.clip(discharge * (1.0 + rng.normal(0.0, 0.03, n)), 0.0, None)

        # Imperfect forecast of the same valid time (known-nowcast envelope).
        rain_fcst = np.clip(rain * rng.lognormal(0.0, 0.30, n), 0.0, None)
        temp_fcst = temp + rng.normal(0.0, 0.8, n)

        frames.append(
            pd.DataFrame(
                {
                    "basin_id": basin_id,
                    "timestamp_utc": t,
                    "rain_mm": rain,
                    "rain_fcst_mm_3h": rain_fcst,
                    "temp_c": temp,
                    "temp_fcst_c": temp_fcst,
                    "freezing_level_m": freezing_level,
                    "soil_moisture": soil,
                    "snow_depth_mm": snow_depth,
                    "snowmelt_mm": snowmelt,
                    "discharge_m3s": discharge,
                    "snow_fraction": snowf,
                    "elevation_mean_m": elev,
                    "terrain_susceptibility": terrain,
                }
            )
        )

    table = pd.concat(frames, ignore_index=True)
    table = _inject_known_events(table)
    return table.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)


def _inject_known_events(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    for basin_id, when, amplitude in KNOWN_EVENTS:
        target = pd.Timestamp(when)
        mask = (out["basin_id"] == basin_id) & (out["timestamp_utc"] == target)
        idx = out.index[mask]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        for k in range(8):
            j = i + k
            if j >= len(out) or out.at[j, "basin_id"] != basin_id:
                break
            out.at[j, "rain_mm"] = float(out.at[j, "rain_mm"]) + amplitude * np.exp(-k / 1.8)
        lo = max(0, i - 10)
        out.loc[lo:i, "soil_moisture"] = np.clip(
            out.loc[lo:i, "soil_moisture"].to_numpy() + 0.25, 0.0, 0.95
        )
        for k in range(0, 14):
            j = i + k
            if j >= len(out) or out.at[j, "basin_id"] != basin_id:
                break
            out.at[j, "discharge_m3s"] = float(out.at[j, "discharge_m3s"]) * (
                1.0 + 1.6 * np.exp(-k / 4.0)
            )
    return out


def _snapshot_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _placeholder_a(features: pd.DataFrame, grid: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """Deterministic stand-in for Model A until M2a lands.

    Clearly labelled synthetic: a smooth cloudburst score, no learned weights.
    It exists so fusion, thresholds, splits and the replay suite can be built
    and reviewed before the person-1 lane publishes its model.
    """
    frame = features.merge(
        grid[grid["split"] != "test"], on=["basin_id", "timestamp_utc"], how="inner"
    )
    terrain = frame["terrain_susceptibility"].astype(float)
    logit = (
        0.10 * (frame["rain_3h"].astype(float) - 8.0)
        + 0.018 * (frame["rain_24h"].astype(float) - 25.0)
        + 2.4 * (frame["soil_moisture"].astype(float) - 0.45)
        + 0.8 * (terrain - 0.5)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    baseline = 1.0 / (
        1.0 + np.exp(-(0.06 * (frame["rain_3h"].astype(float) - 12.0) + 0.4 * (terrain - 0.5)))
    )
    drivers = []
    for row in frame.itertuples(index=False):
        items = [
            {"feature": "rain_3h", "weight": round(float(row.rain_3h), 3)},
            {"feature": "soil_moisture", "weight": round(float(row.soil_moisture), 3)},
            {
                "feature": "terrain_susceptibility",
                "weight": round(float(row.terrain_susceptibility), 3),
            },
        ]
        drivers.append(schema.drivers_to_json(items))

    out = pd.DataFrame(
        {
            "basin_id": frame["basin_id"].astype(str),
            "timestamp_utc": pd.to_datetime(frame["timestamp_utc"], utc=True),
            "module": schema.MODULE_A,
            "split": frame["split"].astype(str),
            "prob": np.round(prob, 6),
            "baseline_prob": np.round(baseline, 6),
            "source": "baseline",
            "top_drivers": drivers,
            "model_version": f"A_cloudburst_cnn_lstm-synthetic-{snapshot[:8]}",
        }
    )
    out = out.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)
    return out


def build_demo(*, force: bool = False) -> dict:
    """Generate features, grid, snapshot, source health and the placeholder A output."""
    if FEATURES_PATH.exists() and not force:
        return {"reused": True, "features": str(FEATURES_PATH)}

    raw = generate_raw()
    cfg = windows.load_split_config()
    train = windows.assign_split(raw["timestamp_utc"], cfg) == "train"
    table = feat.feature_table(raw, train_mask=train)
    feat.write_features(table, FEATURES_PATH)

    grid = windows.build_grid_index(table, cfg=cfg)
    windows.write_grid_index(grid, GRID_PATH)

    snapshot_hash = _snapshot_hash(FEATURES_PATH)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "schema_version": 1,
        "snapshot_id": f"synthetic-{snapshot_hash[:8]}",
        "data_snapshot_hash": snapshot_hash,
        "synthetic": True,
        "start_utc": str(table["timestamp_utc"].min()),
        "end_utc": str(table["timestamp_utc"].max()),
        "basins": sorted(table["basin_id"].unique().tolist()),
        "rows": int(len(table)),
        "split_rows": grid.groupby("split").size().astype(int).to_dict(),
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    # Deterministic placeholder Model A output (person-1 lane placeholder).
    pred_a = _placeholder_a(table, grid, snapshot_hash)
    model_version = str(pred_a["model_version"].iloc[0])
    schema.publish_model_file(
        pred_a,
        PRED_A_PATH,
        module=schema.MODULE_A,
        grid_index=grid,
        metadata={
            "schema_version": schema.SCHEMA_VERSION,
            "seed": 1234,
            "data_snapshot_hash": snapshot_hash,
            "label_horizon_h": schema.LABEL_HORIZON_H,
            "trained_at": "2026-11-08T00:00:00Z",
            "module": schema.MODULE_A,
        },
    )
    MANIFEST_A_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "module": schema.MODULE_A,
                "model_version": model_version,
                "data_snapshot_hash": snapshot_hash,
                "seed": 1234,
                "rows": int(len(pred_a)),
                "sha256": hashlib.sha256(PRED_A_PATH.read_bytes()).hexdigest(),
                "validated_at": "2026-11-08T00:00:00Z",
                "validation_metrics_path": "artifacts/a/metrics_a.json",
                "baseline_wins": True,
                "synthetic_placeholder": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    last = pd.to_datetime(table["timestamp_utc"].max(), utc=True)
    HEALTH_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "archive": {
                        "last_valid_utc": str(last),
                        "lag_hours": 0.0,
                        "expected_cadence_h": 24.0,
                        "stale": False,
                        "fallback_used": False,
                    },
                    "forecast": {
                        "last_valid_utc": str(last),
                        "lag_hours": 0.0,
                        "expected_cadence_h": 6.0,
                        "stale": False,
                        "fallback_used": False,
                    },
                    "flood": {
                        "last_valid_utc": None,
                        "lag_hours": None,
                        "expected_cadence_h": 24.0,
                        "stale": True,
                        "fallback_used": True,
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "reused": False,
        "features": str(FEATURES_PATH),
        "rows": int(len(table)),
        "grid_rows": int(len(grid)),
        "snapshot": snapshot,
    }
