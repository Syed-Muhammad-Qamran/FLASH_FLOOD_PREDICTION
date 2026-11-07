"""Synthetic demo dataset.

Purpose: prove the whole pipeline (labels -> features -> baseline -> eval ->
fusion -> app -> runner) works end to end before a single credential exists.

This data is NOT real. Any metric computed on it is a pipeline test, not a
result. It is deliberately generated with the same two mechanisms so that the
plumbing is exercised: monsoon cloudbursts and degree-day melt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR, basins

# Fixed events so the replay demo always has something to show.
# Aligned with the real late-August 2022 Pakistan flood window.
KNOWN_EVENTS: list[tuple[str, str, float]] = [
    ("swat_chakdara", "2022-08-26T18:00Z", 88.0),
    ("panjkora_dir", "2022-08-26T21:00Z", 74.0),
    ("kabul_peshawar", "2022-08-27T00:00Z", 46.0),
    ("kohat_toi", "2022-08-19T15:00Z", 62.0),
    ("chitral_chitral", "2022-07-05T12:00Z", 28.0),
    ("gomal_dik", "2022-08-11T09:00Z", 55.0),
]

COLUMNS = [
    "basin_id",
    "timestamp_utc",
    "rain_mm_3h",
    "temp_c",
    "soil_moisture",
    "snowmelt_mm",
    "discharge_m3s",
    "rain_fcst_mm_3h",
    "temp_fcst_c",
    "snow_fraction",
    "elevation_mean_m",
]


def _add_burst(burst: np.ndarray, index: int, amplitude: float, width: int) -> None:
    n = len(burst)
    for k in range(width):
        if 0 <= index + k < n:
            burst[index + k] += amplitude * np.exp(-k / 1.6)


def generate(
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    freq: str = "3h",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    n = len(t)
    doy = t.dayofyear.to_numpy().astype(float)
    hour = t.hour.to_numpy().astype(float)

    frames: list[pd.DataFrame] = []
    for b in basins():
        basin_id = b["basin_id"]
        elev = float(b["elevation_mean_m"])
        snowf = float(b["snow_fraction"])

        # --- rainfall: monsoon peak in late July, weak winter westerlies ---
        monsoon = 0.12 + 1.8 * np.exp(-0.5 * ((doy - 210) / 38) ** 2)
        winter = 0.25 * np.exp(-0.5 * ((doy - 70) / 30) ** 2)
        wetness = monsoon + winter
        base = rng.gamma(shape=0.55, scale=0.85, size=n) * wetness

        burst = np.zeros(n)
        n_bursts = int(rng.poisson(8 * (n / 2920)))
        monsoon_idx = np.flatnonzero((doy >= 170) & (doy <= 275))
        for _ in range(n_bursts):
            if len(monsoon_idx) and rng.random() < 0.85:
                i = int(rng.choice(monsoon_idx))
            else:
                i = int(rng.integers(0, n))
            _add_burst(burst, i, float(rng.uniform(22, 70)), int(rng.integers(2, 7)))

        rain = np.clip(base + burst, 0.0, None)

        # --- temperature: seasonal + lapse + diurnal ---
        seasonal = 24.0 - 13.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        lapse = 6.5 * elev / 1000.0
        diurnal = 5.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
        temp = seasonal - lapse + diurnal + rng.normal(0.0, 1.8, n)

        # --- snowmelt: degree-day, gated by seasonal snow availability ---
        snow_avail = np.clip(np.sin(2 * np.pi * (doy - 55) / 365.25), 0.0, 1.0)
        melt = np.clip(temp, 0.0, None) * 2.5 / 8.0 * snowf * snow_avail

        # --- soil moisture: slow bucket, saturates after heavy rain ---
        soil = np.empty(n)
        s = 0.30
        for i in range(n):
            s = 0.965 * s + 0.012 + 0.010 * rain[i]
            s = min(s, 0.95)
            soil[i] = s

        # --- discharge: fast rainfall response + melt baseflow ---
        discharge = np.empty(n)
        q = 40.0 + 400.0 * snowf
        for i in range(n):
            q = 0.82 * q + 1.4 * rain[i] + 25.0 * melt[i] + 3.0
            discharge[i] = q
        discharge *= 1.0 + rng.normal(0.0, 0.03, n)
        discharge = np.clip(discharge, 0.0, None)

        # Imperfect forecast for the same valid time: this is what a model would
        # have seen when issuing a warning. Multiplicative error on rain,
        # additive on temperature.
        rain_fcst = np.clip(rain * rng.lognormal(0.0, 0.35, n), 0.0, None)
        temp_fcst = temp + rng.normal(0.0, 1.0, n)

        frames.append(
            pd.DataFrame(
                {
                    "basin_id": basin_id,
                    "timestamp_utc": t,
                    "rain_mm_3h": rain,
                    "temp_c": temp,
                    "soil_moisture": soil,
                    "snowmelt_mm": melt,
                    "discharge_m3s": discharge,
                    "rain_fcst_mm_3h": rain_fcst,
                    "temp_fcst_c": temp_fcst,
                    "snow_fraction": snowf,
                    "elevation_mean_m": elev,
                }
            )
        )

    table = pd.concat(frames, ignore_index=True)
    table = _inject_known_events(table, rng)
    return table[COLUMNS].sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)


def _inject_known_events(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Guarantee the replay window contains a few strong cloudbursts."""
    out = table.copy()
    for basin_id, when, amplitude in KNOWN_EVENTS:
        target = pd.Timestamp(when)
        sel = (out["basin_id"] == basin_id) & (out["timestamp_utc"] == target)
        idx = out.index[sel]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        width = 4
        for k in range(width):
            j = i + k
            if j >= len(out):
                break
            if out.at[j, "basin_id"] != basin_id:
                break
            out.at[j, "rain_mm_3h"] = float(out.at[j, "rain_mm_3h"]) + amplitude * np.exp(-k / 1.6)
        # Push antecedent soil moisture up so the compound condition triggers.
        lo = max(0, i - 8)
        out.loc[lo:i, "soil_moisture"] = np.clip(
            out.loc[lo:i, "soil_moisture"].to_numpy() + 0.25, 0.0, 0.95
        )
        # Let discharge respond.
        for k in range(0, 12):
            j = i + k
            if j >= len(out):
                break
            if out.at[j, "basin_id"] != basin_id:
                break
            out.at[j, "discharge_m3s"] = float(out.at[j, "discharge_m3s"]) * (1.0 + 1.5 * np.exp(-k / 4.0))
    return out


def demo_path() -> Path:
    return DATA_DIR / "demo" / "features_demo.parquet"


def load_or_generate(force: bool = False) -> pd.DataFrame:
    path = demo_path()
    if path.exists() and not force:
        return pd.read_parquet(path)
    table = generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def window(table: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    t = pd.to_datetime(table["timestamp_utc"], utc=True)
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC")
    return table.loc[(t >= lo) & (t < hi)].copy()


def describe(table: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(table)),
        "basins": int(table["basin_id"].nunique()),
        "start_utc": str(table["timestamp_utc"].min()),
        "end_utc": str(table["timestamp_utc"].max()),
        "mean_rain_mm_3h": float(table["rain_mm_3h"].mean()),
        "max_rain_mm_3h": float(table["rain_mm_3h"].max()),
        "synthetic": True,
    }
