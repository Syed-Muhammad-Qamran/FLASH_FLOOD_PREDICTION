"""Live operational input via Open-Meteo (no API key) and the transparent
rule engine used until trained checkpoints exist.

This is the part that makes the prototype genuinely operational: it runs on a
schedule, against live forecast data, with no credentials. The rule engine is
deliberately simple and auditable; the trained models are drop-in replacements.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import basins as all_basins
from .config import region, thresholds_config
from .features import melt_index
from .schema import tier_for, validate_predictions

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "floodwatch/0.1 (research prototype; shadow mode)"


def _sigmoid(x: float | np.ndarray):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def fetch_basin_forecast(
    basin: dict[str, Any],
    *,
    past_days: int = 2,
    forecast_days: int = 3,
    timeout: float = 25.0,
) -> pd.DataFrame:
    params = {
        "latitude": basin["lat"],
        "longitude": basin["lon"],
        "hourly": "precipitation,temperature_2m,relative_humidity_2m",
        "past_days": int(past_days),
        "forecast_days": int(forecast_days),
        "timezone": "UTC",
    }
    resp = requests.get(
        OPEN_METEO_URL, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(hourly["time"], utc=True),
            "rain_mm": pd.to_numeric(hourly["precipitation"], errors="coerce").fillna(0.0),
            "temp_c": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
            "rh": pd.to_numeric(hourly["relative_humidity_2m"], errors="coerce"),
        }
    )
    df = (
        df.set_index("timestamp_utc")
        .resample("3h")
        .agg({"rain_mm": "sum", "temp_c": "mean", "rh": "mean"})
        .reset_index()
    )
    df["basin_id"] = basin["basin_id"]
    df["snow_fraction"] = float(basin["snow_fraction"])
    df["elevation_mean_m"] = float(basin["elevation_mean_m"])
    return df


def fetch_all(
    basins: list[dict[str, Any]] | None = None,
    *,
    past_days: int = 2,
    forecast_days: int = 3,
) -> pd.DataFrame:
    basins = basins or all_basins()
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for b in basins:
        try:
            frames.append(fetch_basin_forecast(b, past_days=past_days, forecast_days=forecast_days))
        except Exception as exc:  # noqa: BLE001 - one basin must not kill the run
            errors[b["basin_id"]] = f"{type(exc).__name__}: {exc}"
    if not frames:
        raise RuntimeError(
            "Open-Meteo returned no data for any basin. Check network access. "
            f"First error: {next(iter(errors.values()), 'unknown')}"
        )

    table = pd.concat(frames, ignore_index=True)
    table = _add_derived(table)
    table.attrs["fetch_errors"] = errors
    return table


def _add_derived(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["rain_mm_3h"] = out["rain_mm"].clip(lower=0.0)
    out["soil_moisture"] = (
        0.15
        + 0.012
        * out.groupby("basin_id", group_keys=False)["rain_mm_3h"].transform(
            lambda s: s.rolling(8, min_periods=1).sum()
        )
    ).clip(0.05, 0.95)
    out["snowmelt_mm"] = melt_index(out)
    out["discharge_m3s"] = np.nan  # not available from Open-Meteo
    # Open-Meteo rows already include the forecast horizon, so the forecast
    # value for a timestamp is simply that row's value.
    out["rain_fcst_mm_3h"] = out["rain_mm_3h"]
    out["temp_fcst_c"] = out["temp_c"]
    return out.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)


def rule_engine(
    table: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transparent cloudburst + melt scoring.

    Model A: rainfall relative to the operational 3h threshold, wetness-adjusted.
    Model B: degree-day melt relative to a reference melt rate.

    These are *rules*, not learned models. They exist so the operational path is
    live before training finishes, and so the app always has a working fallback.
    """
    cfg = cfg or thresholds_config()
    a_cfg = cfg["model_a"]
    b_cfg = cfg["model_b"]

    rain_thr = float(a_cfg.get("rain_3h_floor_mm", 20.0))
    soil_ref = float(a_cfg.get("soil_moisture_percentile", 0.70))
    melt_ref = 1.5  # mm per 3h, a moderate basin-mean melt rate

    df = table.copy()
    rain = df["rain_mm_3h"].to_numpy(dtype=float)
    soil = df["soil_moisture"].to_numpy(dtype=float)
    melt = df["snowmelt_mm"].to_numpy(dtype=float)
    temp = df["temp_c"].to_numpy(dtype=float)

    prob_a = _sigmoid(3.0 * (rain / max(rain_thr, 1e-6) - 1.0) + 2.0 * (soil - soil_ref))
    prob_b = _sigmoid(2.5 * (melt / melt_ref - 1.0))

    drivers_a = [
        f"rain_3h={r:.1f}mm ({r / max(rain_thr, 1e-6):.0%} of threshold)"
        for r in rain
    ]
    drivers_b = [f"melt={m:.2f}mm/3h at {t:.1f}C" for m, t in zip(melt, temp)]

    cut_a = {k: float(v) for k, v in a_cfg.get("tiers", {}).items()}
    cut_b = {k: float(v) for k, v in b_cfg.get("tiers", {}).items()}

    pred_a = pd.DataFrame(
        {
            "basin_id": df["basin_id"].to_numpy(),
            "timestamp_utc": pd.to_datetime(df["timestamp_utc"], utc=True),
            "module": "A",
            "prob": np.clip(prob_a, 0.0, 1.0),
            "tier": [tier_for(float(p), cut_a) for p in prob_a],
            "top_drivers": _json_lists(
                [
                    [drivers_a[i], f"soil_moisture={soil[i]:.2f}"]
                    for i in range(len(df))
                ]
            ),
        }
    )
    pred_b = pd.DataFrame(
        {
            "basin_id": df["basin_id"].to_numpy(),
            "timestamp_utc": pd.to_datetime(df["timestamp_utc"], utc=True),
            "module": "B",
            "prob": np.clip(prob_b, 0.0, 1.0),
            "tier": [tier_for(float(p), cut_b) for p in prob_b],
            "top_drivers": _json_lists([[drivers_b[i]] for i in range(len(df))]),
        }
    )
    return validate_predictions(pred_a), validate_predictions(pred_b)


def _json_lists(items: list[list[str]]) -> list[str]:
    import json

    return [json.dumps(x, ensure_ascii=False) for x in items]


def describe_source() -> dict[str, Any]:
    return {
        "provider": "Open-Meteo",
        "endpoint": OPEN_METEO_URL,
        "auth": "none",
        "region": region()["name"],
        "note": "forecast + 2 days of past hourly precipitation",
    }
