"""Open-Meteo client for archive, forecast and elevation data.

No API key required. All network access is explicit (callers choose when to
fetch). Responses are normalised toward ``floodwatch.features.RAW_COLUMNS`` where
fields are available; missing hydrology/snowpack fields are left null rather than
invented.

Operational use: cache responses, record ``source_health``, and never treat a
failed fetch as a crash.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

API_BASE = "https://api.open-meteo.com/v1"
ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_BASE = "https://flood-api.open-meteo.com/v1/flood"
ELEVATION_URL = f"{API_BASE}/elevation"
USER_AGENT = "FloodWatchKP/0.1 (+offline-safe; contact local operator)"

HOURLY_WEATHER = (
    "temperature_2m",
    "precipitation",
    "rain",
    "soil_moisture_0_to_7cm",
    "snowfall",
    "snow_depth",
)
HOURLY_FORECAST = HOURLY_WEATHER + ("freezing_level_height",)


class OpenMeteoError(RuntimeError):
    """Network or payload error. Never contains secrets (none are used)."""


@dataclass(frozen=True)
class FetchResult:
    source: str
    frame: pd.DataFrame
    fetched_at: datetime
    latitude: float
    longitude: float
    elevation: float | None = None
    raw: dict[str, Any] | None = None


Fetcher = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _default_fetch(url: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    query = urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise OpenMeteoError(f"Open-Meteo request failed: {type(exc).__name__}") from exc
    try:
        data = json.loads(payload or "{}")
    except ValueError as exc:
        raise OpenMeteoError("Open-Meteo returned invalid JSON") from exc
    if isinstance(data, dict) and data.get("error"):
        reason = data.get("reason", "unknown error")
        raise OpenMeteoError(f"Open-Meteo error: {reason}")
    if not isinstance(data, dict):
        raise OpenMeteoError("Open-Meteo returned an unexpected payload")
    return data


def _hourly_frame(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()
    frame = pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True)})
    for key, values in hourly.items():
        if key == "time":
            continue
        frame[key] = values
    return frame


def _normalise_weather(
    frame: pd.DataFrame,
    *,
    basin_id: str,
    elevation_m: float | None,
    terrain_susceptibility: float | None,
    snow_fraction: float | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["basin_id"] = str(basin_id)
    # Open-Meteo hourly `precipitation`/`rain` are mm for the preceding hour.
    rain = out.get("rain")
    if rain is None:
        rain = out.get("precipitation")
    out["rain_mm"] = pd.to_numeric(rain, errors="coerce")
    # Forecast precipitation is not present in archive responses; leave null.
    out["rain_fcst_mm_3h"] = pd.to_numeric(out.get("rain_fcst_mm_3h"), errors="coerce")
    out["temp_c"] = pd.to_numeric(out.get("temperature_2m"), errors="coerce")
    out["temp_fcst_c"] = pd.to_numeric(out.get("temp_fcst_c"), errors="coerce")
    out["freezing_level_m"] = pd.to_numeric(out.get("freezing_level_height"), errors="coerce")
    soil = out.get("soil_moisture_0_to_7cm")
    out["soil_moisture"] = pd.to_numeric(soil, errors="coerce")
    # Open-Meteo snow_depth is metres in recent APIs; convert defensively.
    snow_depth = pd.to_numeric(out.get("snow_depth"), errors="coerce")
    if snow_depth.notna().any() and float(snow_depth.dropna().median()) < 50:
        snow_depth = snow_depth * 1000.0
    out["snow_depth_mm"] = snow_depth
    # Snowfall is cm in Open-Meteo; approximate melt proxy as 0 when unknown.
    snowfall_cm = pd.to_numeric(out.get("snowfall"), errors="coerce")
    out["snowmelt_mm"] = pd.to_numeric(out.get("snowmelt_mm"), errors="coerce")
    if out["snowmelt_mm"].isna().all() and snowfall_cm.notna().any():
        out["snowmelt_mm"] = 0.0
    out["discharge_m3s"] = pd.to_numeric(out.get("discharge_m3s"), errors="coerce")
    out["snow_fraction"] = float(snow_fraction) if snow_fraction is not None else math.nan
    out["elevation_mean_m"] = float(elevation_m) if elevation_m is not None else math.nan
    if terrain_susceptibility is not None:
        out["terrain_susceptibility"] = float(terrain_susceptibility)
    return out


def fetch_archive(
    *,
    basin_id: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    elevation_m: float | None = None,
    terrain_susceptibility: float | None = None,
    snow_fraction: float | None = None,
    hourly: Iterable[str] = HOURLY_WEATHER,
    timezone: str = "UTC",
    timeout_s: float = 30.0,
    fetcher: Fetcher | None = None,
) -> FetchResult:
    """Fetch historical hourly weather for one point (ERA5-backed archive API)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly),
        "timezone": timezone,
    }
    try:
        payload = (fetcher or _default_fetch)(ARCHIVE_BASE, params, timeout_s)
    except (OpenMeteoError, ValueError, TypeError, OSError) as exc:
        raise OpenMeteoError(f"Open-Meteo request failed: {type(exc).__name__}") from exc
    frame = _normalise_weather(
        _hourly_frame(payload),
        basin_id=basin_id,
        elevation_m=elevation_m if elevation_m is not None else payload.get("elevation"),
        terrain_susceptibility=terrain_susceptibility,
        snow_fraction=snow_fraction,
    )
    return FetchResult(
        source="archive",
        frame=frame,
        fetched_at=datetime.now(UTC),
        latitude=float(payload.get("latitude", latitude)),
        longitude=float(payload.get("longitude", longitude)),
        elevation=payload.get("elevation"),
        raw=payload,
    )


def fetch_forecast(
    *,
    basin_id: str,
    latitude: float,
    longitude: float,
    past_days: int = 2,
    forecast_days: int = 3,
    elevation_m: float | None = None,
    terrain_susceptibility: float | None = None,
    snow_fraction: float | None = None,
    hourly: Iterable[str] = HOURLY_FORECAST,
    timezone: str = "UTC",
    timeout_s: float = 30.0,
    fetcher: Fetcher | None = None,
) -> FetchResult:
    """Fetch recent + forecast hourly weather for one point."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly),
        "past_days": int(past_days),
        "forecast_days": int(forecast_days),
        "timezone": timezone,
    }
    try:
        payload = (fetcher or _default_fetch)(API_BASE + "/forecast", params, timeout_s)
    except (OpenMeteoError, ValueError, TypeError, OSError) as exc:
        raise OpenMeteoError(f"Open-Meteo request failed: {type(exc).__name__}") from exc
    frame = _hourly_frame(payload)
    # Derive a simple 3h forward rain sum when precipitation is present.
    if "precipitation" in frame.columns:
        precip = pd.to_numeric(frame["precipitation"], errors="coerce")
        frame["rain_fcst_mm_3h"] = (
            precip.iloc[::-1].rolling(3, min_periods=1).sum().iloc[::-1].to_numpy()
        )
    frame = _normalise_weather(
        frame,
        basin_id=basin_id,
        elevation_m=elevation_m if elevation_m is not None else payload.get("elevation"),
        terrain_susceptibility=terrain_susceptibility,
        snow_fraction=snow_fraction,
    )
    return FetchResult(
        source="forecast",
        frame=frame,
        fetched_at=datetime.now(UTC),
        latitude=float(payload.get("latitude", latitude)),
        longitude=float(payload.get("longitude", longitude)),
        elevation=payload.get("elevation"),
        raw=payload,
    )


def fetch_flood_discharge(
    *,
    basin_id: str,
    latitude: float,
    longitude: float,
    past_days: int = 7,
    forecast_days: int = 3,
    timeout_s: float = 30.0,
    fetcher: Fetcher | None = None,
) -> FetchResult:
    """Fetch river discharge from the Open-Meteo Flood API (GloFAS-based)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "river_discharge,river_discharge_mean",
        "past_days": int(past_days),
        "forecast_days": int(forecast_days),
        "timezone": "UTC",
    }
    try:
        payload = (fetcher or _default_fetch)(FLOOD_BASE, params, timeout_s)
    except (OpenMeteoError, ValueError, TypeError, OSError) as exc:
        raise OpenMeteoError(f"Open-Meteo request failed: {type(exc).__name__}") from exc
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    frame = (
        pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True)})
        if times
        else pd.DataFrame()
    )
    if not frame.empty:
        discharge = daily.get("river_discharge")
        if discharge is None:
            discharge = daily.get("river_discharge_mean")
        frame["basin_id"] = str(basin_id)
        frame["discharge_m3s"] = pd.to_numeric(discharge, errors="coerce")
    return FetchResult(
        source="flood",
        frame=frame,
        fetched_at=datetime.now(UTC),
        latitude=float(payload.get("latitude", latitude)),
        longitude=float(payload.get("longitude", longitude)),
        elevation=payload.get("elevation"),
        raw=payload,
    )


def fetch_elevations(
    points: Iterable[tuple[float, float]],
    *,
    timeout_s: float = 30.0,
    fetcher: Fetcher | None = None,
) -> list[float | None]:
    """Batch point elevations (metres) from the Open-Meteo elevation API."""
    points = list(points)
    if not points:
        return []
    params = {
        "latitude": ",".join(f"{lat:.6f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.6f}" for _, lon in points),
    }
    try:
        payload = (fetcher or _default_fetch)(ELEVATION_URL, params, timeout_s)
    except (OpenMeteoError, ValueError, TypeError, OSError) as exc:
        raise OpenMeteoError(f"Open-Meteo request failed: {type(exc).__name__}") from exc
    values = payload.get("elevation") or []
    out: list[float | None] = []
    for i in range(len(points)):
        if i < len(values) and values[i] is not None:
            try:
                out.append(float(values[i]))
            except (TypeError, ValueError):
                out.append(None)
        else:
            out.append(None)
    return out


def fetch_basin_catalog(
    basins: Iterable[dict[str, Any]],
    *,
    mode: str = "forecast",
    start_date: str | None = None,
    end_date: str | None = None,
    sleep_s: float = 0.0,
    fetcher: Fetcher | None = None,
) -> list[FetchResult]:
    """Fetch one result per basin, keeping requests sequential and polite."""
    results: list[FetchResult] = []
    for basin in basins:
        common = {
            "basin_id": str(basin["basin_id"]),
            "latitude": float(basin["lat"]),
            "longitude": float(basin["lon"]),
            "elevation_m": basin.get("elevation_mean_m"),
            "terrain_susceptibility": basin.get("terrain_susceptibility"),
            "snow_fraction": basin.get("snow_fraction"),
            "fetcher": fetcher,
        }
        if mode == "archive":
            if not start_date or not end_date:
                raise ValueError("archive mode requires start_date and end_date")
            results.append(fetch_archive(start_date=start_date, end_date=end_date, **common))
        elif mode == "forecast":
            results.append(fetch_forecast(**common))
        else:
            raise ValueError(f"unsupported mode: {mode}")
        if sleep_s:
            time.sleep(sleep_s)
    return results
