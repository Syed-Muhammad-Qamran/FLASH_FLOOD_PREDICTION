"""Open-Meteo client, terrain layer, grid and cache behave offline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from floodwatch.data_sources import cache, grid, openmeteo, terrain


def _weather_payload() -> dict:
    return {
        "latitude": 35.48,
        "longitude": 72.58,
        "elevation": 2000,
        "hourly": {
            "time": ["2022-08-26T00:00:00Z", "2022-08-26T01:00:00Z"],
            "temperature_2m": [10.0, 11.0],
            "precipitation": [2.0, 3.0],
            "rain": [2.0, 3.0],
            "soil_moisture_0_to_7cm": [0.3, 0.35],
            "snow_depth": [0.05, 0.04],
            "snowfall": [0.0, 0.0],
        },
    }


def test_openmeteo_archive_normalises_units() -> None:
    result = openmeteo.fetch_archive(
        basin_id="swat_kalam",
        latitude=35.48,
        longitude=72.58,
        start_date="2022-08-26",
        end_date="2022-08-26",
        terrain_susceptibility=0.82,
        snow_fraction=0.35,
        fetcher=lambda url, params, timeout: _weather_payload(),
    )
    frame = result.frame
    assert len(frame) == 2
    assert list(frame["basin_id"]) == ["swat_kalam", "swat_kalam"]
    assert frame["rain_mm"].tolist() == [2.0, 3.0]
    assert frame["temp_c"].tolist() == [10.0, 11.0]
    # Open-Meteo snow_depth is metres -> converted to mm.
    assert frame["snow_depth_mm"].tolist() == [50.0, 40.0]
    assert frame["terrain_susceptibility"].iloc[0] == 0.82
    assert frame["snow_fraction"].iloc[0] == 0.35


def test_openmeteo_forecast_derives_3h_rain() -> None:
    result = openmeteo.fetch_forecast(
        basin_id="swat_kalam",
        latitude=35.48,
        longitude=72.58,
        fetcher=lambda url, params, timeout: _weather_payload(),
    )
    assert result.frame["rain_fcst_mm_3h"].tolist() == [5.0, 3.0]


def test_openmeteo_flood_fetch() -> None:
    payload = {
        "latitude": 35.48,
        "longitude": 72.58,
        "daily": {
            "time": ["2022-08-26", "2022-08-27"],
            "river_discharge": [120.0, 150.0],
        },
    }
    result = openmeteo.fetch_flood_discharge(
        basin_id="swat_kalam",
        latitude=35.48,
        longitude=72.58,
        fetcher=lambda url, params, timeout: payload,
    )
    assert result.frame["discharge_m3s"].tolist() == [120.0, 150.0]


def test_openmeteo_elevation_batch() -> None:
    payload = {"elevation": [2000.0, 1500.0]}
    values = openmeteo.fetch_elevations(
        [(35.48, 72.58), (34.65, 72.03)],
        fetcher=lambda url, params, timeout: payload,
    )
    assert values == [2000.0, 1500.0]


def test_openmeteo_error_is_reported() -> None:
    def boom(url, params, timeout):
        raise ValueError("bad payload")

    try:
        openmeteo.fetch_archive(
            basin_id="swat_kalam",
            latitude=35.48,
            longitude=72.58,
            start_date="2022-08-26",
            end_date="2022-08-26",
            fetcher=boom,
        )
    except openmeteo.OpenMeteoError:
        return
    raise AssertionError("network error was not converted")


def test_terrain_layer_records_provenance_and_null_derivatives() -> None:
    basins = [
        {
            "basin_id": "swat_kalam",
            "name": "Swat River at Kalam",
            "lat": 35.48,
            "lon": 72.58,
            "elevation_mean_m": 2100,
            "terrain_susceptibility": 0.82,
            "snow_fraction": 0.35,
            "primary_mechanism": "cloudburst",
        }
    ]
    payload = terrain.build_static_layer(basins, elevations_m={"swat_kalam": 2050.0}, source="test")
    row = payload["basins"]["swat_kalam"]
    assert row["elevation_m"] == 2050.0
    assert row["elevation_source"] == "open-meteo-elevation"
    assert row["mean_slope_deg"] is None
    assert payload["provenance"]["sha256"]
    assert "NASA Earthdata" in " ".join(payload["dem_catalog"])


def test_terrain_round_trip(tmp_path: Path) -> None:
    basins = [
        {
            "basin_id": "swat_kalam",
            "name": "x",
            "lat": 1.0,
            "lon": 2.0,
            "elevation_mean_m": 100,
            "terrain_susceptibility": 0.5,
            "snow_fraction": 0.1,
        }
    ]
    payload = terrain.build_static_layer(basins)
    path = terrain.write_terrain(payload, tmp_path / "terrain.json")
    loaded = terrain.load_terrain(path)
    assert terrain.terrain_for_basin("swat_kalam", loaded)["terrain_susceptibility"] == 0.5
    assert terrain.terrain_for_basin("missing", loaded) == {}


def test_grid_patch_has_nine_points_and_a_center() -> None:
    points = grid.patch_3x3(35.48, 72.58, step_deg=0.1)
    assert len(points) == 9
    centers = [point for point in points if point.is_center]
    assert len(centers) == 1
    assert centers[0].lat == 35.48
    assert centers[0].lon == 72.58


def test_cache_round_trip(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    entry = cache.write_frame(frame, key="demo", meta={"source": "test"}, cache_dir=tmp_path)
    assert entry.path.exists()
    loaded = cache.read_frame("demo", cache_dir=tmp_path)
    assert loaded is not None and loaded["a"].tolist() == [1, 2]
    assert cache.read_meta("demo", cache_dir=tmp_path)["source"] == "test"
    assert cache.read_frame("missing", cache_dir=tmp_path) is None
