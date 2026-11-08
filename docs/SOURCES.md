# Data sources

This project uses only sources that can be cited, cached, and run in degraded mode
when unavailable. The committed demo is **synthetic**; live ingest must be explicit.

## Active integrations

### Open-Meteo (no API key)
- **Archive API** (`archive-api.open-meteo.com/v1/archive`): ERA5-based historical
  hourly weather — used for backfills and training-feature construction.
- **Forecast API** (`api.open-meteo.com/v1/forecast`): recent + forecast hourly
  weather — used for operational refreshes.
- **Flood API** (`flood-api.open-meteo.com/v1/flood`): GloFAS-based river discharge
  signals — optional/degraded when absent.
- **Elevation API** (`api.open-meteo.com/v1/elevation`): point elevations for the
  static terrain layer.

Implementation: `src/floodwatch/data_sources/openmeteo.py`,
`scripts/01_fetch_openmeteo.py`. All requests are explicit (`--fetch`), cached
optionally under `data/raw/cache/` (gitignored), and surfaced via
`data/processed/source_health.json` (`floodwatch.data_sources.health`).

## Static terrain / DEM layer
- **Committed layer:** `data/static/basin_terrain.json` (point control points,
  configured susceptibility/snow fraction, optional Open-Meteo point elevation,
  provenance hash). Builder: `scripts/09_build_terrain.py`.
- **Raster DEM catalog (for future slope/aspect/watershed derivatives):**
  NASA Earthdata ASTER GDEM / NASADEM (30 m), USGS EarthExplorer SRTM,
  OpenTopography lidar/topography, Copernicus DEM. Raster processing is not bundled;
  derived fields are `null` until a DEM pipeline produces versioned outputs.

## Quality / safety rules
- Never fabricate missing values; leave null and mark the source degraded.
- Never commit raw downloads or credentials. Open-Meteo needs no key.
- Record fetch time, source, and lag in `source_health.json`.
- The test split remains sealed: ingest and feature scripts must not select
  `split == "test"` (see `scripts/07_test_eval_once.py` and
  `tests/test_no_test_leak.py`).
- Operational messages remain shadow-mode, verified before any delivery attempt.
