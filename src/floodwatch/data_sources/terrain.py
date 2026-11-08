"""Static DEM/terrain layer loading and provenance.

The committed layer is a point-based terrain summary for each monitored control
point (elevation and configured susceptibility). True raster DEM derivatives
(slope/aspect/watershed from SRTM/ASTER/Copernicus DEM) require raster tooling
not bundled here; this module records that boundary honestly and provides a
schema/provenance-aware loader.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TERRAIN = ROOT / "data" / "static" / "basin_terrain.json"
DEFAULT_BASINS = ROOT / "configs" / "basins.yaml"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_terrain(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_TERRAIN)
    if not p.exists():
        return {"schema_version": 1, "basins": {}, "provenance": {"status": "missing"}}
    data = _load_json(p)
    if not isinstance(data, dict):
        return {"schema_version": 1, "basins": {}, "provenance": {"status": "invalid"}}
    data.setdefault("basins", {})
    data.setdefault("provenance", {})
    return data


def terrain_for_basin(basin_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or load_terrain()
    return dict((payload.get("basins") or {}).get(str(basin_id)) or {})


def build_static_layer(
    basins: Iterable[dict[str, Any]],
    *,
    elevations_m: dict[str, float | None] | None = None,
    source: str = "configs/basins.yaml (+ optional Open-Meteo elevation points)",
    dem_catalog: list[str] | None = None,
) -> dict[str, Any]:
    """Build a provenance-rich static terrain layer without raster dependencies."""
    elevations_m = elevations_m or {}
    rows: dict[str, Any] = {}
    for basin in basins:
        basin_id = str(basin["basin_id"])
        config_elev = basin.get("elevation_mean_m")
        fetched = elevations_m.get(basin_id)
        rows[basin_id] = {
            "basin_id": basin_id,
            "name": basin.get("name", basin_id),
            "lat": float(basin["lat"]),
            "lon": float(basin["lon"]),
            "elevation_m": float(fetched)
            if fetched is not None
            else (float(config_elev) if config_elev is not None else None),
            "elevation_source": "open-meteo-elevation"
            if fetched is not None
            else "basin-config-approximate",
            "terrain_susceptibility": float(basin.get("terrain_susceptibility", 0.5)),
            "snow_fraction": float(basin.get("snow_fraction", 0.0)),
            "primary_mechanism": basin.get("primary_mechanism"),
            # Raster DEM derivatives: honest nulls until a DEM pipeline runs.
            "mean_slope_deg": None,
            "aspect_deg": None,
            "watershed_delineated": False,
            "dem_derivatives_source": None,
        }
    dem_catalog = dem_catalog or [
        "NASA Earthdata ASTER GDEM / NASADEM (30 m)",
        "USGS EarthExplorer SRTM",
        "OpenTopography lidar/topography (where available)",
        "Copernicus DEM (via Copernicus/partner access)",
    ]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "spatial_reference": "WGS84 point control points (approximate); not delineated catchments",
        "dem_catalog": dem_catalog,
        "basins": rows,
        "provenance": {
            "source": source,
            "method": "point elevation + configured basin attributes; raster slope/watershed pending DEM processing",
            "notes": [
                "Control points are not survey-grade catchment outlets.",
                "Slope/aspect/watershed require raster DEM processing (e.g. GDAL/richdem/Whitebox).",
                "Use this layer as static enrichment; do not claim sub-basin hydrologic precision.",
            ],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    payload["provenance"]["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_terrain(payload: dict[str, Any], path: str | Path | None = None) -> Path:
    p = Path(path or DEFAULT_TERRAIN)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p
