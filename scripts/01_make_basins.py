"""Write a placeholder basins.geojson so the pipeline runs before delineation.

The polygons are buffered control points, not catchments. Replace this file in
week 1 with a HydroSHEDS / MERIT-Hydro delineation that keeps the same basin_id
values. Everything downstream keys on basin_id only.
"""

from __future__ import annotations

import json

from _common import ROOT  # noqa: F401

from floodwatch.config import basins, basins_geojson_path


def square(lon: float, lat: float, half: float = 0.22) -> list[list[list[float]]]:
    return [
        [
            [lon - half, lat - half],
            [lon + half, lat - half],
            [lon + half, lat + half],
            [lon - half, lat + half],
            [lon - half, lat - half],
        ]
    ]


def main() -> int:
    features = []
    for b in basins():
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "basin_id": b["basin_id"],
                    "name": b["name"],
                    "area_km2": b["area_km2"],
                    "elevation_mean_m": b["elevation_mean_m"],
                    "snow_fraction": b["snow_fraction"],
                    "primary_mechanism": b["primary_mechanism"],
                    "geometry_source": "placeholder_point_buffer",
                },
                "geometry": {"type": "Polygon", "coordinates": square(b["lon"], b["lat"])},
            }
        )

    payload = {
        "type": "FeatureCollection",
        "name": "kp_peshawar_placeholder",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    path = basins_geojson_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {len(features)} basins -> {path}")
    print("NOTE: placeholder geometry. Replace with a real delineation in week 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
