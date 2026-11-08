#!/usr/bin/env python
"""Build the static DEM/terrain layer for monitored basins.

Creates ``data/static/basin_terrain.json`` with provenance. When ``--fetch`` is
passed, point elevations are refreshed from the Open-Meteo elevation API (no key).
Raster DEM derivatives (slope/aspect/watershed) are intentionally left null until
a DEM processing pipeline is added; the DEM catalog is recorded for traceability.

Run:
  python scripts/09_build_terrain.py                 # from basin config only
  python scripts/09_build_terrain.py --fetch         # refresh point elevations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from floodwatch.data_sources import openmeteo, terrain  # noqa: E402

BASINS_PATH = ROOT / "configs" / "basins.yaml"
OUT_PATH = ROOT / "data" / "static" / "basin_terrain.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true", help="fetch point elevations from Open-Meteo"
    )
    parser.add_argument("--basins", default=str(BASINS_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    basins = yaml.safe_load(Path(args.basins).read_text(encoding="utf-8"))["basins"]
    elevations: dict[str, float | None] = {}
    source = "configs/basins.yaml (approximate point attributes)"
    if args.fetch:
        try:
            points = [(float(b["lat"]), float(b["lon"])) for b in basins]
            values = openmeteo.fetch_elevations(points)
            elevations = {str(b["basin_id"]): values[i] for i, b in enumerate(basins)}
            source = "configs/basins.yaml + Open-Meteo elevation API (point elevations)"
            print(f"fetched {len([v for v in values if v is not None])}/{len(values)} elevations")
        except openmeteo.OpenMeteoError as exc:
            print(f"elevation fetch failed, using config values: {exc}", file=sys.stderr)

    payload = terrain.build_static_layer(basins, elevations_m=elevations, source=source)
    path = terrain.write_terrain(payload, args.out)
    print(f"wrote {path}")
    print(f"sha256: {payload['provenance']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
