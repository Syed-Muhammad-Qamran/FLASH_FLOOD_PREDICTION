#!/usr/bin/env python
"""Fetch Open-Meteo data for the configured basins and update source health.

Network access is explicit. By default this script only validates configuration
and prints what it would fetch; pass ``--fetch`` to contact Open-Meteo.

Examples:
  python scripts/01_fetch_openmeteo.py --mode forecast --dry-run
  python scripts/01_fetch_openmeteo.py --mode forecast --fetch --cache
  python scripts/01_fetch_openmeteo.py --mode archive --start-date 2022-08-01 --end-date 2022-08-31 --fetch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from floodwatch.data_sources import cache, health, openmeteo  # noqa: E402

BASINS_PATH = ROOT / "configs" / "basins.yaml"
HEALTH_PATH = ROOT / "data" / "processed" / "source_health.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["forecast", "archive", "flood"], default="forecast")
    parser.add_argument("--fetch", action="store_true", help="perform network requests")
    parser.add_argument("--dry-run", action="store_true", help="print plan; never fetch")
    parser.add_argument(
        "--cache", action="store_true", help="write fetched frames to data/raw/cache"
    )
    parser.add_argument("--start-date", help="YYYY-MM-DD (archive mode)")
    parser.add_argument("--end-date", help="YYYY-MM-DD (archive mode)")
    parser.add_argument("--sleep-s", type=float, default=0.1)
    parser.add_argument("--basins", default=str(BASINS_PATH))
    parser.add_argument("--health-out", default=str(HEALTH_PATH))
    args = parser.parse_args()

    basins = yaml.safe_load(Path(args.basins).read_text(encoding="utf-8"))["basins"]
    print(f"basins: {len(basins)}  mode={args.mode}  fetch={args.fetch and not args.dry_run}")
    if args.dry_run or not args.fetch:
        for basin in basins:
            print(f"  - {basin['basin_id']}: {basin['lat']},{basin['lon']}")
        print("no network requests made (pass --fetch to contact Open-Meteo)")
        return 0

    results = []
    last_valid = {"archive": None, "flood": None, "forecast": None}
    fallback = {"archive": False, "flood": False, "forecast": False}
    try:
        for basin in basins:
            common = dict(
                basin_id=str(basin["basin_id"]),
                latitude=float(basin["lat"]),
                longitude=float(basin["lon"]),
                elevation_m=basin.get("elevation_mean_m"),
                terrain_susceptibility=basin.get("terrain_susceptibility"),
                snow_fraction=basin.get("snow_fraction"),
            )
            if args.mode == "flood":
                result = openmeteo.fetch_flood_discharge(
                    basin_id=common["basin_id"],
                    latitude=common["latitude"],
                    longitude=common["longitude"],
                )
            elif args.mode == "archive":
                if not args.start_date or not args.end_date:
                    print("archive mode requires --start-date and --end-date", file=sys.stderr)
                    return 2
                result = openmeteo.fetch_archive(
                    start_date=args.start_date, end_date=args.end_date, **common
                )
            else:
                result = openmeteo.fetch_forecast(**common)
            results.append(result)
            if not result.frame.empty and "timestamp_utc" in result.frame.columns:
                last = pd.to_datetime(result.frame["timestamp_utc"], utc=True).max()
                prev = last_valid.get(result.source)
                if prev is None or last > pd.Timestamp(prev):
                    last_valid[result.source] = last.isoformat()
            if args.cache and not result.frame.empty:
                key = cache.cache_key(
                    result.source,
                    basin_id=result.basin_id,
                    mode=args.mode,
                    start=args.start_date,
                    end=args.end_date,
                )
                cache.write_frame(
                    result.frame,
                    key=key,
                    meta={"source": result.source, "basin_id": result.basin_id},
                )
            print(
                f"  fetched {result.basin_id if hasattr(result, 'basin_id') else common['basin_id']}: {len(result.frame)} rows"
            )
            if args.sleep_s:
                import time

                time.sleep(args.sleep_s)
    except openmeteo.OpenMeteoError as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        fallback[args.mode if args.mode in fallback else "forecast"] = True

    payload = health.build_health(last_valid=last_valid, fallback=fallback)
    health.write_source_health(payload, args.health_out)
    print(f"source health: {args.health_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
