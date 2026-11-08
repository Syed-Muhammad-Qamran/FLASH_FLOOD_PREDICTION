"""
CLI Script: 09_climatology.py
=============================
Owner: Person 1 (Qamran)
Computes historical p50, p90, p95, p99 percentiles for rainfall and river discharge
to contextualize extreme flash flood events.
Usage:
    python scripts/09_climatology.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute climatological baseline percentiles.")
    parser.add_argument("--input", default="data/processed/raw_features.parquet", help="Raw features file")
    parser.add_argument("--output", default="data/static/climatology.json", help="Output JSON path")
    args = parser.parse_args()

    in_file = Path(args.input)
    if not in_file.exists():
        print("Raw features not found. Creating features first...")
        import importlib
        fetch_mod = importlib.import_module("scripts.01_fetch_openmeteo")
        fetch_mod.main()

    print("Computing climatological percentiles per basin...")
    df = pd.read_parquet(in_file)

    climatology = {
        "schema_version": 1,
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "basins": {}
    }

    for b_id, grp in df.groupby("basin_id"):
        precip = grp["precipitation"].dropna().values
        if len(precip) == 0:
            continue

        p50 = float(np.percentile(precip, 50))
        p90 = float(np.percentile(precip, 90))
        p95 = float(np.percentile(precip, 95))
        p99 = float(np.percentile(precip, 99))

        climatology["basins"][b_id] = {
            "rain_1h": {
                "p50": round(p50, 2),
                "p90": round(p90, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2)
            }
        }
        print(f"  {b_id}: p90={p90:.1f}mm, p95={p95:.1f}mm, p99={p99:.1f}mm")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(climatology, f, indent=2)

    print(f"Climatology profile saved to: {out_path}")


if __name__ == "__main__":
    main()
