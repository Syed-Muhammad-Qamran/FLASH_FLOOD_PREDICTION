"""Generate the synthetic demo dataset and its proxy labels.

Run this first. It proves the pipeline end to end with zero credentials.
The numbers it produces are pipeline tests, not scientific results.
"""

from __future__ import annotations

import argparse

from _common import ROOT  # noqa: F401

from floodwatch import synthetic
from floodwatch.config import DATA_DIR
from floodwatch.labels import build_labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate even if cached")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()

    if args.force:
        table = synthetic.generate(args.start, args.end)
        path = synthetic.demo_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
    else:
        table = synthetic.load_or_generate()

    labelled, meta = build_labels(table)
    out = DATA_DIR / "demo" / "labels_demo.parquet"
    labelled.to_parquet(out, index=False)

    info = synthetic.describe(table)
    print("SYNTHETIC DEMO DATA - not real observations")
    print(f"  rows        : {info['rows']:,}")
    print(f"  basins      : {info['basins']}")
    print(f"  period      : {info['start_utc']} -> {info['end_utc']}")
    print(f"  max rain 3h : {info['max_rain_mm_3h']:.1f} mm")
    print()
    print("proxy label base rates")
    print(f"  A (cloudburst) train : {meta['base_rate_a_train']:.4%}")
    print(f"  A (cloudburst) test  : {meta['base_rate_a_test']:.4%}")
    print(f"  B (melt/high flow) train : {meta['base_rate_b_train']:.4%}")
    print(f"  B (melt/high flow) test  : {meta['base_rate_b_test']:.4%}")
    print()
    print(f"features -> {synthetic.demo_path()}")
    print(f"labels   -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
