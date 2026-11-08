#!/usr/bin/env python
"""Run the FloodWatch pipeline end to end and write ``runs/<run_id>/``.

The runner is offline by default: it reads only the committed model-output
parquet files and the demo features. It never trains, never downloads, and
never transmits unless every delivery lock is open.

Run: python scripts/06_run.py --run-id demo --offline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floodwatch import runner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="demo", help="output directory name under runs/")
    parser.add_argument("--config", default=str(runner.DEFAULT_RUN_CONFIG), help="run config YAML")
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="never touch the network (the default)",
    )
    parser.add_argument(
        "--online",
        dest="offline",
        action="store_false",
        help="allow the configured delivery channel (still gated by FLOODWATCH_SEND)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="write the exact provider payload without opening a socket",
    )
    parser.add_argument(
        "--send",
        dest="dry_run",
        action="store_false",
        help="disable dry-run; requires delivery: whatsapp and FLOODWATCH_SEND=1",
    )
    parser.add_argument("--lang", choices=("en", "ur"), default=None, help="message language")
    parser.add_argument(
        "--as-of",
        default=None,
        help="ISO-8601 replay anchor; overrides the config value",
    )
    parser.add_argument(
        "--persist-state",
        action="store_true",
        help="write data/state/area_state.json (gitignored)",
    )
    parser.add_argument("--quiet", action="store_true", help="print only the output paths")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"missing run config: {config_path}", file=sys.stderr)
        return 2

    if args.as_of:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw["as_of"] = args.as_of
        temp_path = ROOT / "runs" / f".{args.run_id}.config.yaml"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        config_path = temp_path

    result = runner.run_pipeline(
        args.run_id,
        config_path=config_path,
        offline=args.offline,
        dry_run=args.dry_run,
        persist_state=args.persist_state,
        lang=args.lang,
    )

    report = result["report"]
    print(f"run: {result['run_dir']}")
    print(f"fused rows: {report['fusion']['rows']}")
    print(f"decisions: {report['decisions']}")
    print(f"delivery: {json.dumps(report['delivery'])}")
    print(f"verify: {report['verify']['checked']} checked, {report['verify']['blocked']} blocked")
    if report.get("degraded"):
        print(f"degraded: stale sources {report['sources'].get('stale_sources', [])}")
    if not args.quiet:
        print(f"report: {result['run_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
