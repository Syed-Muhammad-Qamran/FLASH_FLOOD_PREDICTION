"""
CLI Script: 10_log_event.py
===========================
Owner: Person 1 (Qamran)
Interactive/CLI tool for logging ground-truth verified flood events.
Usage:
    python scripts/10_log_event.py --basin swat_kalam --start 2022-08-26T04:00 --end 2022-08-26T18:00 --source "PDMA Flash Update" --notes "Severe bridge collapse"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Log ground truth verified flood event.")
    parser.add_argument("--basin", required=True, help="Monitored basin ID")
    parser.add_argument("--start", required=True, help="Start UTC timestamp (ISO)")
    parser.add_argument("--end", required=True, help="End UTC timestamp (ISO)")
    parser.add_argument("--source", required=True, help="Official source / citation URL")
    parser.add_argument("--notes", default="", help="Field notes / damages")
    parser.add_argument(
        "--db-path", default="data/verified_events.json", help="Event inventory file"
    )
    args = parser.parse_args()

    db_file = Path(args.db_path)
    if db_file.exists():
        with open(db_file, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"schema_version": 1, "events": []}

    event_id = f"EVT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{args.basin}"
    new_event = {
        "event_id": event_id,
        "basin_id": args.basin,
        "start_utc": args.start,
        "end_utc": args.end,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "notes": args.notes,
    }

    data["events"].append(new_event)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Logged verified flood event [{event_id}]: {args.basin} ({args.start} to {args.end})")


if __name__ == "__main__":
    main()
