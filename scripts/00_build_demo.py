#!/usr/bin/env python
"""Build the synthetic demo dataset, canonical grid and placeholder Model A.

Person-2 scaffolding for the offline demo. The person-1 lane replaces the
placeholder A output with the real M2a model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floodwatch.demo_data import build_demo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate even if cached")
    args = parser.parse_args()
    result = build_demo(force=args.force)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
