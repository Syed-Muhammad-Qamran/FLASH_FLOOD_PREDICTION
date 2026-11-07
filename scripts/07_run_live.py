"""One live operational pass against Open-Meteo. Shadow mode by default."""

from __future__ import annotations

import argparse

from _common import ROOT  # noqa: F401

from floodwatch.runner import run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-models", action="store_true")
    args = ap.parse_args()

    result = run(mode="live", use_models=not args.no_models)
    print(f"run     : {result.run_id}")
    print(f"engine  : {result.engine}")
    print(f"status  : {result.status}")
    print(f"alerts  : {result.n_alerts}")
    print(f"output  : {result.run_dir}")
    for key, value in result.report["health"].items():
        print(f"  health.{key}: {value}")
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
