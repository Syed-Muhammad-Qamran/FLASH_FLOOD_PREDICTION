"""Produce a replay run over the August 2022 window (synthetic by default)."""

from __future__ import annotations

import argparse

from _common import ROOT  # noqa: F401

from floodwatch.runner import DEMO_END, DEMO_START, run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEMO_START)
    ap.add_argument("--end", default=DEMO_END)
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--force-demo", action="store_true")
    args = ap.parse_args()

    result = run(
        mode="demo",
        start=args.start,
        end=args.end,
        use_models=not args.no_models,
        force_demo=args.force_demo,
    )
    print(f"replay  : {result.run_id}")
    print(f"engine  : {result.engine}")
    print(f"status  : {result.status}")
    print(f"alerts  : {result.n_alerts}")
    print(f"output  : {result.run_dir}")
    print("\nOpen the app:  uv run streamlit run app/streamlit_app.py")
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
