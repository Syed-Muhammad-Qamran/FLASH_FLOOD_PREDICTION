"""Evaluate saved baselines on a split.

The test split is guarded: you must pass --confirm-test, and the script prints a
warning that this consumes the single clean evaluation. Do not run it to "check
something quickly" in week 4.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from _common import ROOT  # noqa: F401

from floodwatch.baselines import load_baseline
from floodwatch.config import DATA_DIR, MODELS_DIR, models_config
from floodwatch.evaluate import align_to_table, evaluate_series
from floodwatch.labels import build_labels
from floodwatch.synthetic import load_or_generate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        print("Refusing to touch the test split.")
        print("The test years are evaluated exactly once, in week 6, and reported once.")
        print("If this is that run, re-run with --confirm-test and record it in docs/log.md.")
        return 1

    if args.split == "test":
        print("=" * 70)
        print("CONSUMING THE SINGLE TEST EVALUATION. Record the run id in docs/log.md.")
        print("=" * 70)

    labelled = DATA_DIR / "demo" / "labels_demo.parquet"
    table = pd.read_parquet(labelled) if labelled.exists() else build_labels(load_or_generate())[0]

    cfg = models_config()
    rows = {}
    for module, label_col in (("A", "label_a"), ("B", "label_b")):
        horizon = int(cfg[f"model_{module.lower()}"]["horizon_steps"])
        model = load_baseline(module)
        prob = align_to_table(table, model.predict(table))
        rows[module] = evaluate_series(
            table, prob, label_col, args.split, model.threshold, horizon_steps=horizon
        )

    out = args.out or str(MODELS_DIR / f"evaluation_{args.split}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)

    print(f"\n{'module':<8}{'POD':>8}{'FAR':>8}{'CSI':>8}{'PR-AUC':>9}{'evPOD':>8}{'evFAR':>8}{'lead_h':>8}")
    for module, r in rows.items():
        print(
            f"{module:<8}{r['pod']:>8.3f}{r['far']:>8.3f}{r['csi']:>8.3f}"
            f"{r['pr_auc']:>9.3f}{r['event_pod']:>8.3f}{r['event_far']:>8.3f}"
            f"{r['median_lead_hours']:>8.1f}"
        )
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
