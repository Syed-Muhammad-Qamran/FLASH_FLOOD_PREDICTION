"""Propose operating thresholds and tier cutoffs from validation data."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from _common import ROOT  # noqa: F401

from floodwatch.baselines import load_baseline
from floodwatch.calibrate import calibrate_module, write_proposal
from floodwatch.config import DATA_DIR, MODELS_DIR, models_config
from floodwatch.evaluate import align_to_table
from floodwatch.labels import build_labels
from floodwatch.synthetic import load_or_generate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["validation"])
    args = ap.parse_args()

    labelled = DATA_DIR / "demo" / "labels_demo.parquet"
    table = pd.read_parquet(labelled) if labelled.exists() else build_labels(load_or_generate())[0]
    cfg = models_config()

    results = {}
    for module, label_col in (("A", "label_a"), ("B", "label_b")):
        horizon = int(cfg[f"model_{module.lower()}"]["horizon_steps"])
        model = load_baseline(module)
        prob = align_to_table(table, model.predict(table))
        results[module] = calibrate_module(
            table, prob, label_col, horizon_steps=horizon, split=args.split
        )

    proposal = write_proposal(results, MODELS_DIR / "thresholds_proposal.yaml")
    (MODELS_DIR / "calibration_report.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    for module, r in results.items():
        op = r["operating"]
        print(
            f"module {module}: threshold {op['threshold']:.4f}  "
            f"CSI {op['csi']:.3f}  POD {op['pod']:.3f}  FAR {op['far']:.3f}  "
            f"alert rate {op['alert_rate']:.2%}"
        )
        print(f"  tiers: {', '.join(f'{k}={v:.3f}' for k, v in r['tiers'].items())}")

    print(f"\nproposal -> {proposal}")
    print("Review it, then copy the values into configs/thresholds.yaml under `calibrated:`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
