"""Train both gradient-boosted baselines and score them on validation."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from _common import ROOT  # noqa: F401

from floodwatch.baselines import save_baseline, train_baseline
from floodwatch.config import DATA_DIR, MODELS_DIR, models_config
from floodwatch.evaluate import align_to_table, evaluate_series
from floodwatch.labels import build_labels
from floodwatch.synthetic import load_or_generate


def load_table(force: bool = False) -> pd.DataFrame:
    labelled = DATA_DIR / "demo" / "labels_demo.parquet"
    if labelled.exists() and not force:
        return pd.read_parquet(labelled)
    table, _ = build_labels(load_or_generate(force=force))
    labelled.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(labelled, index=False)
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-labels", action="store_true")
    args = ap.parse_args()

    cfg = models_config()
    table = load_table(force=args.force_labels)
    print(f"table: {len(table):,} rows, {table['basin_id'].nunique()} basins")

    summary = {}
    for module, label_col in (("A", "label_a"), ("B", "label_b")):
        horizon = int(cfg[f"model_{module.lower()}"]["horizon_steps"])
        print(f"\n=== baseline {module} (horizon {horizon} steps) ===")
        model = train_baseline(table, module, cfg)
        path = save_baseline(model)
        print(f"saved -> {path}")
        print(f"train rows {model.metadata['train_rows']:,}, "
              f"positives {model.metadata['positives']:,}")

        prob = align_to_table(table, model.predict(table))
        res = evaluate_series(
            table,
            prob,
            label_col,
            "validation",
            model.threshold,
            horizon_steps=horizon,
        )
        summary[module] = res
        print(
            f"validation  POD {res['pod']:.3f}  FAR {res['far']:.3f}  "
            f"CSI {res['csi']:.3f}  PR-AUC {res['pr_auc']:.3f}  "
            f"event POD {res['event_pod']:.3f}  lead {res['median_lead_hours']:.1f}h"
        )

    out = MODELS_DIR / "baseline_validation.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nreport -> {out}")
    print("\nReminder: if these numbers came from synthetic data they are pipeline tests only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
