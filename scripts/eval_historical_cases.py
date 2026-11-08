#!/usr/bin/env python
"""Run external historical case evaluation for the 2010 and 2022 Pakistan Floods.
================================================================================
Compares official source-confirmed impact areas against model-predicted risk areas.
Outputs side-by-side verification reports, contingency statistics, and data provenance.

Usage:
  python scripts/eval_historical_cases.py [--year 2010|2022|all] [--model A|B]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floodwatch.data_sources.historical_events import list_citations, load_historical_cases  # noqa: E402
from floodwatch.models.eval_cases import evaluate_case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", choices=["2010", "2022", "all"], default="all")
    parser.add_argument("--model", choices=["A", "B", "both"], default="both")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    cases = load_historical_cases()
    years = [2010, 2022] if args.year == "all" else [int(args.year)]
    models = ["Model A", "Model B"] if args.model == "both" else [f"Model {args.model}"]

    print("=" * 80)
    print("FLOODWATCH KP: HISTORICAL CASE EVALUATION HARNESS")
    print("=" * 80)
    print("Official Reference Citations:")
    for cit in list_citations():
        print(f"  • {cit['name']} ({cit['date']}) — Granularity: {cit['granularity']}")

    for year in years:
        case = cases.get(year)
        if not case:
            continue
        print("\n" + "#" * 80)
        print(f"CASE {year}: {case.title.upper()}")
        print(f"Time Window: {case.start_utc} to {case.end_utc} (Peak: {case.peak_utc})")
        print(f"Primary Hydrological Trigger: {case.primary_mechanism.upper()}")
        print(f"Event Context: {case.description}")
        print("#" * 80)

        for model_name in models:
            is_model_a = "A" in model_name
            art_dir = ROOT / "artifacts" / ("a" if is_model_a else "b")
            manifest_p = art_dir / "manifest_real.json"

            # Check if precomputed case report exists in manifest
            report_dict = None
            if manifest_p.exists():
                try:
                    data = json.loads(manifest_p.read_text(encoding="utf-8"))
                    reports = data.get("case_reports", {})
                    report_dict = reports.get(str(year)) or reports.get(year)
                except Exception:
                    pass

            print(f"\n>>> Evaluation for {model_name} (Threshold: {args.threshold:.2f}) <<<")
            if report_dict:
                m = report_dict.get("metrics", {})
                print(
                    f"Contingency Counts (Comparable Basins): TP={m.get('tp')} | FP={m.get('fp')} | FN={m.get('fn')} | TN={m.get('tn')}"
                )
                print(
                    f"Verification Metrics: POD={m.get('pod')} | FAR={m.get('far')} | CSI={m.get('csi')} | F1={m.get('f1')} | Brier={m.get('brier_score')}"
                )
                print("\nDetailed Basin-by-Basin Breakdown:")
                print(
                    f"{'Basin ID':<18} {'Confirmed Status':<20} {'Pred Risk':<10} {'Max Prob':<10} {'Tier':<8} {'Outcome':<16} {'Spatial Granularity'}"
                )
                print("-" * 105)
                for comp in report_dict.get("comparisons", []):
                    b_id = comp["basin_id"]
                    conf = comp["confirmed_status"]
                    pred = "ALERT" if comp["model_predicted_risk"] else "NORMAL"
                    p_val = f"{comp['model_max_prob']:.3f}"
                    tier = comp["model_tier"]
                    outc = comp["comparison_outcome"]
                    gran = comp["spatial_granularity"]
                    print(
                        f"{b_id:<18} {conf:<20} {pred:<10} {p_val:<10} {tier:<8} {outc:<16} {gran}"
                    )
            else:
                # Run case evaluation using synthetic or demo predictions if real training hasn't been run yet
                demo_p = (
                    ROOT
                    / "artifacts"
                    / ("a" if is_model_a else "b")
                    / f"predictions_{'a' if is_model_a else 'b'}.parquet"
                )
                if not demo_p.exists():
                    demo_p = ROOT / "runs" / "demo" / "predictions_fused.parquet"
                if demo_p.exists():
                    df_preds = pd.read_parquet(demo_p)
                    # Add prob column if using fused
                    if "prob" not in df_preds.columns and (
                        "prob_a" in df_preds.columns or "prob_b" in df_preds.columns
                    ):
                        df_preds["prob"] = df_preds["prob_a"] if is_model_a else df_preds["prob_b"]
                    rep = evaluate_case(
                        case_year=year,
                        predictions_df=df_preds,
                        model_name=model_name,
                        threshold=args.threshold,
                    )
                    print(
                        f"Contingency Counts: TP={rep.metrics['tp']} | FP={rep.metrics['fp']} | FN={rep.metrics['fn']} | TN={rep.metrics['tn']}"
                    )
                    print(
                        f"Verification Metrics: POD={rep.metrics['pod']} | FAR={rep.metrics['far']} | CSI={rep.metrics['csi']} | F1={rep.metrics['f1']}"
                    )
                    for comp in rep.comparisons:
                        print(
                            f"  {comp.basin_id:<18}: Confirmed={comp.confirmed_status:<18} PredRisk={str(comp.model_predicted_risk):<5} MaxProb={comp.model_max_prob:.3f} [{comp.comparison_outcome}]"
                        )
                else:
                    print(
                        f"  No predictions available for {model_name}. Run train script or demo builder first."
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
