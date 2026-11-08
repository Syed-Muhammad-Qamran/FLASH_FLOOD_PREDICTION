"""Historical Case Evaluation Reporter for Pakistan 2010 & 2022 Floods.

Evaluates Model A (Cloudburst Temporal NN) and Model B (Snowmelt LSTM) against
official external benchmark disaster ground truth from NDMA, PDMA KP, and FFC.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run_evaluation() -> None:
    models = [
        ("Model A", "a", "Extreme Convective Precipitation / Cloudburst Risk"),
        ("Model B", "b", "Snowmelt & Rain-on-Snow Risk"),
    ]

    for model_name, sub_dir, desc in models:
        manifest_p = ROOT / "artifacts" / sub_dir / "manifest_real.json"
        print("=" * 88)
        print(f"EVALUATION: {model_name.upper()} - {desc.upper()}")
        print("=" * 88)

        if not manifest_p.exists():
            print(f"Warning: {manifest_p} not found.")
            continue

        m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
        case_reports = m_data.get("case_reports", {})

        for year in ["2010", "2022"]:
            rep = case_reports.get(year)
            if not rep:
                print(f"No case report for {year} in {manifest_p.name}")
                continue

            title = rep["title"]
            w_start, w_end = rep["window_start"], rep["window_end"]
            mech = rep["mechanism"].upper()
            m = rep["metrics"]

            print(f"\n--- Case {year}: {title} ---")
            print(f"Window: {w_start} to {w_end} | Mechanism: {mech}")
            print(
                f"Metrics: POD={m['pod']:.3f} ({m['pod']*100:.1f}%) | "
                f"FAR={m['far']:.3f} | CSI={m['csi']:.3f} | F1={m['f1']:.3f} | Accuracy={m['accuracy']:.3f} | Brier={m['brier_score']:.4f}"
            )
            print(f"Contingency: TP={int(m['tp'])}, FP={int(m['fp'])}, FN={int(m['fn'])}, TN={int(m['tn'])}")

            rows = []
            for c in rep["comparisons"]:
                rows.append({
                    "Basin": c["basin_id"],
                    "Confirmed Status": c["confirmed_status"],
                    "Granularity": c["spatial_granularity"],
                    "Model Alert": "ALERT" if c["model_predicted_risk"] else "NORMAL",
                    "Max Prob": f"{c['model_max_prob']:.4f}",
                    "Tier": c["model_tier"],
                    "Outcome": c["comparison_outcome"],
                })
            df = pd.DataFrame(rows)
            print(df.to_string(index=False))
            print()


if __name__ == "__main__":
    run_evaluation()
