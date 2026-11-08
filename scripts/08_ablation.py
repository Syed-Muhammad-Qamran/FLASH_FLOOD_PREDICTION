#!/usr/bin/env python
"""Validation-only fusion ablation: six variants, one comparison table.

The script never reads or scores the test split. It refuses to run if a
prediction file contains test rows. Results go to
``artifacts/ablation_validation.json`` and ``docs/ABLATION.md``.

Run: python scripts/08_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from floodwatch import schema, thresholds  # noqa: E402
from floodwatch.models import eval as metrics  # noqa: E402

ART_A = ROOT / "artifacts" / "a" / "predictions_a.parquet"
ART_B = ROOT / "artifacts" / "b" / "predictions_b.parquet"
GRID_PATH = ROOT / "data" / "processed" / "grid_index.parquet"
FEATURES_PATH = ROOT / "data" / "demo" / "features_demo.parquet"

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "golden"
FIXTURE_A = FIXTURE_DIR / "predictions_a.parquet"
FIXTURE_B = FIXTURE_DIR / "predictions_b.parquet"
FIXTURE_LABELS = FIXTURE_DIR / "labels.parquet"

OUTPUT_PATH = ROOT / "artifacts" / "ablation_validation.json"
DOCS_PATH = ROOT / "docs" / "ABLATION.md"


def ensure_validation_only(frame: pd.DataFrame) -> None:
    """Refuse to run on any test-split row."""
    if "split" not in frame.columns:
        return
    splits = set(frame["split"].astype(str))
    if "test" in splits:
        raise ValueError("ablation refuses to run on test-split rows")


def grid_for(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Canonical key set from the prediction files themselves."""
    keys = pd.concat(
        [frame[["basin_id", "timestamp_utc", "split"]] for frame in frames],
        ignore_index=True,
    ).drop_duplicates(["basin_id", "timestamp_utc"])
    return keys.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)


def _has_rain_on_snow(value: object) -> bool:
    return any(
        "rain_on_snow" in str(driver.get("feature", "")) for driver in schema.parse_drivers(value)
    )


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    description: str
    prob: np.ndarray
    threshold: float


def build_variants(frame: pd.DataFrame, thresholds_config: dict[str, Any]) -> list[Variant]:
    modules = thresholds_config["modules"]
    cut_a = modules[schema.MODULE_A]["tiers"]
    cut_b = modules[schema.MODULE_B]["tiers"]
    cut_fused = modules[schema.MODULE_FUSED]["tiers"]

    prob_a = frame["prob_a"].to_numpy(dtype=float)
    prob_b = frame["prob_b"].to_numpy(dtype=float)
    max_prob = np.maximum(prob_a, prob_b)
    noisy_or = 1.0 - (1.0 - prob_a) * (1.0 - prob_b)

    compound = (prob_a >= float(cut_a["high"])) & (prob_b >= float(cut_b["high"]))
    rain_on_snow = frame["top_drivers_a"].map(_has_rain_on_snow).to_numpy(dtype=bool) | frame[
        "top_drivers_b"
    ].map(_has_rain_on_snow).to_numpy(dtype=bool)

    high = float(cut_fused["high"])
    severe = float(cut_fused["severe"])
    max_compound = np.where(compound, np.maximum(max_prob, high), max_prob)
    max_compound_ros = np.where(
        compound & rain_on_snow, np.maximum(max_compound, severe), max_compound
    )

    return [
        Variant(
            "a_only", "Model A only", "Cloudburst mechanism alone", prob_a, float(cut_a["high"])
        ),
        Variant("b_only", "Model B only", "Snowmelt mechanism alone", prob_b, float(cut_b["high"])),
        Variant("max", "max(prob_a, prob_b)", "Agreed fusion rule", max_prob, high),
        Variant(
            "noisy_or",
            "noisy_or(prob_a, prob_b)",
            "Probabilistic union of independent mechanisms",
            noisy_or,
            high,
        ),
        Variant(
            "max_compound",
            "max + compound",
            "max score, raised to the high floor when both mechanisms fire",
            max_compound,
            high,
        ),
        Variant(
            "max_compound_rain_on_snow",
            "max + compound + rain-on-snow",
            "max score with compound and rain-on-snow escalation floors",
            max_compound_ros,
            high,
        ),
    ]


def evaluate_variants(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    thresholds_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    predictions = frame[["basin_id", "timestamp_utc"]].copy()
    for variant in build_variants(frame, thresholds_config):
        scored = predictions.assign(prob=np.clip(variant.prob, 0.0, 1.0))
        block = metrics.evaluate(labels, scored, variant.threshold)
        rows.append(
            {
                "key": variant.key,
                "name": variant.name,
                "description": variant.description,
                "threshold": round(float(variant.threshold), 4),
                "metrics": {
                    key: round(float(value), 4)
                    for key, value in block.items()
                    if key
                    in {
                        "rows",
                        "pod",
                        "far",
                        "csi",
                        "precision",
                        "pr_auc",
                        "positive_rate",
                        "event_pod",
                        "median_lead_time_h",
                    }
                },
            }
        )
    return rows


def pick_best(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    def rank(row: dict[str, Any]) -> tuple[float, float, float]:
        block = row["metrics"]
        return (
            float(block.get("pr_auc", 0.0)),
            float(block.get("csi", 0.0)),
            -float(block.get("far", 1.0)),
        )

    ordered = sorted(rows, key=rank, reverse=True)
    best = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    block = best["metrics"]
    justification = (
        f"{best['name']} ranks first on validation PR-AUC "
        f"({block['pr_auc']:.4f}) and CSI ({block['csi']:.4f}) "
        f"with FAR {block['far']:.4f}."
    )
    if runner_up is not None:
        other = runner_up["metrics"]
        justification += (
            f" It beats {runner_up['name']} by "
            f"{block['pr_auc'] - other['pr_auc']:+.4f} PR-AUC and "
            f"{block['csi'] - other['csi']:+.4f} CSI."
        )
    return best, justification


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fusion ablation (validation only)",
        "",
        "Generated by `scripts/08_ablation.py`. The test split is never read.",
        "",
        f"- Rows: {report['rows']}",
        f"- Snapshot: `{report['data_snapshot_hash']}`",
        f"- Thresholds: `{report['thresholds_fingerprint']}`",
        f"- Best variant: **{report['best_variant']}**",
        "",
        report["justification"],
        "",
        "| Variant | POD | FAR | CSI | PR-AUC | Event POD | Median lead (h) | Threshold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["variants"]:
        block = row["metrics"]
        label = f"**{row['name']}**" if row["key"] == report["best_variant"] else row["name"]
        lines.append(
            f"| {label} | {block['pod']:.4f} | {block['far']:.4f} | {block['csi']:.4f} "
            f"| {block['pr_auc']:.4f} | {block.get('event_pod', 0.0):.4f} "
            f"| {block.get('median_lead_time_h', 0.0):.2f} | {row['threshold']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def load_default_inputs(
    a_path: Path, b_path: Path, grid_path: Path, labels_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    if a_path.exists() and b_path.exists() and grid_path.exists():
        grid = pd.read_parquet(grid_path)
        grid["timestamp_utc"] = pd.to_datetime(grid["timestamp_utc"], utc=True)
        if not labels_path.exists():
            if not FEATURES_PATH.exists():
                raise FileNotFoundError(
                    "labels are missing and data/demo/features_demo.parquet is absent"
                )
            features = pd.read_parquet(FEATURES_PATH)
            features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True)
            labels = features[["basin_id", "timestamp_utc", "label"]].merge(
                grid[["basin_id", "timestamp_utc", "split"]].astype(
                    {"basin_id": str, "split": str}
                ),
                on=["basin_id", "timestamp_utc"],
                how="inner",
            )
            labels_path = ROOT / "artifacts" / "ablation_labels.parquet"
            labels.to_parquet(labels_path, index=False)
        return a_path, b_path, grid, labels_path

    # CI has no processed grid or model parquets: the golden fixtures are
    # self-contained and carry the truth labels for the same rows.
    if not (FIXTURE_A.exists() and FIXTURE_B.exists() and FIXTURE_LABELS.exists()):
        raise FileNotFoundError("no prediction inputs found; run scripts/09_build_golden.py first")
    frame_a = pd.read_parquet(FIXTURE_A)
    frame_b = pd.read_parquet(FIXTURE_B)
    for frame in (frame_a, frame_b):
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return FIXTURE_A, FIXTURE_B, grid_for([frame_a, frame_b]), FIXTURE_LABELS


def run(
    *,
    a_path: Path,
    b_path: Path,
    grid: pd.DataFrame,
    labels_path: Path,
    output_path: Path = OUTPUT_PATH,
    docs_path: Path = DOCS_PATH,
) -> dict[str, Any]:
    thresholds_config = thresholds.load_thresholds()
    thresholds.validate_thresholds(thresholds_config)

    validated_a = schema.validate_model_file(a_path, schema.MODULE_A, grid)
    validated_b = schema.validate_model_file(b_path, schema.MODULE_B, grid)
    ensure_validation_only(validated_a)
    ensure_validation_only(validated_b)

    frame = validated_a[["basin_id", "timestamp_utc", "split", "prob", "top_drivers"]].rename(
        columns={"prob": "prob_a", "top_drivers": "top_drivers_a"}
    )
    frame = frame.merge(
        validated_b[["basin_id", "timestamp_utc", "prob", "top_drivers"]].rename(
            columns={"prob": "prob_b", "top_drivers": "top_drivers_b"}
        ),
        on=["basin_id", "timestamp_utc"],
        how="inner",
        validate="one_to_one",
    )
    frame = frame[frame["split"].astype(str) == "validation"].reset_index(drop=True)
    if frame.empty:
        raise ValueError("the validation split is empty; refusing to score test rows")

    labels = pd.read_parquet(labels_path)
    labels["timestamp_utc"] = pd.to_datetime(labels["timestamp_utc"], utc=True)
    labels = labels[labels["split"].astype(str) == "validation"][
        ["basin_id", "timestamp_utc", "label"]
    ]
    labels = labels[labels["basin_id"].astype(str).isin(frame["basin_id"].astype(str))]
    ensure_validation_only(labels.assign(split="validation"))

    merged_labels = frame[["basin_id", "timestamp_utc"]].merge(
        labels, on=["basin_id", "timestamp_utc"], how="left", validate="one_to_one"
    )
    if merged_labels["label"].isna().any():
        missing = int(merged_labels["label"].isna().sum())
        raise ValueError(f"{missing} validation rows have no truth label")
    merged_labels["label"] = merged_labels["label"].astype(int)

    variants = evaluate_variants(frame, merged_labels, thresholds_config)
    best, justification = pick_best(variants)
    meta_a = schema.read_parquet_metadata(a_path)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "split": "validation",
        "rows": int(len(frame)),
        "data_snapshot_hash": meta_a.get("data_snapshot_hash", ""),
        "thresholds_fingerprint": thresholds_config.get("provenance_fingerprint", ""),
        "variants": variants,
        "best_variant": best["key"],
        "best_variant_name": best["name"],
        "justification": justification,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-a", default=str(ART_A))
    parser.add_argument("--predictions-b", default=str(ART_B))
    parser.add_argument("--grid", default=str(GRID_PATH))
    parser.add_argument("--labels", default=str(ROOT / "artifacts" / "ablation_labels.parquet"))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--docs", default=str(DOCS_PATH))
    args = parser.parse_args()

    try:
        a_path, b_path, grid, labels_path = load_default_inputs(
            Path(args.predictions_a), Path(args.predictions_b), Path(args.grid), Path(args.labels)
        )
        report = run(
            a_path=a_path,
            b_path=b_path,
            grid=grid,
            labels_path=labels_path,
            output_path=Path(args.output),
            docs_path=Path(args.docs),
        )
    except (ValueError, FileNotFoundError, schema.SchemaError) as exc:
        print(f"ablation refused: {exc}", file=sys.stderr)
        return 2

    print(f"rows: {report['rows']}")
    for row in report["variants"]:
        block = row["metrics"]
        print(
            f"  {row['name']:<32} pr_auc={block['pr_auc']:.4f} "
            f"csi={block['csi']:.4f} pod={block['pod']:.4f} far={block['far']:.4f}"
        )
    print(f"best: {report['best_variant_name']}")
    print(f"report: {args.output}")
    print(f"table: {args.docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
