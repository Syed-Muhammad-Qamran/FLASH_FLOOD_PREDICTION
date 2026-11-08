#!/usr/bin/env python
"""Single test evaluation (M7). Runs once, by exclusive-create guard.

This script is the ONLY place allowed to read ``split == "test"`` rows. It
requires a human-approved ``--confirm <snapshot-hash>`` that matches the frozen
snapshot and writes ``runs/TEST_EVALUATED.json`` with ``open(..., "x")``. If the
marker already exists, it exits non-zero and does nothing.

Run (only after D2 freeze and explicit approval):
    python scripts/07_test_eval_once.py --confirm <snapshot-hash>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from floodwatch import fusion, schema, thresholds  # noqa: E402
from floodwatch.models import eval as metrics  # noqa: E402
from floodwatch.models import windows  # noqa: E402

MARKER_PATH = ROOT / "runs" / "TEST_EVALUATED.json"
SNAPSHOT_PATH = ROOT / "data" / "processed" / "SNAPSHOT.json"
GRID_PATH = ROOT / "data" / "processed" / "grid_index.parquet"
FEATURES_PATH = ROOT / "data" / "demo" / "features_demo.parquet"
PRED_A = ROOT / "artifacts" / "a" / "predictions_a.parquet"
PRED_B = ROOT / "artifacts" / "b" / "predictions_b.parquet"
REPORT_PATH = ROOT / "docs" / "TEST_REPORT.md"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _load_snapshot_hash() -> str:
    if SNAPSHOT_PATH.exists():
        return str(json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["data_snapshot_hash"])
    # Fall back to manifests; they must agree.
    hashes = []
    for path in (
        ROOT / "artifacts" / "a" / "manifest.json",
        ROOT / "artifacts" / "b" / "manifest.json",
    ):
        if path.exists():
            hashes.append(json.loads(path.read_text(encoding="utf-8"))["data_snapshot_hash"])
    if not hashes or len(set(hashes)) != 1:
        raise ValueError("cannot determine the frozen snapshot hash")
    return hashes[0]


def _create_marker(marker_path: Path, payload: dict[str, Any]) -> Path:
    """Exclusive-create: a second run exits non-zero before touching the test set."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with marker_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return marker_path


def _load_test_model(
    path: Path, module: str, grid: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    meta = schema.read_parquet_metadata(path)
    frame = pd.read_parquet(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    schema.require_columns(frame, schema.MODEL_FILE_COLUMNS, what=path.name)
    if (frame["module"].astype(str) != module).any():
        raise schema.SchemaError(f"{path.name} must contain module {module!r}")
    test = frame[frame["split"].astype(str) == "test"].copy()
    if test.empty:
        raise ValueError(f"{path.name} contains no test-split rows; cannot run M7")
    # Keep only test rows that exist on the canonical grid's test keys.
    test_grid = grid[grid["split"].astype(str) == "test"][["basin_id", "timestamp_utc"]]
    test = test.merge(
        test_grid.astype({"basin_id": str}), on=["basin_id", "timestamp_utc"], how="inner"
    )
    if test.empty:
        raise ValueError(f"{path.name} has no test rows aligned with the grid")
    return test.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True), meta


def run(*, confirm: str, marker_path: Path = MARKER_PATH) -> dict[str, Any]:
    if marker_path.exists():
        raise FileExistsError(
            f"{marker_path} already exists; the test split may be evaluated only once"
        )
    snapshot_hash = _load_snapshot_hash()
    if confirm != snapshot_hash:
        raise ValueError(
            f"--confirm {confirm!r} does not match the frozen snapshot hash {snapshot_hash!r}"
        )
    if not GRID_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError("M7 requires the frozen grid and demo feature table")

    grid = windows.read_grid_index(GRID_PATH)
    thresholds_config = thresholds.load_thresholds()
    thresholds.validate_thresholds(thresholds_config)

    pred_a, meta_a = _load_test_model(PRED_A, schema.MODULE_A, grid)
    pred_b, meta_b = _load_test_model(PRED_B, schema.MODULE_B, grid)
    if (
        meta_a.get("data_snapshot_hash") != snapshot_hash
        or meta_b.get("data_snapshot_hash") != snapshot_hash
    ):
        raise ValueError("model files do not match the confirmed snapshot hash")

    features = pd.read_parquet(FEATURES_PATH)
    features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True)
    labels = features[features["basin_id"].astype(str).isin(grid["basin_id"].astype(str))]
    labels = labels.merge(
        grid[grid["split"].astype(str) == "test"][["basin_id", "timestamp_utc"]],
        on=["basin_id", "timestamp_utc"],
        how="inner",
    )[["basin_id", "timestamp_utc", "label"]]

    fused = fusion.fuse_frames(
        pred_a,
        pred_b,
        thresholds_config=thresholds_config,
        meta_a=meta_a,
        meta_b=meta_b,
    )
    cut_a = thresholds_config["modules"][schema.MODULE_A]["tiers"]["high"]
    cut_b = thresholds_config["modules"][schema.MODULE_B]["tiers"]["high"]
    cut_f = thresholds_config["modules"][schema.MODULE_FUSED]["tiers"]["high"]

    report = {
        "schema_version": 1,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "data_snapshot_hash": snapshot_hash,
        "model_versions": {
            "a": str(pred_a["model_version"].iloc[0]),
            "b": str(pred_b["model_version"].iloc[0]),
        },
        "seeds": {"a": meta_a.get("seed", ""), "b": meta_b.get("seed", "")},
        "thresholds_fingerprint": thresholds_config["provenance_fingerprint"],
        "rows": int(len(fused)),
        "metrics": {
            "a": {
                key: round(float(value), 4)
                for key, value in metrics.evaluate(
                    labels, pred_a[["basin_id", "timestamp_utc", "prob"]], float(cut_a)
                ).items()
            },
            "b": {
                key: round(float(value), 4)
                for key, value in metrics.evaluate(
                    labels, pred_b[["basin_id", "timestamp_utc", "prob"]], float(cut_b)
                ).items()
            },
            "fused": {
                key: round(float(value), 4)
                for key, value in metrics.evaluate(
                    labels, fused[["basin_id", "timestamp_utc", "prob"]], float(cut_f)
                ).items()
            },
        },
    }
    _create_marker(marker_path, report)
    _write_test_report(report)
    return report


def _write_test_report(report: dict[str, Any]) -> None:
    lines = [
        "# Test report (M7, single evaluation)",
        "",
        "This file is written only by `scripts/07_test_eval_once.py` and is guarded by",
        "`runs/TEST_EVALUATED.json` (exclusive create). The test split was evaluated",
        "exactly once.",
        "",
        f"- Evaluated (UTC): {report['evaluated_at']}",
        f"- Git SHA: `{report['git_sha']}`",
        f"- Snapshot: `{report['data_snapshot_hash']}`",
        f"- Thresholds: `{report['thresholds_fingerprint']}`",
        f"- Rows: {report['rows']}",
        "",
        "| Module | PR-AUC | CSI | POD | FAR | Rows |",
        "|---|---|---|---|---|---|",
    ]
    for name, block in report["metrics"].items():
        lines.append(
            f"| {name} | {block['pr_auc']:.4f} | {block['csi']:.4f} | {block['pod']:.4f} "
            f"| {block['far']:.4f} | {int(block['rows'])} |"
        )
    lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True, help="frozen data snapshot hash")
    parser.add_argument("--marker", default=str(MARKER_PATH))
    args = parser.parse_args()
    try:
        report = run(confirm=args.confirm, marker_path=Path(args.marker))
    except (FileExistsError, ValueError, FileNotFoundError, schema.SchemaError) as exc:
        print(f"M7 refused: {exc}", file=sys.stderr)
        return 2
    print(f"test rows: {report['rows']}")
    for name, block in report["metrics"].items():
        print(f"  {name}: pr_auc={block['pr_auc']:.4f} csi={block['csi']:.4f}")
    print(f"marker: {args.marker}")
    print(f"report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
