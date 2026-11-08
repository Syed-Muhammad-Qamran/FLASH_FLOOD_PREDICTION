"""End-to-end runner on a synthetic grid, built entirely in tmp_path.

The committed demo needs ``data/processed`` and the model parquets, both
gitignored, so this test constructs its own tiny snapshot. It proves the
runner writes every artefact, validates ``run_report.json`` and never touches
the test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from floodwatch import report_schema, runner, schema
from floodwatch.models import windows

BASINS = ("chitral_drosh", "swat_kalam")
SNAPSHOT = "0" * 63 + "1"


def _grid() -> pd.DataFrame:
    stamps = pd.date_range("2022-08-27T00:00Z", periods=6, freq="h")
    rows = []
    for basin in BASINS:
        for index, stamp in enumerate(stamps):
            rows.append(
                {
                    "basin_id": basin,
                    "timestamp_utc": stamp,
                    "split": "train" if index < 2 else "validation",
                }
            )
    return pd.DataFrame(rows)


def _predictions(grid: pd.DataFrame, module: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "basin_id": grid["basin_id"].astype(str).to_numpy(),
            "timestamp_utc": grid["timestamp_utc"].to_numpy(),
            "module": module,
            "split": grid["split"].astype(str).to_numpy(),
            "prob": 0.9,
            "baseline_prob": 0.2,
            "source": "deep",
            "top_drivers": ['[{"feature": "rain_3h", "weight": 2.0}]'] * len(grid),
            "model_version": f"{module}-test",
        }
    )


def _publish(path: Path, frame: pd.DataFrame, module: str, grid: pd.DataFrame) -> None:
    schema.publish_model_file(
        frame.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True),
        path,
        module=module,
        grid_index=grid,
        metadata={
            "schema_version": 1,
            "seed": 42,
            "data_snapshot_hash": SNAPSHOT,
            "label_horizon_h": 6,
            "trained_at": "2026-01-01T00:00:00Z",
            "module": module,
        },
    )


def test_runner_writes_every_artefact_and_a_valid_report(monkeypatch, tmp_path: Path) -> None:
    grid = _grid()
    grid_path = windows.write_grid_index(grid, tmp_path / "grid_index.parquet")
    snapshot_path = tmp_path / "SNAPSHOT.json"
    snapshot_path.write_text(json.dumps({"data_snapshot_hash": SNAPSHOT}), encoding="utf-8")
    pred_a = tmp_path / "predictions_a.parquet"
    pred_b = tmp_path / "predictions_b.parquet"
    _publish(pred_a, _predictions(grid, schema.MODULE_A), schema.MODULE_A, grid)
    _publish(pred_b, _predictions(grid, schema.MODULE_B), schema.MODULE_B, grid)

    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "delivery": "file",
                "language": "en",
                "dry_run": False,
                "cooldown_hours": 6,
                "audit_hours": 24,
                "as_of": "2022-08-27T05:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runner, "GRID_PATH", grid_path)
    monkeypatch.setattr(runner, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(runner, "PRED_A", pred_a)
    monkeypatch.setattr(runner, "PRED_B", pred_b)
    health = {
        "schema_version": 1,
        "sources": {
            "archive": {"stale": False},
            "flood": {"stale": False},
            "forecast": {"stale": True},
        },
    }
    monkeypatch.setattr(runner.health, "read_source_health", lambda: health)

    result = runner.run_pipeline(
        "synthetic", config_path=config_path, offline=True, run_root=tmp_path / "runs"
    )
    run_dir = Path(result["run_dir"])
    for name in (
        "predictions_fused.parquet",
        "run_report.json",
        "alerts.json",
        "outbound.json",
        "decisions.jsonl",
    ):
        assert (run_dir / name).exists(), name

    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    report_schema.validate_report(report)
    assert report["fusion"]["rows"] == len(grid)
    assert report["evaluated_at"].startswith("2022-08-27T05:00:00")
    assert report["delivery"]["mode"] == "file"
    assert report["delivery"]["recorded"] == 2
    assert report["delivery"]["sent"] == 0
    assert report["verify"]["checked"] == 2
    assert report["verify"]["blocked"] == 0
    assert report["degraded"] is True
    assert "forecast" in report["sources"]["stale_sources"]
    assert report["data_snapshot_hash"] == SNAPSHOT

    fused = pd.read_parquet(run_dir / "predictions_fused.parquet")
    assert set(fused["split"]) <= {"train", "validation"}
    assert len(fused) == len(grid)

    outbound = json.loads((run_dir / "outbound.json").read_text(encoding="utf-8"))
    assert len(outbound) == 2
    assert all(message["blocked_by_verify"] is False for message in outbound)

    decisions = [
        json.loads(line) for line in (run_dir / "decisions.jsonl").read_text().splitlines() if line
    ]
    assert len(decisions) == len(grid)


def test_missing_source_health_degrades_instead_of_crashing(monkeypatch, tmp_path: Path) -> None:
    grid = _grid()
    grid_path = windows.write_grid_index(grid, tmp_path / "grid_index.parquet")
    snapshot_path = tmp_path / "SNAPSHOT.json"
    snapshot_path.write_text(json.dumps({"data_snapshot_hash": SNAPSHOT}), encoding="utf-8")
    pred_a = tmp_path / "predictions_a.parquet"
    pred_b = tmp_path / "predictions_b.parquet"
    _publish(pred_a, _predictions(grid, schema.MODULE_A), schema.MODULE_A, grid)
    _publish(pred_b, _predictions(grid, schema.MODULE_B), schema.MODULE_B, grid)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump({"delivery": "file", "dry_run": False, "as_of": "2022-08-27T05:00:00Z"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "GRID_PATH", grid_path)
    monkeypatch.setattr(runner, "SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(runner, "PRED_A", pred_a)
    monkeypatch.setattr(runner, "PRED_B", pred_b)
    monkeypatch.setattr(
        runner.health, "read_source_health", lambda: runner.health.degraded_health("test")
    )

    result = runner.run_pipeline(
        "degraded", config_path=config_path, offline=True, run_root=tmp_path / "runs"
    )
    report = json.loads(Path(result["run_dir"], "run_report.json").read_text(encoding="utf-8"))
    assert report["degraded"] is True
    assert set(report["sources"]["stale_sources"]) == {"archive", "flood", "forecast"}
