"""The §3.7 model-output interface, checked on the committed fixtures.

The published prediction parquets are deliberately gitignored, so CI validates
the interface through the golden fixtures plus the committed manifests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from floodwatch import schema

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "golden"


def _grid(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.concat(
            [
                frame_a[["basin_id", "timestamp_utc", "split"]],
                frame_b[["basin_id", "timestamp_utc", "split"]],
            ]
        )
        .drop_duplicates(["basin_id", "timestamp_utc"])
        .reset_index(drop=True)
    )


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_a = pd.read_parquet(FIXTURE_DIR / "predictions_a.parquet")
    frame_b = pd.read_parquet(FIXTURE_DIR / "predictions_b.parquet")
    frame_a["timestamp_utc"] = pd.to_datetime(frame_a["timestamp_utc"], utc=True)
    frame_b["timestamp_utc"] = pd.to_datetime(frame_b["timestamp_utc"], utc=True)
    return frame_a, frame_b


def test_both_fixtures_obey_the_nine_column_contract() -> None:
    frame_a, frame_b = _fixtures()
    grid = _grid(frame_a, frame_b)
    schema.validate_model_file(FIXTURE_DIR / "predictions_a.parquet", schema.MODULE_A, grid)
    schema.validate_model_file(FIXTURE_DIR / "predictions_b.parquet", schema.MODULE_B, grid)


def test_fixtures_share_the_same_key_set() -> None:
    frame_a, frame_b = _fixtures()
    keys_a = set(zip(frame_a["basin_id"], frame_a["timestamp_utc"], strict=False))
    keys_b = set(zip(frame_b["basin_id"], frame_b["timestamp_utc"], strict=False))
    assert keys_a == keys_b


def test_fixtures_never_contain_test_rows() -> None:
    frame_a, frame_b = _fixtures()
    assert set(frame_a["split"]) <= {"train", "validation"}
    assert set(frame_b["split"]) <= {"train", "validation"}


def test_manifests_agree_on_the_data_snapshot() -> None:
    manifest_a = json.loads((ROOT / "artifacts" / "a" / "manifest.json").read_text())
    manifest_b = json.loads((ROOT / "artifacts" / "b" / "manifest.json").read_text())
    assert manifest_a["data_snapshot_hash"] == manifest_b["data_snapshot_hash"]
    assert manifest_a["module"] == schema.MODULE_A
    assert manifest_b["module"] == schema.MODULE_B
    assert manifest_b["architecture"]["lstm_layers"] == 1
    assert 32 <= manifest_b["architecture"]["hidden_units"] <= 64


def test_model_versions_are_prefixed_by_module() -> None:
    frame_a, frame_b = _fixtures()
    assert frame_a["model_version"].str.startswith(schema.MODULE_A).all()
    assert frame_b["model_version"].str.startswith(schema.MODULE_B).all()
