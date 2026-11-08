"""Smoke tests for the validation-only ablation suite."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from floodwatch import thresholds

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "golden"


def _load_ablation():
    spec = importlib.util.spec_from_file_location(
        "ablation_script", ROOT / "scripts" / "08_ablation.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ablation():
    return _load_ablation()


@pytest.fixture(scope="module")
def inputs():
    frame_a = pd.read_parquet(FIXTURES / "predictions_a.parquet")
    frame_b = pd.read_parquet(FIXTURES / "predictions_b.parquet")
    labels = pd.read_parquet(FIXTURES / "labels.parquet")
    for frame in (frame_a, frame_b, labels):
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    config = thresholds.load_thresholds()
    thresholds.validate_thresholds(config)
    merged = frame_a[["basin_id", "timestamp_utc", "split", "prob", "top_drivers"]].rename(
        columns={"prob": "prob_a", "top_drivers": "top_drivers_a"}
    )
    merged = merged.merge(
        frame_b[["basin_id", "timestamp_utc", "prob", "top_drivers"]].rename(
            columns={"prob": "prob_b", "top_drivers": "top_drivers_b"}
        ),
        on=["basin_id", "timestamp_utc"],
        how="inner",
    )
    return merged, labels, config


def test_ablation_evaluates_six_variants(ablation, inputs) -> None:
    frame, labels, config = inputs
    rows = ablation.evaluate_variants(frame, labels, config)
    assert len(rows) == 6
    assert {row["key"] for row in rows} == {
        "a_only",
        "b_only",
        "max",
        "noisy_or",
        "max_compound",
        "max_compound_rain_on_snow",
    }
    for row in rows:
        block = row["metrics"]
        assert block["rows"] == 960
        assert 0.0 <= block["pod"] <= 1.0
        assert 0.0 <= block["far"] <= 1.0
        assert 0.0 <= block["pr_auc"] <= 1.0


def test_ablation_picks_a_best_variant_with_justification(ablation, inputs) -> None:
    frame, labels, config = inputs
    rows = ablation.evaluate_variants(frame, labels, config)
    best, justification = ablation.pick_best(rows)
    assert best["key"] in {row["key"] for row in rows}
    assert "PR-AUC" in justification
    assert justification.strip()


def test_ablation_refuses_test_rows(ablation) -> None:
    frame = pd.DataFrame({"split": ["validation", "test"]})
    with pytest.raises(ValueError, match="test-split"):
        ablation.ensure_validation_only(frame)


def test_ablation_writes_json_and_markdown(ablation, tmp_path: Path) -> None:
    a_path, b_path, grid, labels_path = ablation.load_default_inputs(
        FIXTURES / "predictions_a.parquet",
        FIXTURES / "predictions_b.parquet",
        FIXTURES / "grid_index_missing.parquet",
        FIXTURES / "labels.parquet",
    )
    report = ablation.run(
        a_path=a_path,
        b_path=b_path,
        grid=grid,
        labels_path=labels_path,
        output_path=tmp_path / "ablation.json",
        docs_path=tmp_path / "ABLATION.md",
    )
    assert report["split"] == "validation"
    assert report["rows"] == 960
    payload = json.loads((tmp_path / "ablation.json").read_text(encoding="utf-8"))
    assert payload["best_variant"] == report["best_variant"]
    text = (tmp_path / "ABLATION.md").read_text(encoding="utf-8")
    assert "validation only" in text
    assert report["best_variant_name"] in text or report["best_variant"] in text
