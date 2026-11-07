from __future__ import annotations

import json

import pandas as pd

from floodwatch.runner import run
from floodwatch.schema import validate_predictions


def test_demo_run_end_to_end(tmp_path):
    result = run(
        mode="demo",
        start="2022-08-20",
        end="2022-08-28",
        run_id="pytest_demo",
        use_models=False,
    )
    assert result.status == "ok"
    assert result.engine == "rule_engine"
    for name in (
        "predictions_a.parquet",
        "predictions_b.parquet",
        "predictions_fused.parquet",
        "alerts.json",
        "run_report.json",
    ):
        assert (result.run_dir / name).exists(), name

    fused = pd.read_parquet(result.run_dir / "predictions_fused.parquet")
    validate_predictions(fused)
    assert set(fused["module"]) == {"fused"}
    assert fused["basin_id"].nunique() == 8
    assert fused["prob"].between(0, 1).all()

    report = json.loads((result.run_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "demo"
    assert report["health"]["status"] == "ok"
    assert "not real data" in report["source"]["note"].lower()
