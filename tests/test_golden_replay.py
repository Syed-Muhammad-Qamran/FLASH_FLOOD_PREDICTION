"""Golden replay: deterministic fusion regression on committed fixtures.

The fixtures are self-contained (a clean clone has no data grid and no model
parquets). A threshold edit, a fusion rule change or a driver-weight tweak
changes the expected fingerprint or the row values and fails loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from floodwatch import fusion, schema, thresholds

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
FIXTURE_A = FIXTURE_DIR / "predictions_a.parquet"
FIXTURE_B = FIXTURE_DIR / "predictions_b.parquet"
EXPECTED_PATH = FIXTURE_DIR / "golden_expected.json"
LABELS_PATH = FIXTURE_DIR / "labels.parquet"
MAX_FIXTURE_BYTES = 500_000
PROB_TOLERANCE = 1e-4


def _grid_from_fixtures(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> pd.DataFrame:
    grid = pd.concat(
        [
            frame_a[["basin_id", "timestamp_utc", "split"]],
            frame_b[["basin_id", "timestamp_utc", "split"]],
        ]
    ).drop_duplicates(["basin_id", "timestamp_utc"])
    return grid.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def replay():
    assert FIXTURE_A.exists(), f"missing fixture: {FIXTURE_A}"
    assert FIXTURE_B.exists(), f"missing fixture: {FIXTURE_B}"
    assert EXPECTED_PATH.exists(), f"missing expectation: {EXPECTED_PATH}"

    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    config = thresholds.load_thresholds()
    thresholds.validate_thresholds(config)

    raw_a = pd.read_parquet(FIXTURE_A)
    raw_b = pd.read_parquet(FIXTURE_B)
    raw_a["timestamp_utc"] = pd.to_datetime(raw_a["timestamp_utc"], utc=True)
    raw_b["timestamp_utc"] = pd.to_datetime(raw_b["timestamp_utc"], utc=True)
    grid = _grid_from_fixtures(raw_a, raw_b)

    frame_a = schema.validate_model_file(FIXTURE_A, schema.MODULE_A, grid)
    frame_b = schema.validate_model_file(FIXTURE_B, schema.MODULE_B, grid)
    fused = fusion.fuse_frames(
        frame_a,
        frame_b,
        thresholds_config=config,
        meta_a=schema.read_parquet_metadata(FIXTURE_A),
        meta_b=schema.read_parquet_metadata(FIXTURE_B),
    )
    return {"fused": fused, "expected": expected, "config": config}


def test_fixtures_stay_within_budget() -> None:
    total = sum(
        path.stat().st_size
        for path in (FIXTURE_A, FIXTURE_B, LABELS_PATH, EXPECTED_PATH)
        if path.exists()
    )
    assert total < MAX_FIXTURE_BYTES, (
        f"golden fixtures are {total} bytes; limit {MAX_FIXTURE_BYTES}"
    )


def test_replay_is_validation_only(replay) -> None:
    fused = replay["fused"]
    assert set(fused["split"]) <= {"train", "validation"}
    assert "test" not in set(fused["split"])
    expected_window = replay["expected"]["window"]
    assert fused["timestamp_utc"].min() >= pd.Timestamp(expected_window["start_utc"])
    assert fused["timestamp_utc"].max() < pd.Timestamp(expected_window["end_utc"])


def test_threshold_fingerprint_matches_expected(replay) -> None:
    assert (
        replay["expected"]["thresholds_fingerprint"] == replay["config"]["provenance_fingerprint"]
    ), (
        "configs/thresholds.yaml changed without regenerating the golden fixtures; "
        "run scripts/09_build_golden.py and review the diff"
    )


def test_replay_matches_every_expected_row(replay) -> None:
    fused = replay["fused"]
    expected_rows = replay["expected"]["rows"]
    assert len(fused) == len(expected_rows)

    actual = {
        (str(row.basin_id), pd.Timestamp(row.timestamp_utc).isoformat()): row
        for row in fused.itertuples(index=False)
    }
    mismatches: list[str] = []
    for expected in expected_rows:
        key = (expected["basin_id"], expected["timestamp_utc"])
        row = actual.get(key)
        if row is None:
            mismatches.append(f"{key}: missing from replay")
            continue
        if abs(float(row.prob) - float(expected["prob"])) > PROB_TOLERANCE:
            mismatches.append(
                f"{key}: prob {row.prob:.6f} != {expected['prob']:.6f} (tol {PROB_TOLERANCE})"
            )
        if str(row.tier) != expected["tier"]:
            mismatches.append(f"{key}: tier {row.tier} != {expected['tier']}")
        if bool(row.compound) != bool(expected["compound"]):
            mismatches.append(f"{key}: compound {row.compound} != {expected['compound']}")
        if bool(row.escalated) != bool(expected["escalated"]):
            mismatches.append(f"{key}: escalated {row.escalated} != {expected['escalated']}")
    assert not mismatches, "golden replay diverged:\n  " + "\n  ".join(mismatches[:10])


def test_threshold_edit_changes_the_fingerprint(replay) -> None:
    import copy

    edited = copy.deepcopy(replay["config"])
    edited["modules"][schema.MODULE_A]["tiers"]["high"] = round(
        float(edited["modules"][schema.MODULE_A]["tiers"]["high"]) + 0.05, 4
    )
    assert thresholds.fingerprint(edited) != replay["config"]["provenance_fingerprint"]


def test_hand_edited_thresholds_are_rejected(replay) -> None:
    import copy

    edited = copy.deepcopy(replay["config"])
    edited["modules"][schema.MODULE_B]["tiers"]["severe"] = 0.999
    with pytest.raises(ValueError, match="provenance_fingerprint mismatch"):
        thresholds.validate_thresholds(edited)
