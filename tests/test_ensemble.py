"""Multi-seed ensemble: std, tier stability and provisional flags."""

from __future__ import annotations

import numpy as np
import pytest

from floodwatch import ensemble, schema
from floodwatch.tiers import tier_to_level


def _cutoffs() -> dict[str, float]:
    return {"none": 0.0, "low": 0.1, "moderate": 0.3, "high": 0.6, "severe": 0.8}


def test_ensemble_probabilities_mean_and_std() -> None:
    a = np.array([0.1, 0.5, 0.9])
    b = np.array([0.2, 0.5, 0.8])
    mean, std = ensemble.ensemble_probabilities([a, b])
    assert mean.tolist() == pytest.approx([0.15, 0.5, 0.85])
    assert std[0] == pytest.approx(0.05)
    assert std[1] == pytest.approx(0.0)


def test_single_seed_has_zero_std() -> None:
    a = np.array([0.2, 0.7])
    mean, std = ensemble.ensemble_probabilities([a])
    assert (std == 0.0).all()
    assert mean.tolist() == a.tolist()


def test_tier_stability_detects_disagreement() -> None:
    a = np.array([0.2, 0.75])
    b = np.array([0.2, 0.55])
    stable = ensemble.tier_stability([a, b], _cutoffs())
    assert stable.tolist() == [True, False]


def test_provisional_flags_disagreement_and_large_spread() -> None:
    std = np.array([0.01, 0.02, 0.10])
    stable = np.array([True, False, True])
    flags = ensemble.provisional_flags(std, stable, std_threshold=0.05)
    assert flags.tolist() == [False, True, True]


def test_build_ensemble_end_to_end() -> None:
    a = np.array([0.2, 0.75])
    b = np.array([0.2, 0.55])
    summary = ensemble.build_ensemble([a, b], _cutoffs(), std_threshold=0.05)
    assert summary.n_seeds == 2
    assert summary.tier_stable.tolist() == [True, False]
    assert summary.provisional.tolist() == [False, True]


def test_attach_ensemble_columns_single_seed_leaves_prob() -> None:
    import pandas as pd

    frame = pd.DataFrame({"prob": [0.4, 0.6]})
    summary = ensemble.build_ensemble([frame["prob"].to_numpy()], _cutoffs())
    out = ensemble.attach_ensemble_columns(frame, summary)
    assert out["prob"].tolist() == [0.4, 0.6]
    assert (out["prob_std"] == 0.0).all()
    assert out["tier_stable"].tolist() == [True, True]


def test_attach_ensemble_columns_multi_seed_overwrites_mean() -> None:
    import pandas as pd

    frame = pd.DataFrame({"prob": [0.4, 0.6]})
    a = np.array([0.3, 0.7])
    b = np.array([0.5, 0.5])
    summary = ensemble.build_ensemble([a, b], _cutoffs())
    out = ensemble.attach_ensemble_columns(frame, summary)
    assert out["prob"].tolist() == [0.4, 0.6]
    assert out["prob_std"].tolist() == pytest.approx([0.1, 0.1])


def test_ensemble_from_frames_aligns_and_carries_metadata() -> None:
    import pandas as pd

    frame_a = pd.DataFrame(
        {
            "basin_id": ["b", "a"],
            "timestamp_utc": pd.to_datetime(["2022-08-27T01:00Z", "2022-08-27T00:00Z"], utc=True),
            "prob": [0.3, 0.7],
            "module": ["B_snowmelt_lstm"] * 2,
            "split": ["validation"] * 2,
        }
    )
    frame_b = pd.DataFrame(
        {
            "basin_id": ["b", "a"],
            "timestamp_utc": pd.to_datetime(["2022-08-27T01:00Z", "2022-08-27T00:00Z"], utc=True),
            "prob": [0.5, 0.5],
            "module": ["B_snowmelt_lstm"] * 2,
            "split": ["validation"] * 2,
        }
    )
    out = ensemble.ensemble_from_frames([frame_a, frame_b], cutoffs=_cutoffs(), seeds=(1, 2))
    assert list(out["basin_id"]) == ["a", "b"]
    assert out["prob"].tolist() == [0.6, 0.4]
    assert out["module"].iloc[0] == "B_snowmelt_lstm"


def test_ensemble_from_frames_rejects_mismatched_keys() -> None:
    import pandas as pd

    a = pd.DataFrame(
        {
            "basin_id": ["a"],
            "timestamp_utc": pd.to_datetime(["2022-08-27T00:00Z"], utc=True),
            "prob": [0.5],
        }
    )
    b = pd.DataFrame(
        {
            "basin_id": ["b"],
            "timestamp_utc": pd.to_datetime(["2022-08-27T00:00Z"], utc=True),
            "prob": [0.5],
        }
    )
    try:
        ensemble.ensemble_from_frames([a, b], cutoffs=_cutoffs())
    except ValueError as exc:
        assert "key set" in str(exc)
        return
    raise AssertionError("mismatched keys were not rejected")


def test_attach_ensemble_length_mismatch_raises() -> None:
    import pandas as pd

    frame = pd.DataFrame({"prob": [0.4]})
    summary = ensemble.build_ensemble([np.array([0.4, 0.5])], _cutoffs())
    try:
        ensemble.attach_ensemble_columns(frame, summary)
    except ValueError:
        return
    raise AssertionError("length mismatch was not rejected")


def test_provisional_never_disrupts_fusion_tier() -> None:
    # Provisional is an annotation; fusion tiers come from probabilities.
    assert schema.tier_for(0.9, _cutoffs()) == "severe"
    assert tier_to_level("severe").name == "RED"
