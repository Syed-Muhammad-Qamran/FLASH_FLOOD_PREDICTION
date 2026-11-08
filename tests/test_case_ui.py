"""Tests for historical case evaluation reporting and UI data contracts."""

from __future__ import annotations

import pandas as pd

from floodwatch.models import eval_cases


def test_eval_case_separates_truth_from_predictions() -> None:
    # Synthetic predictions for 2022 case window
    dates = pd.date_range("2022-08-25", "2022-08-28", freq="h", tz="UTC")
    rows = []
    for b_id in [
        "swat_kalam",
        "swat_madyan",
        "swat_chakdara",
        "panjkora_dir",
        "panjkora_talash",
        "chitral_mastuj",
        "chitral_drosh",
        "chitral_chitral",
    ]:
        for d in dates:
            # Predict high risk for Swat & Dir, low for Chitral
            p = 0.85 if "swat" in b_id or "panjkora" in b_id else 0.05
            rows.append({"basin_id": b_id, "timestamp_utc": d, "prob": p})

    preds_df = pd.DataFrame(rows)

    report = eval_cases.evaluate_case(
        case_year=2022,
        predictions_df=preds_df,
        model_name="Model A",
        threshold=0.50,
    )

    assert report.case_id == "pakistan_monsoon_flood_2022"
    assert len(report.comparisons) == 8

    # Swat Kalam: confirmed flood, model predicted risk -> TRUE_POSITIVE
    kalam_comp = next(c for c in report.comparisons if c.basin_id == "swat_kalam")
    assert kalam_comp.confirmed_status == "confirmed_flood"
    assert kalam_comp.model_predicted_risk is True
    assert kalam_comp.comparison_outcome == "TRUE_POSITIVE"

    # Chitral Drosh: no confirmed flood, model predicted low -> TRUE_NEGATIVE
    drosh_comp = next(c for c in report.comparisons if c.basin_id == "chitral_drosh")
    assert drosh_comp.confirmed_status == "no_confirmed_flood"
    assert drosh_comp.model_predicted_risk is False
    assert drosh_comp.comparison_outcome == "TRUE_NEGATIVE"

    # Chitral Mastuj: localized minor melt, non-comparable for cloudburst Model A
    mastuj_comp = next(c for c in report.comparisons if c.basin_id == "chitral_mastuj")
    assert mastuj_comp.is_comparable is False
    assert mastuj_comp.comparison_outcome == "NON_COMPARABLE"

    # Verify metrics block
    m = report.metrics
    assert m["pod"] > 0.0
    assert m["far"] == 0.0
    assert m["csi"] > 0.0
    assert "brier_score" in m


def test_eval_case_dictionary_serialization() -> None:
    dates = pd.date_range("2010-07-28", "2010-08-01", freq="h", tz="UTC")
    rows = [{"basin_id": "swat_kalam", "timestamp_utc": d, "prob": 0.9} for d in dates]
    rep = eval_cases.evaluate_case(
        case_year=2010, predictions_df=pd.DataFrame(rows), model_name="Model A"
    )
    d = rep.to_dict()
    assert d["year"] == 2010
    assert isinstance(d["comparisons"], list)
    assert isinstance(d["metrics"], dict)
    assert len(d["provenance_citations"]) > 0
