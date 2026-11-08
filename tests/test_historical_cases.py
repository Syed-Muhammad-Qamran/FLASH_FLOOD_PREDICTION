"""Tests for historical case configurations, provenance, and district-vs-basin mapping."""

from __future__ import annotations

import json
from pathlib import Path

from floodwatch.data_sources import historical_events

ROOT = Path(__file__).resolve().parents[1]


def test_historical_cases_config_validity() -> None:
    cases = historical_events.load_historical_cases()
    assert 2010 in cases, "2010 case missing from historical_cases.yaml"
    assert 2022 in cases, "2022 case missing from historical_cases.yaml"

    case_2010 = cases[2010]
    assert case_2010.case_id == "pakistan_superflood_2010"
    assert case_2010.primary_mechanism == "cloudburst"
    assert len(case_2010.ground_truth) == 8, "Expected 8 monitored basins"

    case_2022 = cases[2022]
    assert case_2022.case_id == "pakistan_monsoon_flood_2022"
    assert len(case_2022.ground_truth) == 8


def test_verified_events_json_schema() -> None:
    json_path = ROOT / "data" / "verified_events.json"
    assert json_path.exists(), "data/verified_events.json must exist"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == 1
    events = data.get("events", [])
    assert len(events) >= 2


def test_district_vs_basin_control_point_mapping_caveats() -> None:
    """Ensure that district reports do NOT automatically label all control points as flooded.

    In 2010 and 2022, Upper Chitral (chitral_mastuj) is recognized as having a different
    snow/GLOF regime and was NOT inundated by the lower Swat superflood wave.
    """
    cases = historical_events.load_historical_cases()
    for year in (2010, 2022):
        case = cases[year]
        kalam_gt = case.ground_truth["swat_kalam"]
        mastuj_gt = case.ground_truth["chitral_mastuj"]
        drosh_gt = case.ground_truth["chitral_drosh"]

        # Swat Kalam is confirmed flood
        assert kalam_gt.confirmed_status == "confirmed_flood"
        assert kalam_gt.comparable_with_model_a is True

        # Chitral Drosh gauge remained within normal seasonal envelope
        assert drosh_gt.confirmed_status == "no_confirmed_flood"

        # Mastuj is minor/localized melt and NOT comparable directly with cloudburst model A
        assert mastuj_gt.comparable_with_model_a is False
        assert mastuj_gt.comparable_with_model_b is True


def test_citations_provenance_integrity() -> None:
    citations = historical_events.list_citations()
    assert len(citations) >= 4, "Must have official citations for NDMA, PDMA, FFC, DFO"
    for cit in citations:
        assert cit["name"], "Citation must have a name"
        assert cit["url"], "Citation must have a source URL"
        assert cit["granularity"], "Citation must define its spatial granularity"
