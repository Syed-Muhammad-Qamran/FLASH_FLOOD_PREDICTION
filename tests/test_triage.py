"""Deterministic triage: aliases, dates, advisories and bounded verdicts."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from floodwatch import triage
from floodwatch.tiers import Level

NOW = datetime(2025, 7, 1, 12, tzinfo=UTC)
FRESH = triage.Advisory(
    title="PDMA flood advisory",
    url="https://example.test/advisory",
    areas=frozenset({"swat_kalam"}),
    issued_at=datetime(2025, 6, 30, 12, tzinfo=UTC),
    source="PDMA",
)
STALE = triage.Advisory(
    title="Old advisory",
    url="",
    areas=frozenset({"swat_kalam"}),
    issued_at=datetime(2025, 6, 1, 12, tzinfo=UTC),
)


def test_alias_matching_is_case_insensitive_and_longest_first() -> None:
    assert triage.triage(
        "Kalam valley is flooding", levels={"swat_kalam": Level.ORANGE}
    ).matched_basins == ["swat_kalam"]
    assert triage.triage(
        "Upper Dir is flooding", levels={"panjkora_dir": Level.ORANGE}
    ).matched_basins == ["panjkora_dir"]


def test_urdu_alias_matches() -> None:
    result = triage.triage("کالام میں سیلاب", levels={"swat_kalam": Level.ORANGE})
    assert result.matched_basins == ["swat_kalam"]


def test_no_basin_is_cannot_assess() -> None:
    result = triage.triage("water is rising quickly", levels={}, now=NOW)
    assert result.verdict is triage.Verdict.CANNOT_ASSESS
    assert result.matched_basins == []


def test_matched_basin_without_a_level_is_cannot_assess() -> None:
    result = triage.triage("Kalam is flooding", levels={}, now=NOW)
    assert result.verdict is triage.Verdict.CANNOT_ASSESS


def test_orange_without_an_advisory_is_consistent_with_conditions() -> None:
    result = triage.triage("Kalam is flooding", levels={"swat_kalam": Level.ORANGE}, now=NOW)
    assert result.verdict is triage.Verdict.CONSISTENT
    assert result.level is Level.ORANGE


def test_green_without_an_advisory_finds_no_signal() -> None:
    result = triage.triage("Kalam is flooding", levels={"swat_kalam": Level.GREEN}, now=NOW)
    assert result.verdict is triage.Verdict.NO_SIGNAL


def test_a_fresh_advisory_supports_the_claim() -> None:
    result = triage.triage(
        "Kalam is flooding", levels={"swat_kalam": Level.ORANGE}, advisories=[FRESH], now=NOW
    )
    assert result.verdict is triage.Verdict.SUPPORTED
    assert any("PDMA flood advisory" in line for line in result.evidence)


def test_a_stale_advisory_is_ignored() -> None:
    result = triage.triage(
        "Kalam is flooding", levels={"swat_kalam": Level.ORANGE}, advisories=[STALE], now=NOW
    )
    assert result.verdict is triage.Verdict.CONSISTENT


def test_an_old_date_downgrades_a_supported_claim() -> None:
    result = triage.triage(
        "Kalam flood on 2020-08-01",
        levels={"swat_kalam": Level.ORANGE},
        advisories=[FRESH],
        now=NOW,
    )
    assert result.verdict is triage.Verdict.CANNOT_ASSESS
    assert result.extracted_date == date(2020, 8, 1)


def test_an_old_date_downgrades_a_conditions_match() -> None:
    result = triage.triage(
        "Kalam flood on 2020-08-01", levels={"swat_kalam": Level.ORANGE}, now=NOW
    )
    assert result.verdict is triage.Verdict.NO_SIGNAL


def test_levels_accept_names_and_integers() -> None:
    by_name = triage.triage("Kalam flood", levels={"swat_kalam": "RED"}, now=NOW)
    by_int = triage.triage("Kalam flood", levels={"swat_kalam": 3}, now=NOW)
    assert by_name.level is Level.RED
    assert by_int.level is Level.RED


def test_oversized_input_is_truncated_and_flagged() -> None:
    result = triage.triage("Kalam " + "x" * (triage.MAX_INPUT_CHARS + 10), levels={}, now=NOW)
    assert result.truncated is True


def test_evidence_never_uses_banned_words() -> None:
    result = triage.triage("Kalam is flooding", levels={"swat_kalam": Level.GREEN}, now=NOW)
    lowered = " ".join(result.evidence).lower()
    for banned in triage.BANNED_EVIDENCE:
        assert banned not in lowered


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("flood on 27/06/2025", date(2025, 6, 27)),
        ("flood on 06/27/2025", date(2025, 6, 27)),
        ("flood on 27 June 2025", date(2025, 6, 27)),
        ("flood in August 2022", date(2022, 8, 1)),
        ("flood in 2022", date(2022, 1, 1)),
        ("سیلاب ۲۷ اگست ۲۰۲۲", date(2022, 8, 27)),
    ],
)
def test_date_extraction(text: str, expected: date) -> None:
    assert expected in triage.extract_dates(text, now=date(2025, 7, 1))


def test_future_dates_are_never_returned() -> None:
    assert triage.extract_dates("flood on 2025-12-31", now=date(2025, 7, 1)) == []


def test_bare_year_needs_flood_context() -> None:
    assert triage.extract_dates("meeting in 2022", now=date(2025, 7, 1)) == []
    assert triage.extract_dates("flood in 2022", now=date(2025, 7, 1)) == [date(2022, 1, 1)]


def test_season_phrasing_is_not_a_report_date() -> None:
    assert triage.extract_dates("flood since 2022 season", now=date(2025, 7, 1)) == []
