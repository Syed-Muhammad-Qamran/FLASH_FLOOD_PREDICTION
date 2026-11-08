"""Bilingual templates must pass the same verifier for every tier."""

from __future__ import annotations

import pytest

from floodwatch import messages, verify
from floodwatch.interop import RiskResult
from floodwatch.tiers import Level

LEVELS = {
    "none": Level.GREEN,
    "low": Level.GREEN,
    "moderate": Level.YELLOW,
    "high": Level.ORANGE,
    "severe": Level.RED,
}


def _result(tier: str) -> RiskResult:
    return RiskResult(
        area_id="swat_kalam",
        score=0.123,
        level=LEVELS[tier],
        reasons=["rain_3h (12.5)", "soil_moisture (0.71)"],
        computed_at="2022-08-27T00:00:00+00:00",
        tier=tier,
    )


@pytest.mark.parametrize("lang", ["en", "ur"])
@pytest.mark.parametrize("tier", ["none", "low", "moderate", "high", "severe"])
def test_alert_passes_verification(tier: str, lang: str) -> None:
    result = _result(tier)
    text = messages.format_alert(result, lang=lang)
    assert verify.check_risk_alert(text, result) == []
    assert len(text) < 1600


@pytest.mark.parametrize("lang", ["en", "ur"])
@pytest.mark.parametrize("tier", ["none", "moderate", "severe"])
def test_all_clear_passes_verification(tier: str, lang: str) -> None:
    result = _result(tier)
    text = messages.format_all_clear(result, lang=lang)
    assert verify.check_risk_alert(text, result) == []


@pytest.mark.parametrize("lang", ["en", "ur"])
def test_test_message_is_green_and_passes(lang: str) -> None:
    result = _result("none")
    text = messages.format_test_message("swat_kalam", computed_at=result.computed_at, lang=lang)
    assert verify.check_risk_alert(text, result) == []


@pytest.mark.parametrize("lang", ["en", "ur"])
@pytest.mark.parametrize("action", ["alert", "all_clear", "none"])
def test_decision_dispatch_passes(action: str, lang: str) -> None:
    result = _result("high")
    text = messages.format_decision(result, action, lang=lang)
    assert verify.check_risk_alert(text, result) == []


def test_message_uses_exactly_one_latin_level_word() -> None:
    result = _result("high")
    text = messages.format_alert(result, lang="ur")
    assert "ORANGE" in text
    assert "Warning" not in text
    assert "GREEN" not in text and "RED" not in text


def test_message_contains_basin_id_and_hotline() -> None:
    text = messages.format_alert(_result("severe"))
    assert "swat_kalam" in text
    assert "Rescue 1122" in text
    assert "2022-08-27T00:00:00+00:00" in text


def test_no_speculative_words() -> None:
    for action in ("alert", "all_clear", "none"):
        for lang in ("en", "ur"):
            lowered = messages.format_decision(_result("severe"), action, lang=lang).lower()
            for word in ("fake", "hoax", "confirmed"):
                assert word not in lowered
