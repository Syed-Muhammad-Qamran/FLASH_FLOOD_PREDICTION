"""Pre-transmission verification rules."""

from __future__ import annotations

from floodwatch import verify
from floodwatch.interop import RiskResult
from floodwatch.tiers import Level


def _result(tier: str = "severe", score: float = 0.62) -> RiskResult:
    return RiskResult(
        area_id="swat_kalam",
        score=score,
        level={
            "none": Level.GREEN,
            "moderate": Level.YELLOW,
            "high": Level.ORANGE,
            "severe": Level.RED,
        }[tier],
        reasons=["rain_3h (12.5)"],
        computed_at="2022-08-27T00:00:00+00:00",
        tier=tier,
    )


def test_clean_text_passes() -> None:
    text = (
        "FLOODWATCH ALERT\nBasin: swat_kalam\nLevel: RED\nRisk score: 0.620\n"
        "Issued (UTC): 2022-08-27T00:00:00+00:00\nHotline: Rescue 1122"
    )
    assert verify.check_risk_alert(text, _result()) == []


def test_wrong_number_is_blocked() -> None:
    text = "Basin: swat_kalam\nLevel: RED\nRisk score: 0.990\nHotline: Rescue 1122"
    problems = verify.check_risk_alert(text, _result())
    assert any("numbers not justified" in problem for problem in problems)


def test_wrong_basin_is_blocked() -> None:
    text = "Basin: chitral_drosh\nLevel: RED\nRisk score: 0.620\nHotline: Rescue 1122"
    problems = verify.check_risk_alert(text, _result())
    assert any("basin id" in problem for problem in problems)


def test_two_level_words_are_blocked() -> None:
    text = "Basin: swat_kalam\nLevel: RED\nWarning\nRisk score: 0.620\nHotline: Rescue 1122"
    problems = verify.check_risk_alert(text, _result())
    assert any("conflicting level vocabulary" in problem for problem in problems)


def test_mixed_vocabulary_warning_and_red_is_blocked() -> None:
    text = "Basin: swat_kalam\nLevel: Warning\nRED\nHotline: Rescue 1122"
    problems = verify.check_risk_alert(text, _result("high"))
    assert any("conflicting level vocabulary" in problem for problem in problems)


def test_urdu_digits_are_accepted() -> None:
    text = "Basin: swat_kalam\nLevel: RED\nRisk score: ۰.۶۲\nHotline: Rescue 1122"
    assert verify.check_risk_alert(text, _result()) == []


def test_empty_and_overlong_text_are_blocked() -> None:
    assert verify.check_alert("", basin_id="swat_kalam", tier="severe", allowed_numbers=set())
    overlong = "swat_kalam RED " + ("x" * verify.MAX_LENGTH)
    assert any("maximum" in problem for problem in verify.check_risk_alert(overlong, _result()))


def test_speculative_words_are_blocked() -> None:
    text = "Basin: swat_kalam\nLevel: RED\nThis is confirmed true.\nHotline: Rescue 1122"
    problems = verify.check_risk_alert(text, _result())
    assert any("speculative" in problem for problem in problems)


def test_allowed_numbers_include_scores_and_timestamp() -> None:
    allowed = verify.allowed_numbers_for(_result())
    assert "1122" in allowed
    assert "0.62" in allowed
    assert "2022" in allowed


def test_all_five_tiers_have_a_consistent_word_set() -> None:
    for tier in ("none", "low", "moderate", "high", "severe"):
        from floodwatch.tiers import words_for_tier

        assert words_for_tier(tier)
