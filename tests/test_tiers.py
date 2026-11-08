"""The tier vocabulary is the single source of truth."""

from __future__ import annotations

import pytest

from floodwatch import tiers


def test_table_matches_the_contract() -> None:
    assert tiers.TIERS == ("none", "low", "moderate", "high", "severe")
    assert tiers.tier_to_level("none") is tiers.Level.GREEN
    assert tiers.tier_to_level("low") is tiers.Level.GREEN
    assert tiers.tier_to_level("moderate") is tiers.Level.YELLOW
    assert tiers.tier_to_level("high") is tiers.Level.ORANGE
    assert tiers.tier_to_level("severe") is tiers.Level.RED


def test_public_words() -> None:
    assert tiers.tier_to_public_word("none") is None
    assert tiers.tier_to_public_word("low") is None
    assert tiers.tier_to_public_word("moderate") == "Watch"
    assert tiers.tier_to_public_word("high") == "Warning"
    assert tiers.tier_to_public_word("severe") == "Emergency"


def test_level_ordering() -> None:
    assert tiers.Level.RED > tiers.Level.ORANGE > tiers.Level.YELLOW > tiers.Level.GREEN
    assert tiers.tier_rank("severe") > tiers.tier_rank("high") > tiers.tier_rank("moderate")


def test_no_public_word_collision() -> None:
    words = [word for word in tiers.TIER_TO_PUBLIC_WORD.values() if word]
    assert len(words) == len(set(words))
    assert tiers.PUBLIC_WORD_TO_TIER["Warning"] == "high"


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("none", "low"),
        ("low", "moderate"),
        ("moderate", "high"),
        ("high", "severe"),
        ("severe", "none"),
    ],
)
def test_next_tier(tier: str, expected: str) -> None:
    assert tiers.next_tier(tier) == expected


def test_escalate_is_capped() -> None:
    assert tiers.escalate("moderate") == "high"
    assert tiers.escalate("high", 2) == "severe"
    assert tiers.escalate("severe", 5) == "severe"


def test_max_tier() -> None:
    assert tiers.max_tier("none", "high", "low") == "high"
    assert tiers.max_tier() == "none"


def test_unknown_tier_raises() -> None:
    with pytest.raises(ValueError):
        tiers.tier_to_level("unknown")


def test_words_for_tier_never_overlap() -> None:
    # ``none`` and ``low`` intentionally share GREEN; every other word must be
    # unique to its tier so the verifier can count level vocabulary.
    seen: set[str] = set()
    for tier in tiers.TIERS:
        words = tiers.words_for_tier(tier)
        level_word = tiers.tier_to_level(tier).name
        public = words - {level_word}
        assert not (public & seen)
        seen |= public
        assert level_word in words
