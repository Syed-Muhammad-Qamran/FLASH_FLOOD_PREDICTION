"""Single source of truth for the tier <-> level <-> public-word vocabulary.

Contract file: change only with a joint `contract` PR.

    tier      level    public word   alert?
    none      GREEN    -             no
    low       GREEN    -             no
    moderate  YELLOW   Watch         watch only
    high      ORANGE   Warning       yes
    severe    RED      Emergency     yes

`none` and `low` share GREEN on purpose: the difference stays inside the risk
result and the display table, never in a page.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal

Tier = Literal["none", "low", "moderate", "high", "severe"]

TIERS: tuple[str, ...] = ("none", "low", "moderate", "high", "severe")
TIER_RANK: dict[str, int] = {tier: rank for rank, tier in enumerate(TIERS)}


class Level(IntEnum):
    """Operational colour level. IntEnum keeps ordering (RED > ORANGE)."""

    GREEN = 0
    YELLOW = 1
    ORANGE = 2
    RED = 3


TIER_TO_LEVEL: dict[str, Level] = {
    "none": Level.GREEN,
    "low": Level.GREEN,
    "moderate": Level.YELLOW,
    "high": Level.ORANGE,
    "severe": Level.RED,
}

TIER_TO_PUBLIC_WORD: dict[str, str | None] = {
    "none": None,
    "low": None,
    "moderate": "Watch",
    "high": "Warning",
    "severe": "Emergency",
}

PUBLIC_WORD_TO_TIER: dict[str, str] = {
    word: tier for tier, word in TIER_TO_PUBLIC_WORD.items() if word is not None
}

ALL_LEVEL_WORDS: frozenset[str] = frozenset(level.name for level in Level)
ALL_PUBLIC_WORDS: frozenset[str] = frozenset(PUBLIC_WORD_TO_TIER)

# The lowest tier that maps to each level. Used when a caller only has a level.
LEVEL_TO_TIER: dict[Level, str] = {
    Level.GREEN: "none",
    Level.YELLOW: "moderate",
    Level.ORANGE: "high",
    Level.RED: "severe",
}


def _check_tier(tier: str) -> str:
    if tier not in TIER_RANK:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    return tier


def tier_to_level(tier: str) -> Level:
    return TIER_TO_LEVEL[_check_tier(tier)]


def tier_to_public_word(tier: str) -> str | None:
    return TIER_TO_PUBLIC_WORD[_check_tier(tier)]


def words_for_tier(tier: str) -> frozenset[str]:
    """Every word that legitimately describes this tier (level name + public word)."""
    _check_tier(tier)
    words = {tier_to_level(tier).name}
    public = tier_to_public_word(tier)
    if public:
        words.add(public)
    return frozenset(words)


def forbidden_words_for_tier(tier: str) -> frozenset[str]:
    """All vocabulary words that do NOT describe this tier."""
    return frozenset(ALL_LEVEL_WORDS | ALL_PUBLIC_WORDS) - words_for_tier(tier)


def level_to_tier(level: Level | int) -> str:
    return LEVEL_TO_TIER[Level(level)]


def tier_rank(tier: str) -> int:
    return TIER_RANK.get(tier, 0)


def max_tier(*tiers: str) -> str:
    return max(tiers, key=tier_rank) if tiers else "none"


def next_tier(tier: str) -> str:
    """The next tier upward; `severe` wraps to `none` as the explainer contract says."""
    _check_tier(tier)
    idx = TIER_RANK[tier]
    return TIERS[idx + 1] if idx + 1 < len(TIERS) else "none"


def escalate(tier: str, steps: int = 1) -> str:
    _check_tier(tier)
    return TIERS[min(TIER_RANK[tier] + steps, len(TIERS) - 1)]
