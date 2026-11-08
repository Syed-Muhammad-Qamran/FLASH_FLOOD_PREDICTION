"""Pre-transmission alert verification (published, stable signatures).

``verify.py`` is the safety gatekeeper: every outbound alert, all-clear and test
message must pass :func:`check_risk_alert` before delivery.

This module is originally the person-1 lane; the signatures below are frozen by
the v2 backlog §6.3 and are implemented here because messages/runner/app depend
on them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .tiers import (
    ALL_LEVEL_WORDS,
    ALL_PUBLIC_WORDS,
    tier_to_public_word,
    words_for_tier,
)

MAX_LENGTH = 1600
HOTLINE = "1122"
FORBIDDEN_EXTRA = ("fake", "false", "hoax", "confirmed", "true")

_URDU = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_MAP = {ord(ch): str(i) for i, ch in enumerate(_URDU)} | {
    ord(ch): str(i) for i, ch in enumerate(_ARABIC)
}
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def normalise_digits(text: str) -> str:
    return str(text).translate(_DIGIT_MAP)


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(normalise_digits(text))


def _numeric_set(values: Iterable[str | float | int]) -> set[float]:
    out: set[float] = set()
    for value in values:
        for token in _numbers(str(value)):
            try:
                out.add(float(token))
            except ValueError:
                continue
    return out


def _format_variants(value: float | None) -> set[str]:
    if value is None:
        return set()
    variants = {
        f"{float(value):g}",
        f"{float(value):.2f}",
        f"{float(value):.3f}",
        f"{float(value) * 100:.0f}",
        f"{float(value) * 100:.1f}",
    }
    return variants


def _driver_strings(result: Any) -> list[str]:
    reasons = getattr(result, "reasons", None)
    if reasons is None:
        reasons = getattr(result, "top_drivers", None)
    if reasons is None:
        return []
    if isinstance(reasons, str):
        return [reasons]
    out: list[str] = []
    for item in reasons:
        if isinstance(item, dict):
            out.append(str(item.get("feature", "")))
            out.append(str(item.get("weight", "")))
        else:
            out.append(str(item))
    return out


def allowed_numbers_for(result: Any) -> set[str]:
    """Every number the result itself justifies, plus the emergency hotline."""
    allowed: set[str] = {HOTLINE, "1122"}
    for attribute in ("score", "prob", "prob_a", "prob_b", "baseline_prob"):
        value = getattr(result, attribute, None)
        if isinstance(value, (int, float)):
            allowed |= _format_variants(float(value))
    allowed |= set(_numbers(str(getattr(result, "computed_at", ""))))
    for text in _driver_strings(result):
        allowed |= set(_numbers(text))
    return allowed


def _whole_word(word: str, text: str) -> bool:
    return (
        re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text, re.IGNORECASE) is not None
    )


def check_alert(
    text: str,
    *,
    basin_id: str,
    tier: str,
    allowed_numbers: Iterable[str | float | int],
) -> list[str]:
    """Return a list of problems. An empty list means safe to send."""
    problems: list[str] = []
    if not isinstance(text, str) or not text.strip():
        return ["text is empty"]
    if len(text) > MAX_LENGTH:
        problems.append(f"text is {len(text)} characters; maximum is {MAX_LENGTH}")

    normalised = normalise_digits(text)
    if basin_id not in normalised:
        problems.append(f"basin id {basin_id!r} does not appear in the text")

    allowed_words = words_for_tier(tier)
    if not any(_whole_word(word, normalised) for word in allowed_words):
        expected = " or ".join(sorted(allowed_words))
        problems.append(f"text does not contain the expected level word ({expected})")

    forbidden = (ALL_LEVEL_WORDS | ALL_PUBLIC_WORDS) - allowed_words
    present = sorted(word for word in forbidden if _whole_word(word, normalised))
    if present:
        problems.append(f"conflicting level vocabulary present: {present}")

    allowed_values = _numeric_set(allowed_numbers)
    bad = sorted({token for token in _numbers(normalised) if float(token) not in allowed_values})
    if bad:
        problems.append(f"numbers not justified by the result: {bad}")

    lowered = normalised.lower()
    for word in FORBIDDEN_EXTRA:
        if re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", lowered):
            problems.append(f"speculative word {word!r} is not allowed")

    return problems


def check_risk_alert(
    text: str,
    result: Any,
    *,
    extra_numbers: Iterable[str | float | int] = (),
    extra_texts: Iterable[str] = (),
) -> list[str]:
    """Verify a rendered message against the result it claims to describe."""
    basin_id = str(getattr(result, "area_id", None) or getattr(result, "basin_id", ""))
    tier = str(getattr(result, "tier", "none"))
    allowed = allowed_numbers_for(result) | set(_numbers(" ".join(str(t) for t in extra_texts)))
    allowed |= set(extra_numbers)
    return check_alert(text, basin_id=basin_id, tier=tier, allowed_numbers=allowed)


def level_word(tier: str) -> str | None:
    return tier_to_public_word(tier)


__all__ = [
    "MAX_LENGTH",
    "allowed_numbers_for",
    "check_alert",
    "check_risk_alert",
    "normalise_digits",
]
