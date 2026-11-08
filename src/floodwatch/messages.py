"""Deterministic bilingual alert message templates.

Every rendered message carries the Latin basin id, exactly one Latin level word,
the execution timestamp and ``Rescue 1122``. Urdu messages keep the Latin level
word and basin id so the same verifier applies unchanged.
"""

from __future__ import annotations

from typing import Any

from .tiers import tier_to_level, tier_to_public_word

HOTLINE = "Rescue 1122"
_URDU_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def _level_word(tier: str) -> str:
    """Exactly one word: the Level name. Public words stay out of the text."""
    return tier_to_level(tier).name


def _level_phrase(tier: str, lang: str) -> str:
    if lang == "ur":
        return {"GREEN": "سبز", "YELLOW": "پیلا", "ORANGE": "نارنجی", "RED": "سرخ"}.get(
            tier_to_level(tier).name, "سبز"
        )
    return tier_to_public_word(tier) or ("watch" if tier == "moderate" else "normal")


def _reasons(result: Any) -> list[str]:
    reasons = getattr(result, "reasons", None) or []
    if isinstance(reasons, str):
        return [reasons]
    out: list[str] = []
    for item in reasons:
        if isinstance(item, dict):
            feature = str(item.get("feature", ""))
            weight = item.get("weight")
            out.append(f"{feature} ({float(weight):g})" if weight else feature)
        else:
            out.append(str(item))
    return out


def _stamp(result: Any) -> str:
    return str(getattr(result, "computed_at", ""))


def _area_id(result: Any) -> str:
    return str(getattr(result, "area_id", None) or getattr(result, "basin_id", ""))


def _score(result: Any) -> str:
    return f"{float(getattr(result, 'score', 0.0)):.3f}"


def format_alert(result: Any, *, lang: str = "en", catalog: Any = None) -> str:
    area = _area_id(result)
    tier = str(getattr(result, "tier", "none"))
    word = _level_word(tier)
    score = _score(result)
    stamp = _stamp(result)
    reasons = _reasons(result)

    if lang == "ur":
        lines = [
            "فلڈ واچ ہنگامی اطلاع",
            f"بیسن: {area}",
            f"سطح: {word}",
            f"خطرے کا سکور: {score}",
            f"جاری کردہ وقت (UTC): {stamp}",
        ]
        if reasons:
            lines.append("اسباب: " + "؛ ".join(reasons))
        lines.append(f"ہنگامی رابطہ: {HOTLINE}")
        return "\n".join(lines)

    lines = [
        "FLOODWATCH ALERT",
        f"Basin: {area}",
        f"Level: {word}",
        f"Risk score: {score}",
        f"Issued (UTC): {stamp}",
    ]
    if reasons:
        lines.append("Drivers: " + "; ".join(reasons))
    lines.append(f"Hotline: {HOTLINE}")
    return "\n".join(lines)


def format_all_clear(result: Any, *, lang: str = "en", catalog: Any = None) -> str:
    area = _area_id(result)
    tier = str(getattr(result, "tier", "none"))
    word = _level_word(tier)
    stamp = _stamp(result)

    if lang == "ur":
        return "\n".join(
            [
                "فلڈ واچ: خطرہ کم ہو گیا",
                f"بیسن: {area}",
                f"سطح: {word}",
                f"جاری کردہ وقت (UTC): {stamp}",
                "ابھی کوئی انخلا یا ہنگامی اطلاع نہیں۔",
                f"مقامی صورتحال کی تصدیق کریں۔ ہنگامی رابطہ: {HOTLINE}",
            ]
        )
    return "\n".join(
        [
            "FLOODWATCH ALL CLEAR",
            f"Basin: {area}",
            f"Level: {word}",
            f"Issued (UTC): {stamp}",
            "Alert condition has ended. No alert is active.",
            f"Verify local conditions. Hotline: {HOTLINE}",
        ]
    )


def format_test_message(area_id: str, *, computed_at: str, lang: str = "en") -> str:
    """Test broadcast. Says GREEN so it passes the same verifier."""
    if lang == "ur":
        return "\n".join(
            [
                "فلڈ واچ ٹیسٹ پیغام",
                f"بیسن: {area_id}",
                "سطح: GREEN",
                f"وقت (UTC): {computed_at}",
                "یہ ایک آزمائشی پیغام ہے، کوئی ہنگامی اطلاع نہیں۔",
                f"ہنگامی رابطہ: {HOTLINE}",
            ]
        )
    return "\n".join(
        [
            "FLOODWATCH TEST MESSAGE",
            f"Basin: {area_id}",
            "Level: GREEN",
            f"Issued (UTC): {computed_at}",
            "This is a test. No alert is active.",
            f"Hotline: {HOTLINE}",
        ]
    )


def format_decision(result: Any, action: str, *, lang: str = "en", catalog: Any = None) -> str:
    if action == "alert":
        return format_alert(result, lang=lang, catalog=catalog)
    if action == "all_clear":
        return format_all_clear(result, lang=lang, catalog=catalog)
    area = _area_id(result)
    tier = str(getattr(result, "tier", "none"))
    word = _level_word(tier)
    stamp = _stamp(result)
    if lang == "ur":
        return "\n".join(
            [
                "فلڈ واچ: کوئی نیا انتباہ نہیں",
                f"بیسن: {area}",
                f"سطح: {word}",
                f"وقت (UTC): {stamp}",
                f"ہنگامی رابطہ: {HOTLINE}",
            ]
        )
    return "\n".join(
        [
            "FLOODWATCH STATUS",
            f"Basin: {area}",
            f"Level: {word}",
            f"Issued (UTC): {stamp}",
            "No new alert is issued at this time.",
            f"Hotline: {HOTLINE}",
        ]
    )


__all__ = [
    "format_alert",
    "format_all_clear",
    "format_decision",
    "format_test_message",
]
