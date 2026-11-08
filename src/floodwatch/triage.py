"""Deterministic misinformation triage and local basin aliases.

The verdict is code only: which monitored basins are named, whether a fresh
official advisory covers them, and the current risk level. The module never
says "fake" or "true" and never executes instructions found in the input.

Person-1 lane; implemented here because the Streamlit app and the runner's
triage tab depend on it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .tiers import Level

ROOT = Path(__file__).resolve().parents[2]
ALIASES_PATH = ROOT / "data" / "static" / "basin_aliases.json"
ADVISORIES_PATH = ROOT / "data" / "advisories.json"

MAX_INPUT_CHARS = 20000
BANNED_EVIDENCE = (
    "fake",
    "false",
    "hoax",
    "untrue",
    "debunked",
    "not real",
    "did not happen",
    "never happened",
)

_URDU = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC = "٠١٢٣٤٥٦٧٨٩"
_DIGITS = {ord(ch): str(i) for i, ch in enumerate(_URDU)} | {
    ord(ch): str(i) for i, ch in enumerate(_ARABIC)
}

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
    "جنوری": 1,
    "فروری": 2,
    "مارچ": 3,
    "اپریل": 4,
    "مئی": 5,
    "جون": 6,
    "جولائی": 7,
    "اگست": 8,
    "ستمبر": 9,
    "اکتوبر": 10,
    "نومبر": 11,
    "دسمبر": 12,
}

_FLOOD_WORDS = (
    "flood",
    "flash flood",
    "water level",
    "river",
    "rain",
    "سیلاب",
    "طغیانی",
    "دریا",
    "بارش",
)

STANDARD_EVIDENCE = "Check PDMA / PMD / Rescue 1122. Do not forward unverified."


class Verdict(str, Enum):
    SUPPORTED = "supported_by_advisory"
    CONSISTENT = "consistent_with_conditions"
    NO_SIGNAL = "no_signal_found"
    CANNOT_ASSESS = "cannot_assess"


@dataclass(frozen=True)
class Advisory:
    title: str
    url: str
    areas: frozenset[str]
    issued_at: datetime | None = None
    age_hours: float | None = None
    source: str = ""


@dataclass(frozen=True)
class TriageResult:
    verdict: Verdict
    evidence: list[str] = field(default_factory=list)
    matched_basins: list[str] = field(default_factory=list)
    extracted_date: date | None = None
    level: Level | None = None
    truncated: bool = False


def normalise(text: str) -> str:
    return str(text).translate(_DIGITS)


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _flood_context(text: str, start: int, end: int, window: int = 40) -> bool:
    around = text[max(0, start - window) : min(len(text), end + window)].lower()
    return any(word in around for word in _FLOOD_WORDS)


def extract_dates(text: str, *, now: datetime | date | None = None) -> list[date]:
    """Extract plausible report dates. Future dates are never returned."""
    now_date = (
        now.date()
        if isinstance(now, datetime)
        else now
        if isinstance(now, date)
        else datetime.now(UTC).date()
    )
    cleaned = normalise(text)
    found: set[date] = set()

    # 2025-06-27
    for match in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", cleaned):
        _add_date(found, int(match.group(1)), int(match.group(2)), int(match.group(3)), now_date)

    # 27/06/2025 and 06/27/2025; ambiguous numeric dates are dd/mm/yyyy (D-02).
    for match in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", cleaned):
        first, second, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        candidate = _make_date(year, second, first, now_date)
        if candidate is None:
            candidate = _make_date(year, first, second, now_date)
        if candidate is not None:
            found.add(candidate)

    # 27 June 2025 / ۲۷ اگست ۲۰۲۲
    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    for match in re.finditer(
        rf"\b(\d{{1,2}})\s+({month_names})\s+(20\d{{2}})\b", cleaned, re.IGNORECASE
    ):
        day, month_name, year = match.group(1), match.group(2).lower(), match.group(3)
        month = _MONTHS.get(month_name)
        if month:
            _add_date(found, int(year), month, int(day), now_date)

    # August 2022 / اگست ۲۰۲۲
    for match in re.finditer(rf"\b({month_names})\s+(20\d{{2}})\b", cleaned, re.IGNORECASE):
        month_name, year = match.group(1).lower(), int(match.group(2))
        month = _MONTHS.get(month_name)
        if month:
            _add_date(found, year, month, 1, now_date)

    # Bare year: only near a flood word, never the current year, never in
    # "since/from/in ... season" phrasing.
    for match in re.finditer(r"(?<!\d)(20\d{2})(?!\d)", cleaned):
        year = int(match.group(1))
        if year == now_date.year or year > now_date.year:
            continue
        if not _flood_context(cleaned, match.start(), match.end()):
            continue
        prefix = cleaned[max(0, match.start() - 24) : match.start()].lower()
        suffix = cleaned[match.end() : match.end() + 24].lower()
        if re.search(r"\b(since|from)\s*$", prefix) or re.search(r"^\s*(?:season|موسم)", suffix):
            continue
        _add_date(found, year, 1, 1, now_date)

    return sorted(found)


def _make_date(year: int, month: int, day: int, now_date: date) -> date | None:
    try:
        candidate = date(year, month, day)
    except ValueError:
        return None
    return candidate if candidate <= now_date else None


def _add_date(found: set[date], year: int, month: int, day: int, now_date: date) -> None:
    candidate = _make_date(year, month, day, now_date)
    if candidate is not None:
        found.add(candidate)


def load_aliases(path: str | Path | None = None) -> dict[str, str]:
    data = json.loads(Path(path or ALIASES_PATH).read_text(encoding="utf-8"))
    aliases = data.get("aliases", {}) if isinstance(data, dict) else {}
    return {str(alias).strip().lower(): str(basin) for alias, basin in aliases.items()}


def load_advisories(path: str | Path | None = None) -> list[Advisory]:
    data = json.loads(Path(path or ADVISORIES_PATH).read_text(encoding="utf-8"))
    out: list[Advisory] = []
    for block in data.get("advisories", []) if isinstance(data, dict) else []:
        if not isinstance(block, dict):
            continue
        out.append(
            Advisory(
                title=str(block.get("title", "")),
                url=str(block.get("url", "")),
                areas=frozenset(str(area) for area in block.get("areas", [])),
                issued_at=_as_utc(block.get("issued_at")),
                source=str(block.get("source", "")),
            )
        )
    return out


def _matched_basins(text: str, aliases: dict[str, str]) -> list[str]:
    lowered = normalise(text).lower()
    matches: set[str] = set()
    # Longest alias first so "upper dir" wins over "dir".
    for alias in sorted(aliases, key=len, reverse=True):
        if alias and alias in lowered:
            matches.add(aliases[alias])
    return sorted(matches)


def _level_of(value: Any) -> Level | None:
    if value is None:
        return None
    if isinstance(value, Level):
        return value
    if isinstance(value, str):
        try:
            return Level[value.upper()]
        except KeyError:
            return None
    try:
        return Level(int(value))
    except (TypeError, ValueError):
        return None


def _fresh_advisory(
    advisories: Iterable[Advisory],
    matched: list[str],
    now_dt: datetime,
    max_age_h: float,
) -> Advisory | None:
    best: tuple[float, Advisory] | None = None
    for advisory in advisories:
        if not advisory.areas.intersection(matched):
            continue
        age = advisory.age_hours
        if advisory.issued_at is not None:
            age = (now_dt - advisory.issued_at).total_seconds() / 3600.0
        if age is None or age > max_age_h or age < 0:
            continue
        if best is None or age < best[0]:
            best = (age, advisory)
    return best[1] if best else None


def triage(
    text: str,
    *,
    aliases: dict[str, str] | None = None,
    levels: dict[str, Level | int | str] | None = None,
    advisories: Iterable[Advisory] = (),
    now: datetime | date | None = None,
    max_advisory_age_h: float = 48.0,
    stale_days: float = 30.0,
) -> TriageResult:
    """Deterministic verdict with bounded work and no model call."""
    now_dt = _as_utc(now) if isinstance(now, (str, datetime)) else None
    if now_dt is None:
        today = now if isinstance(now, date) else datetime.now(UTC).date()
        now_dt = datetime(today.year, today.month, today.day, tzinfo=UTC)

    raw = "" if text is None else str(text)
    truncated = len(raw) > MAX_INPUT_CHARS
    cleaned = normalise(raw[:MAX_INPUT_CHARS])
    aliases = load_aliases() if aliases is None else aliases
    levels = levels or {}
    advisories = list(advisories)

    matched = _matched_basins(cleaned, aliases)
    dates = extract_dates(cleaned, now=now_dt)
    newest = max(dates) if dates else None

    known_levels = {
        basin: _level_of(levels.get(basin))
        for basin in matched
        if _level_of(levels.get(basin)) is not None
    }
    level = max(known_levels.values()) if known_levels else None

    evidence: list[str] = []
    if not matched:
        verdict = Verdict.CANNOT_ASSESS
        evidence.append("No monitored basin is named in this message.")
    elif not known_levels:
        verdict = Verdict.CANNOT_ASSESS
        evidence.append(
            "No current risk estimate is available for "
            + ", ".join(matched)
            + "; the message cannot be assessed against conditions."
        )
    else:
        advisory = _fresh_advisory(advisories, matched, now_dt, max_advisory_age_h)
        if advisory is not None:
            verdict = Verdict.SUPPORTED
            evidence.append(
                f"Official advisory covers {', '.join(matched)}: "
                f"{advisory.title}" + (f" ({advisory.source})" if advisory.source else "")
            )
            if advisory.url:
                evidence.append(f"Advisory link: {advisory.url}")
        elif level >= Level.ORANGE:
            verdict = Verdict.CONSISTENT
            evidence.append(
                f"Current risk level for {', '.join(matched)} is {level.name}, "
                "so an event of this kind is plausible. No official source was found."
            )
        else:
            verdict = Verdict.NO_SIGNAL
            evidence.append(
                f"No model signal and no official advisory found for {', '.join(matched)} "
                f"(current level: {level.name}). Absence of a signal is not proof that "
                "nothing happened. It may also describe another place or an earlier date."
            )

    if newest is not None:
        age_days = (now_dt.date() - newest).days
        if age_days > stale_days:
            if verdict is Verdict.SUPPORTED:
                verdict = Verdict.CANNOT_ASSESS
                evidence.append(
                    f"The message describes {newest.isoformat()}; verify whether this is current."
                )
            elif verdict is Verdict.CONSISTENT:
                verdict = Verdict.NO_SIGNAL
                evidence.append(
                    f"The message describes {newest.isoformat()}, not current conditions."
                )
            else:
                evidence.append(f"The message describes {newest.isoformat()}.")

    evidence.append(STANDARD_EVIDENCE)
    for line in evidence:
        lowered = line.lower()
        for banned in BANNED_EVIDENCE:
            if banned in lowered:
                raise AssertionError(f"banned evidence word {banned!r} in {line!r}")

    return TriageResult(
        verdict=verdict,
        evidence=evidence,
        matched_basins=matched,
        extracted_date=newest,
        level=level,
        truncated=truncated,
    )


__all__ = [
    "Advisory",
    "TriageResult",
    "Verdict",
    "extract_dates",
    "load_advisories",
    "load_aliases",
    "triage",
]
