"""Ingest stream health: cadence, lag and staleness.

Qamran's lane in §6.7; implemented minimally because the runner report and the
dashboard must show it. Never crashes a run: a missing or unreadable health file
degrades to an ``unknown`` block.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = ROOT / "data" / "processed" / "source_health.json"
STREAMS = ("archive", "flood", "forecast")
CADENCE_H = {"archive": 24.0, "flood": 24.0, "forecast": 6.0}


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


@dataclass(frozen=True)
class SourceHealth:
    last_valid_utc: str | None
    lag_hours: float | None
    expected_cadence_h: float
    stale: bool
    fallback_used: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_source(
    name: str,
    *,
    last_valid_utc: str | datetime | None,
    expected_cadence_h: float | None = None,
    now: str | datetime | None = None,
    fallback_used: bool = False,
) -> SourceHealth:
    cadence = float(expected_cadence_h or CADENCE_H.get(name, 24.0))
    last = _as_utc(last_valid_utc)
    now_dt = _as_utc(now) or datetime.now(UTC)
    if last is None:
        return SourceHealth(None, None, cadence, True, fallback_used)
    lag = max(0.0, (now_dt - last).total_seconds() / 3600.0)
    return SourceHealth(
        last_valid_utc=last.isoformat(),
        lag_hours=round(lag, 2),
        expected_cadence_h=cadence,
        stale=bool(lag > 2.0 * cadence),
        fallback_used=fallback_used,
    )


def build_health(
    *,
    last_valid: dict[str, str | datetime | None],
    now: str | datetime | None = None,
    fallback: dict[str, bool] | None = None,
) -> dict[str, Any]:
    fallback = fallback or {}
    sources = {
        name: assess_source(
            name,
            last_valid_utc=last_valid.get(name),
            now=now,
            fallback_used=bool(fallback.get(name, False)),
        ).as_dict()
        for name in STREAMS
    }
    return {
        "schema_version": 1,
        "generated_at": (_as_utc(now) or datetime.now(UTC)).isoformat(),
        "any_stale": any(block["stale"] for block in sources.values()),
        "sources": sources,
    }


def write_source_health(payload: dict[str, Any], path: str | Path | None = None) -> Path:
    p = Path(path or DEFAULT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def read_source_health(path: str | Path | None = None) -> dict[str, Any]:
    """Read the health block; a missing file degrades, never crashes."""
    p = Path(path or DEFAULT_PATH)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return degraded_health("source health file is missing or unreadable")
    if not isinstance(payload, dict) or "sources" not in payload:
        return degraded_health("source health file is malformed")
    return payload


def degraded_health(reason: str) -> dict[str, Any]:
    sources = {
        name: SourceHealth(None, None, CADENCE_H[name], True, True).as_dict() for name in STREAMS
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "any_stale": True,
        "degraded": True,
        "reason": reason,
        "sources": sources,
    }


def stale_sources(payload: dict[str, Any]) -> list[str]:
    return [
        name
        for name, block in (payload.get("sources") or {}).items()
        if bool(block.get("stale", False))
    ]


__all__ = [
    "CADENCE_H",
    "STREAMS",
    "SourceHealth",
    "assess_source",
    "build_health",
    "degraded_health",
    "read_source_health",
    "stale_sources",
    "write_source_health",
]
