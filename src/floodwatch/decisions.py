"""Alert decision state machine and atomic cooldown store.

Rules (v2 §6.2):
* alert at ORANGE and above only;
* repeat the same level only after the 6 h cooldown;
* an escalation alerts immediately even inside a cooldown;
* exactly one all-clear when the level drops below ORANGE;
* areas are processed in sorted order.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from .tiers import Level

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = ROOT / "data" / "state" / "area_state.json"

Action = Literal["alert", "all_clear", "none"]
COOLDOWN_HOURS = 6.0


def _as_level(value: Level | int | str) -> Level:
    if isinstance(value, Level):
        return value
    if isinstance(value, str):
        try:
            return Level[value.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown level name {value!r}") from exc
    return Level(int(value))


def _as_utc(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class AreaState:
    area_id: str
    level: Level = Level.GREEN
    last_alert_at: datetime | None = None
    alerted_level: Level | None = None
    all_clear_sent: bool = False


@dataclass(frozen=True)
class AlertDecision:
    area_id: str
    action: Action
    level: Level
    previous_level: Level
    reason: str
    cooldown_remaining_h: float | None = None


def decide(
    state: AreaState,
    level: Level | int | str,
    *,
    now: datetime | str | None = None,
    min_level: Level = Level.ORANGE,
    cooldown_hours: float = COOLDOWN_HOURS,
    escalate_only: bool = False,
) -> AlertDecision:
    """Decide one basin's action for this evaluation cycle."""
    current = _as_level(level)
    previous = _as_level(state.level)
    now_dt = _as_utc(now) or datetime.now(UTC)
    cooldown = timedelta(hours=float(cooldown_hours))

    def decision(action: Action, reason: str, remaining: float | None = None) -> AlertDecision:
        return AlertDecision(
            area_id=state.area_id,
            action=action,
            level=current,
            previous_level=previous,
            reason=reason,
            cooldown_remaining_h=remaining,
        )

    # Below the alert bar: the only possible action is a single all-clear.
    if current < min_level:
        if previous >= min_level and not state.all_clear_sent:
            return decision("all_clear", "level dropped below ORANGE")
        return decision("none", "below alert threshold")

    # At or above the alert bar.
    if previous < min_level:
        return decision("alert", "first transition to ORANGE or above")

    if current > previous:
        return decision("alert", f"escalation {previous.name} -> {current.name}")

    if current < previous:
        return decision("none", f"de-escalation {previous.name} -> {current.name}, no new page")

    # Same level: respect the cooldown unless this is escalate-only mode.
    if state.last_alert_at is not None:
        last = _as_utc(state.last_alert_at)
        elapsed = now_dt - last if last is not None else cooldown
        remaining = cooldown - elapsed
        remaining_h = max(0.0, remaining.total_seconds() / 3600.0)
        if remaining_h > 0:
            return decision(
                "none",
                "same level inside cooldown",
                round(remaining_h, 3),
            )
    if escalate_only:
        return decision("none", "escalate-only mode")
    return decision("alert", "same level after cooldown")


def apply_decisions(
    states: dict[str, AreaState],
    levels: dict[str, Level | int | str],
    *,
    now: datetime | str | None = None,
    min_level: Level = Level.ORANGE,
    cooldown_hours: float = COOLDOWN_HOURS,
    escalate_only: bool = False,
) -> tuple[list[AlertDecision], dict[str, AreaState]]:
    """Run the state machine deterministically over sorted area ids."""
    now_dt = _as_utc(now) or datetime.now(UTC)
    decisions: list[AlertDecision] = []
    updated: dict[str, AreaState] = {}

    for area_id in sorted(levels):
        priority = _as_level(levels[area_id])
        previous = states.get(area_id) or AreaState(area_id=area_id)
        decision = decide(
            previous,
            priority,
            now=now_dt,
            min_level=min_level,
            cooldown_hours=cooldown_hours,
            escalate_only=escalate_only,
        )
        decisions.append(decision)

        last_alert_at = previous.last_alert_at
        alerted_level = previous.alerted_level
        all_clear_sent = previous.all_clear_sent
        if decision.action == "alert":
            last_alert_at = now_dt
            alerted_level = priority
            all_clear_sent = False
        elif decision.action == "all_clear":
            all_clear_sent = True
        elif priority < min_level:
            # Keep the "all clear already sent" latch across quiet cycles.
            all_clear_sent = previous.all_clear_sent
        else:
            all_clear_sent = False

        updated[area_id] = AreaState(
            area_id=area_id,
            level=priority,
            last_alert_at=last_alert_at,
            alerted_level=alerted_level,
            all_clear_sent=all_clear_sent,
        )

    # Preserve states for areas absent from this cycle.
    for area_id, state in states.items():
        updated.setdefault(area_id, state)
    return decisions, updated


class StateStore:
    """Atomic JSON persistence for ``data/state/area_state.json``."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or DEFAULT_STATE_PATH)

    def load(self) -> dict[str, AreaState]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            return {}
        areas = raw.get("areas", {}) if isinstance(raw, dict) else {}
        if not isinstance(areas, dict):
            return {}
        out: dict[str, AreaState] = {}
        for area_id, block in areas.items():
            try:
                level = _as_level(block.get("level", "GREEN"))
                alerted = block.get("alerted_level")
                out[str(area_id)] = AreaState(
                    area_id=str(area_id),
                    level=level,
                    last_alert_at=_as_utc(block.get("last_alert_at")),
                    alerted_level=_as_level(alerted) if alerted else None,
                    all_clear_sent=bool(block.get("all_clear_sent", False)),
                )
            except (TypeError, ValueError, AttributeError):
                # One corrupt basin does not poison the rest of the store.
                continue
        return out

    def save(
        self,
        states: dict[str, AreaState],
        *,
        updated_at: datetime | str | None = None,
    ) -> Path:
        stamp = _as_utc(updated_at) or datetime.now(UTC)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "updated_at": stamp.isoformat(),
            "areas": {
                area_id: {
                    "level": state.level.name,
                    "last_alert_at": (
                        _as_utc(state.last_alert_at).isoformat() if state.last_alert_at else None
                    ),
                    "alerted_level": state.alerted_level.name if state.alerted_level else None,
                    "all_clear_sent": bool(state.all_clear_sent),
                }
                for area_id, state in sorted(states.items())
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path

    def update(
        self,
        levels: dict[str, Level | int | str],
        *,
        now: datetime | str | None = None,
        **kwargs: Any,
    ) -> list[AlertDecision]:
        states = self.load()
        decisions, updated = apply_decisions(states, levels, now=now, **kwargs)
        self.save(updated, updated_at=now)
        return decisions


def decision_to_row(decision: AlertDecision) -> dict[str, Any]:
    row = asdict(decision)
    row["level"] = decision.level.name
    row["previous_level"] = decision.previous_level.name
    return row


def action_counts(decisions: Iterable[AlertDecision]) -> dict[str, int]:
    counts = {"alert": 0, "all_clear": 0, "none": 0}
    for decision in decisions:
        counts[decision.action] = counts.get(decision.action, 0) + 1
    return counts


__all__ = [
    "AlertDecision",
    "AreaState",
    "COOLDOWN_HOURS",
    "StateStore",
    "action_counts",
    "apply_decisions",
    "decide",
    "decision_to_row",
]
