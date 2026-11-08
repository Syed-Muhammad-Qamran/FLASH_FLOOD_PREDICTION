"""Alert assembly: decisions -> verified message drafts -> delivery.

The fallback template is deliberately minimal: basin id, level word, timestamp
and the hotline. It uses only numbers the verifier already allows, so a
formatting bug can never push an unverified number out of the door.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from . import messages, verify
from .channels import router


@dataclass(frozen=True)
class Outbound:
    area_id: str
    action: str
    text: str
    problems: list[str]
    blocked_by_verify: bool = False
    fallback_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "basin_id": self.area_id,
            "action": self.action,
            "text": self.text,
            "problems": self.problems,
            "blocked_by_verify": self.blocked_by_verify,
            "fallback_used": self.fallback_used,
        }


def fallback_text(result: Any, action: str) -> str:
    area_id = str(getattr(result, "area_id", "") or getattr(result, "basin_id", ""))
    tier = str(getattr(result, "tier", "none"))
    from .tiers import tier_to_level

    word = tier_to_level(tier).name
    stamp = str(getattr(result, "computed_at", ""))
    headline = "FLOODWATCH ALL CLEAR" if action == "all_clear" else "FLOODWATCH ALERT"
    return "\n".join(
        [
            headline,
            f"Basin: {area_id}",
            f"Level: {word}",
            f"Issued (UTC): {stamp}",
            f"Hotline: {messages.HOTLINE}",
        ]
    )


def build_outbound(
    results: Iterable[Any],
    decisions: Iterable[Any],
    *,
    lang: str = "en",
    include_none: bool = False,
) -> list[Outbound]:
    """Draft and verify one message per actionable decision."""
    by_area = {result.area_id: result for result in results}
    out: list[Outbound] = []
    for decision in decisions:
        action = str(getattr(decision, "action", "none"))
        if action == "none" and not include_none:
            continue
        area_id = str(decision.area_id)
        result = by_area.get(area_id)
        if result is None:
            continue
        text = messages.format_decision(result, action, lang=lang)
        problems = verify.check_risk_alert(text, result)
        blocked = bool(problems)
        fallback_used = False
        if problems:
            fallback = fallback_text(result, action)
            fallback_problems = verify.check_risk_alert(fallback, result)
            text = fallback
            fallback_used = True
            problems = problems + [f"fallback: {p}" for p in fallback_problems]
        out.append(
            Outbound(
                area_id=area_id,
                action=action,
                text=text,
                problems=problems,
                blocked_by_verify=blocked,
                fallback_used=fallback_used,
            )
        )
    return out


def deliver(
    outbound: Iterable[Outbound],
    *,
    cfg: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    dry_run: bool = True,
    out_dir: str | None = None,
    results: Iterable[Any] = (),
) -> router.DeliveryReport:
    by_area = {result.area_id: result for result in results}
    items = []
    for message in outbound:
        result = by_area.get(message.area_id)
        if result is None:
            continue
        items.append((result, message.action, message.text))
    return router.dispatch(
        items,
        cfg=cfg,
        environ=environ,
        dry_run=dry_run,
        out_dir=out_dir,
    )


def verify_summary(outbound: Iterable[Outbound], report: router.DeliveryReport) -> dict[str, Any]:
    outbound = list(outbound)
    blocked = [message.as_dict() for message in outbound if message.blocked_by_verify]
    problems: list[str] = []
    for message in outbound:
        problems.extend(f"{message.area_id}: {problem}" for problem in message.problems)
    return {
        "checked": len(outbound),
        "blocked": len(blocked) + int(report.blocked),
        "problems": problems,
        "blocked_messages": blocked,
    }


__all__ = ["Outbound", "build_outbound", "deliver", "fallback_text", "verify_summary"]
