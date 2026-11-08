"""Outbound delivery router with the three-lock safety gate.

Lock 1: ``delivery: whatsapp`` in the run config (default ``file``).
Lock 2: ``FLOODWATCH_SEND=1`` in the environment.
Lock 3: no global ``kill`` and no basin ``suppress`` in ``data/overrides.json``.

Always: the text has passed ``verify.check_risk_alert``. ``--dry-run`` writes
the exact provider payload and opens no socket.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import verify
from . import whatsapp

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OVERRIDES = ROOT / "data" / "overrides.json"
MAX_MESSAGES_PER_RUN = 5


@dataclass(frozen=True)
class DeliveryResult:
    area_id: str
    action: str
    mode: str
    attempted: bool
    sent: bool
    blocked: bool
    reasons: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    text: str = ""


@dataclass
class DeliveryReport:
    mode: str = "file"
    attempted: int = 0
    sent: int = 0
    failed: int = 0
    skipped: int = 0
    blocked: int = 0
    recorded: int = 0
    messages: list[DeliveryResult] = field(default_factory=list)
    payload_path: str | None = None
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "attempted": self.attempted,
            "sent": self.sent,
            "failed": self.failed,
            "skipped": self.skipped,
            "blocked": self.blocked,
            "recorded": self.recorded,
            "messages": [
                {
                    "area_id": message.area_id,
                    "action": message.action,
                    "mode": message.mode,
                    "attempted": message.attempted,
                    "sent": message.sent,
                    "blocked": message.blocked,
                    "reasons": message.reasons,
                    "problems": message.problems,
                }
                for message in self.messages
            ],
            "payload_path": self.payload_path,
            "problems": self.problems,
        }


def load_overrides(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_OVERRIDES)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return {"schema_version": 1, "kill": False, "basins": {}}
    if not isinstance(data, dict):
        return {"schema_version": 1, "kill": False, "basins": {}}
    data.setdefault("kill", False)
    data.setdefault("basins", {})
    return data


def gate(
    area_id: str,
    *,
    cfg: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Return ``(all_locks_open, reasons_for_failure)``."""
    cfg = cfg or {}
    env = environ if environ is not None else dict(os.environ)
    overrides = overrides if overrides is not None else load_overrides()
    reasons: list[str] = []

    if str(cfg.get("delivery", "file")).lower() != "whatsapp":
        reasons.append("lock1: delivery mode is not whatsapp")
    if env.get("FLOODWATCH_SEND") != "1":
        reasons.append("lock2: FLOODWATCH_SEND is not 1")
    if bool(overrides.get("kill", False)):
        reasons.append("lock3: global kill switch is active")
    basin_override = (overrides.get("basins") or {}).get(area_id) or {}
    if bool(basin_override.get("suppress", False)):
        reasons.append(f"lock3: basin {area_id} is suppressed by an operator override")

    return (not reasons, reasons)


def _write_payload(
    out_dir: Path, items: list[dict[str, Any]], cfg: whatsapp.WhatsAppConfig | None
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "outbound_payload.json"
    payload = {
        "dry_run": True,
        "api": f"{whatsapp.GRAPH_URL}/{whatsapp.API_VERSION}/<phone_number_id>/messages",
        "recipient_configured": bool(cfg),
        "messages": items,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def dispatch(
    items: Iterable[tuple[Any, str, str]],
    *,
    cfg: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    out_dir: str | Path | None = None,
) -> DeliveryReport:
    """Verify, gate and dispatch a bounded list of ``(result, action, text)``."""
    cfg = cfg or {}
    mode = "dry_run" if dry_run else str(cfg.get("delivery", "file")).lower()
    report = DeliveryReport(mode=mode)
    overrides = overrides if overrides is not None else load_overrides()
    env = environ if environ is not None else dict(os.environ)
    out_path = Path(out_dir) if out_dir else None
    wa_cfg: whatsapp.WhatsAppConfig | None = None
    if mode == "whatsapp" and not dry_run:
        try:
            wa_cfg = whatsapp.WhatsAppConfig.from_env(env)
        except whatsapp.WhatsAppError as exc:
            report.problems.append(str(exc))

    payload_items: list[dict[str, Any]] = []
    items = list(items)
    # The cap is a WhatsApp broadcast guard: a file-mode run simply records
    # every verified message and opens no socket.
    if mode != "file" and len(items) > MAX_MESSAGES_PER_RUN:
        report.problems.append(
            f"hard cap exceeded: {len(items)} messages > {MAX_MESSAGES_PER_RUN}; nothing sent"
        )
        report.skipped = len(items)
        return report

    for result, action, text in items:
        area_id = str(getattr(result, "area_id", "") or getattr(result, "basin_id", ""))
        problems = verify.check_risk_alert(text, result)
        if problems:
            report.blocked += 1
            report.skipped += 1
            report.messages.append(
                DeliveryResult(
                    area_id=area_id,
                    action=action,
                    mode=mode,
                    attempted=False,
                    sent=False,
                    blocked=True,
                    reasons=["verification failed"],
                    problems=problems,
                    text=text,
                )
            )
            continue

        if dry_run:
            # Build the exact provider body but never open a socket.
            to = env.get("WHATSAPP_TO", "<WHATSAPP_TO>")
            payload_items.append(
                {"area_id": area_id, "action": action, **whatsapp.build_payload(text, to=to)}
            )
            report.attempted += 1
            report.messages.append(
                DeliveryResult(
                    area_id=area_id,
                    action=action,
                    mode="dry_run",
                    attempted=True,
                    sent=False,
                    blocked=False,
                    reasons=["dry-run: payload written, no network call"],
                    text=text,
                )
            )
            continue

        if not dry_run and str(cfg.get("delivery", "file")).lower() != "whatsapp":
            report.recorded += 1
            report.messages.append(
                DeliveryResult(
                    area_id=area_id,
                    action=action,
                    mode="file",
                    attempted=False,
                    sent=False,
                    blocked=False,
                    reasons=["file mode: message recorded, not transmitted"],
                    text=text,
                )
            )
            continue

        allowed, reasons = gate(area_id, cfg=cfg, environ=env, overrides=overrides)
        if not allowed:
            report.blocked += 1
            report.skipped += 1
            report.messages.append(
                DeliveryResult(
                    area_id=area_id,
                    action=action,
                    mode=mode,
                    attempted=False,
                    sent=False,
                    blocked=True,
                    reasons=reasons,
                    text=text,
                )
            )
            continue

        report.attempted += 1
        try:
            if wa_cfg is None:
                wa_cfg = whatsapp.WhatsAppConfig.from_env(env)
            whatsapp.send_text(text, wa_cfg)
            report.sent += 1
            report.messages.append(
                DeliveryResult(
                    area_id=area_id,
                    action=action,
                    mode="whatsapp",
                    attempted=True,
                    sent=True,
                    blocked=False,
                    text=text,
                )
            )
        except whatsapp.WhatsAppError as exc:
            report.failed += 1
            report.messages.append(
                DeliveryResult(
                    area_id=area_id,
                    action=action,
                    mode="whatsapp",
                    attempted=True,
                    sent=False,
                    blocked=False,
                    reasons=[str(exc)],
                    text=text,
                )
            )

    if dry_run and out_path is not None:
        report.payload_path = str(_write_payload(out_path, payload_items, wa_cfg))
    return report


__all__ = [
    "DeliveryReport",
    "DeliveryResult",
    "MAX_MESSAGES_PER_RUN",
    "dispatch",
    "gate",
    "load_overrides",
]
