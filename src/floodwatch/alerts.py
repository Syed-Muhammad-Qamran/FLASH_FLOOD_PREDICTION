"""Alert construction and delivery.

Shadow mode is the default: alerts are written to `alerts.json` and shown in the
app, but nothing is sent to a human. Leave shadow mode only after the week-7
replay gate and only with a named person on call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import TIER_RANK, parse_drivers


def build_alerts(
    fused: pd.DataFrame,
    *,
    min_tier: str = "high",
    horizon_hours: float | None = None,
) -> list[dict[str, Any]]:
    if fused.empty:
        return []
    rank = TIER_RANK[min_tier]
    df = fused.copy()
    df["_rank"] = df["tier"].map(lambda t: TIER_RANK.get(str(t), 0))
    df = df[df["_rank"] >= rank]
    if df.empty:
        return []

    if horizon_hours is not None:
        cutoff = df["timestamp_utc"].max() - pd.Timedelta(hours=horizon_hours)
        df = df[df["timestamp_utc"] >= cutoff]
        if df.empty:
            return []

    alerts: list[dict[str, Any]] = []
    for basin_id, grp in df.groupby("basin_id"):
        grp = grp.sort_values("timestamp_utc")
        peak_idx = grp["prob"].idxmax()
        peak = grp.loc[peak_idx]
        first = grp.iloc[0]
        alerts.append(
            {
                "basin_id": str(basin_id),
                "first_alert_utc": first["timestamp_utc"].isoformat(),
                "peak_utc": peak["timestamp_utc"].isoformat(),
                "tier": str(peak["tier"]),
                "peak_prob": round(float(peak["prob"]), 4),
                "drivers": parse_drivers(peak["top_drivers"]),
                "compound": bool(
                    any("compound" in d.lower() for d in parse_drivers(peak["top_drivers"]))
                ),
                "lead_hours_from_first_alert": round(
                    (peak["timestamp_utc"] - first["timestamp_utc"]).total_seconds() / 3600, 1
                ),
            }
        )

    alerts.sort(key=lambda a: (TIER_RANK.get(a["tier"], 0), a["peak_prob"]), reverse=True)
    return alerts


def format_text(alert: dict[str, Any]) -> str:
    drivers = "; ".join(alert.get("drivers", [])[:3])
    flag = " [COMPOUND]" if alert.get("compound") else ""
    return (
        f"{alert['tier'].upper()}{flag} - {alert['basin_id']} "
        f"(p={alert['peak_prob']:.2f}, peak {alert['peak_utc']})\n"
        f"  cause: {drivers}"
    )


def format_telegram(alerts: list[dict[str, Any]], run_id: str) -> str:
    lines = [f"FloodWatch shadow alert - run {run_id}", ""]
    lines += [format_text(a) for a in alerts]
    lines += ["", "Prototype output. Not an official warning."]
    return "\n".join(lines)


def deliver(
    alerts: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    run_id: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Write alerts.json always. Optionally push to Telegram when configured."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "alerts.json"
    path.write_text(json.dumps(alerts, indent=2), encoding="utf-8")

    delivery = str(cfg.get("delivery", "file")).lower()
    status: dict[str, Any] = {"mode": delivery, "file": str(path), "sent": 0}
    if delivery == "file" or not alerts:
        return status

    if delivery == "telegram":
        token = os.environ.get(str(cfg.get("telegram_bot_token_env", "TELEGRAM_BOT_TOKEN")))
        chat_id = os.environ.get(str(cfg.get("telegram_chat_id_env", "TELEGRAM_CHAT_ID")))
        if not token or not chat_id:
            status["error"] = "telegram credentials missing; stayed in file mode"
            return status
        import requests

        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": format_telegram(alerts, run_id)},
            timeout=20,
        )
        status["sent"] = int(resp.ok)
        status["http_status"] = resp.status_code
        return status

    status["error"] = f"unknown delivery mode {delivery!r}; stayed in file mode"
    return status
