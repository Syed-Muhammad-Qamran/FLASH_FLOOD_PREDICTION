"""End-to-end pipeline runner and run-report generation.

Orchestrates: validated model outputs -> fusion -> tiers -> decisions ->
messages -> verification -> delivery, writing every artefact under
``runs/<run_id>/``. A missing or stale data source degrades the report; it never
aborts the run. Test-split rows are never read before M7.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import alerts, decisions, fusion, interop, report_schema, schema, thresholds
from .data_sources import health
from .models import windows

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_CONFIG = ROOT / "configs" / "run.yaml"
GRID_PATH = ROOT / "data" / "processed" / "grid_index.parquet"
SNAPSHOT_PATH = ROOT / "data" / "processed" / "SNAPSHOT.json"
PRED_A = ROOT / "artifacts" / "a" / "predictions_a.parquet"
PRED_B = ROOT / "artifacts" / "b" / "predictions_b.parquet"


def load_run_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_RUN_CONFIG)
    if not p.exists():
        return {
            "schema_version": 1,
            "delivery": "file",
            "language": "en",
            "dry_run": True,
            "cooldown_hours": 6,
            "audit_hours": 24,
        }
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("delivery", "file")
    data.setdefault("language", "en")
    data.setdefault("dry_run", True)
    data.setdefault("cooldown_hours", 6)
    data.setdefault("audit_hours", 24)
    return data


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "name") and type(value).__name__ == "Level":
        return value.name
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    return path


def _model_block(
    predictions: pd.DataFrame, meta: dict[str, str], module: str, path: Path
) -> dict[str, Any]:
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        # Synthetic/test runs may point outside the repository root.
        display_path = str(path)
    return {
        "module": module,
        "path": display_path,
        "rows": int(len(predictions)),
        "model_version": str(predictions["model_version"].iloc[0]) if len(predictions) else "",
        "data_snapshot_hash": meta.get("data_snapshot_hash", ""),
        "seed": meta.get("seed", ""),
    }


def _audit_records(
    fused: pd.DataFrame,
    *,
    hours: float,
    now: pd.Timestamp,
    cooldown_hours: float,
) -> tuple[list[dict[str, Any]], dict[str, decisions.AreaState]]:
    """Replay the state machine over the audit window, one record per timestep."""
    frame = fused.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    start = now - pd.Timedelta(hours=float(hours))
    frame = frame[(frame["timestamp_utc"] >= start) & (frame["timestamp_utc"] <= now)].sort_values(
        ["timestamp_utc", "basin_id"]
    )

    states: dict[str, decisions.AreaState] = {}
    records: list[dict[str, Any]] = []
    for timestamp, group in frame.groupby("timestamp_utc", sort=True):
        results = interop.to_risk_results(group, computed_at=pd.Timestamp(timestamp).isoformat())
        levels = {result.area_id: result.level for result in results}
        cycle, states = decisions.apply_decisions(
            states,
            levels,
            now=pd.Timestamp(timestamp).to_pydatetime(),
            cooldown_hours=cooldown_hours,
        )
        action_by_area = {decision.area_id: decision.action for decision in cycle}
        by_area = {result.area_id: result for result in results}
        for basin_id, row in group.set_index("basin_id").iterrows():
            result = by_area[basin_id]
            drivers = [
                driver["feature"] for driver in schema.parse_drivers(row.get("top_drivers", "[]"))
            ]
            records.append(
                {
                    "basin_id": str(basin_id),
                    "timestamp_utc": pd.Timestamp(timestamp).isoformat(),
                    "prob_a": round(float(row.get("prob_a", 0.0)), 6),
                    "prob_b": round(float(row.get("prob_b", 0.0)), 6),
                    "prob_fused": round(float(row["prob"]), 6),
                    "tier": result.tier,
                    "compound": bool(row.get("compound", False)),
                    "alert_decision": action_by_area.get(str(basin_id), "none"),
                    "drivers": drivers,
                }
            )
    return records, states


def run_pipeline(
    run_id: str = "demo",
    *,
    config_path: str | Path | None = None,
    offline: bool = True,
    dry_run: bool | None = None,
    persist_state: bool = False,
    lang: str | None = None,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the full pipeline and return a summary with output paths."""
    cfg = load_run_config(config_path)
    language = lang or str(cfg.get("language", "en"))
    is_dry_run = bool(cfg.get("dry_run", True)) if dry_run is None else dry_run
    cooldown_hours = float(cfg.get("cooldown_hours", 6))
    audit_hours = float(cfg.get("audit_hours", 24))

    run_dir = Path(run_root or (ROOT / "runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    grid = windows.read_grid_index(GRID_PATH)
    thresholds_config = thresholds.load_thresholds()
    thresholds.validate_thresholds(thresholds_config)

    snapshot_hash = str(json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["data_snapshot_hash"])

    pred_a, meta_a = fusion.load_model_file(PRED_A, schema.MODULE_A, grid)
    pred_b, meta_b = fusion.load_model_file(PRED_B, schema.MODULE_B, grid)
    fused = fusion.fuse_frames(
        pred_a,
        pred_b,
        thresholds_config=thresholds_config,
        meta_a=meta_a,
        meta_b=meta_b,
    )
    fused_path = run_dir / "predictions_fused.parquet"
    fused.to_parquet(fused_path, index=False)

    latest = fused.sort_values("timestamp_utc").groupby("basin_id", as_index=False).tail(1)
    as_of = cfg.get("as_of")
    if as_of:
        cutoff = pd.Timestamp(as_of)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        window = fused[fused["timestamp_utc"] <= cutoff].sort_values("timestamp_utc")
        latest = window.groupby("basin_id", as_index=False).tail(1)
    now = pd.Timestamp(latest["timestamp_utc"].max())
    results = interop.to_risk_results(latest, computed_at=now.isoformat())
    levels = {result.area_id: result.level for result in results}

    if persist_state:
        store = decisions.StateStore()
        states = store.load()
        cycle, updated = decisions.apply_decisions(
            states,
            levels,
            now=now.to_pydatetime(),
            cooldown_hours=cooldown_hours,
        )
        store.save(updated, updated_at=now.to_pydatetime())
    else:
        cycle, updated = decisions.apply_decisions(
            {},
            levels,
            now=now.to_pydatetime(),
            cooldown_hours=cooldown_hours,
        )

    outbound = alerts.build_outbound(results, cycle, lang=language)
    alert_rows = [
        decisions.decision_to_row(decision)
        for decision in cycle
        if decision.action in {"alert", "all_clear"}
    ]
    _write_json(run_dir / "alerts.json", alert_rows)
    _write_json(
        run_dir / "outbound.json",
        [message.as_dict() for message in outbound],
    )

    delivery = alerts.deliver(
        outbound,
        cfg=cfg,
        dry_run=is_dry_run,
        out_dir=str(run_dir),
        results=results,
    )
    verify_block = alerts.verify_summary(outbound, delivery)

    audit_rows, _ = _audit_records(
        fused,
        hours=audit_hours,
        now=now,
        cooldown_hours=cooldown_hours,
    )
    _write_jsonl(run_dir / "decisions.jsonl", audit_rows)

    source_health = health.read_source_health()
    stale = health.stale_sources(source_health)

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluated_at": now.isoformat(),
        "offline": bool(offline),
        "synthetic_data": True,
        "data_snapshot_hash": snapshot_hash,
        "thresholds": {
            "path": "configs/thresholds.yaml",
            "fitted_on": thresholds_config.get("fitted_on"),
            "fingerprint": thresholds_config.get("provenance_fingerprint"),
            "guard_pass": bool((thresholds_config.get("guard") or {}).get("pass", False)),
            "drift_breaches": thresholds.drift_breaches(thresholds_config),
        },
        "models": {
            "a": _model_block(pred_a, meta_a, schema.MODULE_A, PRED_A),
            "b": _model_block(pred_b, meta_b, schema.MODULE_B, PRED_B),
        },
        "fusion": {
            "rules": fusion.describe_fusion_rules(thresholds_config),
            "rows": int(len(fused)),
            "compound_rows": int(fused["compound"].sum()) if "compound" in fused else 0,
        },
        "decisions": {
            **decisions.action_counts(cycle),
            "rows": int(len(cycle)),
            "audit_rows": int(len(audit_rows)),
            "cooldown_hours": cooldown_hours,
            "min_level": "ORANGE",
            "persisted": bool(persist_state),
        },
        "sources": source_health,
        "delivery": delivery.as_dict(),
        "verify": verify_block,
        "areas": [
            {
                "area_id": result.area_id,
                "tier": result.tier,
                "level": result.level.name,
                "score": result.score,
                "compound": result.compound,
                "reasons": result.reasons,
            }
            for result in results
        ],
        "degraded": bool(stale),
    }
    report["sources"]["stale_sources"] = stale
    report_schema.validate_report(report)
    _write_json(run_dir / "run_report.json", report)

    return {
        "run_dir": str(run_dir),
        "predictions_fused": str(fused_path),
        "run_report": str(run_dir / "run_report.json"),
        "alerts": str(run_dir / "alerts.json"),
        "outbound": str(run_dir / "outbound.json"),
        "decisions_jsonl": str(run_dir / "decisions.jsonl"),
        "payload": delivery.payload_path,
        "report": report,
    }


__all__ = ["load_run_config", "run_pipeline"]
