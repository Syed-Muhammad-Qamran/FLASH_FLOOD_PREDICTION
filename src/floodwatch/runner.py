"""One command that produces a complete, auditable run.

    python -m floodwatch.runner --mode demo
    python -m floodwatch.runner --mode live

A run directory contains everything needed to reproduce and review the output:

    runs/<run_id>/
        predictions_a.parquet
        predictions_b.parquet
        predictions_fused.parquet
        alerts.json
        run_report.json
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .alerts import build_alerts, deliver
from .baselines import load_baseline
from .config import (
    RUNS_DIR,
    basin_ids,
    ensure_dirs,
    region,
    thresholds_config,
)
from .fusion import fuse
from .schema import parse_drivers, tier_for, validate_predictions
from .splits import describe as describe_splits

DEMO_START = "2022-08-01"
DEMO_END = "2022-09-16"


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    mode: str
    engine: str
    status: str
    n_alerts: int
    report: dict[str, Any]

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.run_id}: {self.status}, engine={self.engine}, alerts={self.n_alerts}"


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _drivers_a(table: pd.DataFrame) -> list[str]:
    rain = table["rain_mm_3h"].to_numpy(dtype=float)
    soil = table["soil_moisture"].to_numpy(dtype=float)
    return [
        json.dumps([f"rain_3h={rain[i]:.1f}mm", f"soil_moisture={soil[i]:.2f}"])
        for i in range(len(table))
    ]


def _drivers_b(table: pd.DataFrame) -> list[str]:
    melt = table["snowmelt_mm"].to_numpy(dtype=float)
    temp = table["temp_c"].to_numpy(dtype=float)
    return [
        json.dumps([f"melt={melt[i]:.2f}mm/3h", f"temp={temp[i]:.1f}C"])
        for i in range(len(table))
    ]


def _to_predictions(df: pd.DataFrame, module: str, cutoffs: dict[str, float]) -> pd.DataFrame:
    drivers = _drivers_a(df) if module == "A" else _drivers_b(df)
    out = pd.DataFrame(
        {
            "basin_id": df["basin_id"].to_numpy(),
            "timestamp_utc": pd.to_datetime(df["timestamp_utc"], utc=True),
            "module": module,
            "prob": np.clip(df["prob"].to_numpy(dtype=float), 0.0, 1.0),
            "tier": [tier_for(float(p), cutoffs) for p in df["prob"]],
            "top_drivers": drivers,
        }
    )
    return validate_predictions(out)


def _health(table: pd.DataFrame, preds: list[pd.DataFrame], now: pd.Timestamp, mode: str) -> dict[str, Any]:
    expected = set(basin_ids())
    observed = set(preds[0]["basin_id"].unique()) if preds else set()
    probs_ok = all(bool(p["prob"].between(0, 1).all()) for p in preds)
    no_nan = all(bool(p["prob"].notna().all()) for p in preds)

    checks: dict[str, Any] = {
        "rows": int(len(table)),
        "basin_coverage": sorted(expected - observed),
        "prob_range_ok": probs_ok,
        "no_nan_probs": no_nan,
        "has_rows": bool(len(table) > 0),
    }

    if mode == "live":
        latest = pd.to_datetime(table["timestamp_utc"], utc=True).max()
        age_hours = float((now - latest).total_seconds() / 3600.0)
        checks["data_age_hours"] = round(age_hours, 2)
        checks["freshness_ok"] = age_hours <= 9.0
    else:
        checks["freshness_ok"] = True
        checks["freshness_note"] = "demo mode: historical window, freshness not applicable"

    ok = (
        checks["has_rows"]
        and probs_ok
        and no_nan
        and not checks["basin_coverage"]
        and checks["freshness_ok"]
    )
    checks["status"] = "ok" if ok else "degraded"
    return checks


def run(
    mode: str = "demo",
    *,
    start: str | None = None,
    end: str | None = None,
    now: Any | None = None,
    run_id: str | None = None,
    use_models: bool = True,
    force_demo: bool = False,
) -> RunResult:
    if mode not in {"demo", "live"}:
        raise ValueError("mode must be 'demo' or 'live'")

    ensure_dirs()
    now_ts = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    run_id = run_id or f"{mode}_{now_ts:%Y%m%dT%H%M%SZ}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. inputs ----------
    if mode == "demo":
        from . import synthetic

        full = synthetic.load_or_generate(force=force_demo)
        table = synthetic.window(full, start or DEMO_START, end or DEMO_END)
        if table.empty:
            raise ValueError(f"demo window {start or DEMO_START}..{end or DEMO_END} is empty")
        source = {"kind": "synthetic", "note": "PIPELINE TEST ONLY - not real data"}
    else:
        from . import operational

        table = operational.fetch_all()
        source = operational.describe_source()

    table = table.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)

    # ---------- 2. inference ----------
    cfg = thresholds_config()
    cut_a = {k: float(v) for k, v in cfg["model_a"].get("tiers", {}).items()}
    cut_b = {k: float(v) for k, v in cfg["model_b"].get("tiers", {}).items()}

    engine = "rule_engine"
    pred_a = pred_b = None
    if use_models:
        try:
            base_a = load_baseline("A")
            base_b = load_baseline("B")
            pred_a = _to_predictions(base_a.predict(table), "A", cut_a)
            pred_b = _to_predictions(base_b.predict(table), "B", cut_b)
            engine = "baseline"
        except FileNotFoundError:
            pred_a = pred_b = None

    if pred_a is None or pred_b is None:
        from . import operational

        pred_a, pred_b = operational.rule_engine(table, cfg)
        engine = "rule_engine"

    # Keep only rows both modules can score, so fusion is not silently partial.
    common = pred_a[["basin_id", "timestamp_utc"]].merge(
        pred_b[["basin_id", "timestamp_utc"]], on=["basin_id", "timestamp_utc"]
    )
    pred_a = pred_a.merge(common, on=["basin_id", "timestamp_utc"])
    pred_b = pred_b.merge(common, on=["basin_id", "timestamp_utc"])

    # ---------- 3. fusion ----------
    fused = fuse(pred_a, pred_b, cfg)

    # ---------- 4. alerts ----------
    alerts = build_alerts(fused, min_tier=str(cfg["alerts"].get("min_tier", "high")))
    delivery = deliver(alerts, cfg["alerts"], run_id=run_id, out_dir=run_dir)

    # ---------- 5. write artefacts ----------
    pred_a.to_parquet(run_dir / "predictions_a.parquet", index=False)
    pred_b.to_parquet(run_dir / "predictions_b.parquet", index=False)
    fused.to_parquet(run_dir / "predictions_fused.parquet", index=False)

    health = _health(table, [pred_a, pred_b, fused], now_ts, mode)
    report = {
        "run_id": run_id,
        "mode": mode,
        "engine": engine,
        "created_utc": now_ts.isoformat(),
        "version": __version__,
        "python": platform.python_version(),
        "region": region()["name"],
        "source": source,
        "splits": describe_splits().to_dict(orient="records"),
        "threshold_provenance": {
            "model_a": cfg["model_a"].get("_provenance"),
            "model_b": cfg["model_b"].get("_provenance"),
        },
        "counts": {
            "rows": int(len(fused)),
            "basins": int(fused["basin_id"].nunique()),
            "timestamps": int(fused["timestamp_utc"].nunique()),
            "compound_rows": int(
                sum("compound" in d for d in fused["top_drivers"].astype(str))
            ),
            "alerts": len(alerts),
        },
        "health": health,
        "delivery": delivery,
        "alert_tiers": {
            t: int((fused["tier"] == t).sum())
            for t in ("none", "low", "moderate", "high", "severe")
        },
        "disclaimer": "Research prototype. Shadow mode. Not an official warning service.",
    }
    (run_dir / "run_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (RUNS_DIR / "latest.txt").write_text(run_id, encoding="utf-8")

    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        mode=mode,
        engine=engine,
        status=health["status"],
        n_alerts=len(alerts),
        report=report,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run one FloodWatch inference pass.")
    ap.add_argument("--mode", choices=["demo", "live"], default="demo")
    ap.add_argument("--start", default=None, help="demo window start (UTC)")
    ap.add_argument("--end", default=None, help="demo window end (UTC)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-models", action="store_true", help="force the rule engine")
    ap.add_argument("--force-demo", action="store_true", help="regenerate synthetic data")
    args = ap.parse_args(argv)

    result = run(
        mode=args.mode,
        start=args.start,
        end=args.end,
        run_id=args.run_id,
        use_models=not args.no_models,
        force_demo=args.force_demo,
    )
    print(f"run     : {result.run_id}")
    print(f"mode    : {result.mode}   engine: {result.engine}")
    print(f"status  : {result.status}")
    print(f"alerts  : {result.n_alerts}")
    print(f"output  : {result.run_dir}")
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
