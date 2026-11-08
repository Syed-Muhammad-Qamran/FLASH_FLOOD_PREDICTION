"""FloodWatch KP — operational dashboard and offline demo viewer.

This module is import-safe in the lightweight runtime: it never imports torch or
scikit-learn. The Streamlit UI loads committed ``runs/demo`` artefacts by
default and degrades gracefully when a source is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Support `streamlit run app/streamlit_app.py` from a checkout when the
# package isn't installed in the active environment (src/ layout).
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from floodwatch import explain, interop, schema, thresholds, triage  # noqa: E402
from floodwatch.tiers import Level  # noqa: E402

RUNS_ROOT = ROOT / "runs"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None


def _discover_runs() -> list[str]:
    if not RUNS_ROOT.exists():
        return []
    runs = []
    for child in sorted(RUNS_ROOT.iterdir()):
        if child.is_dir() and (child / "run_report.json").exists():
            runs.append(child.name)
    if "demo" in runs:
        runs = ["demo"] + [name for name in runs if name != "demo"]
    return runs


def _latest_results(fused: pd.DataFrame) -> list[interop.RiskResult]:
    frame = schema.validate_predictions(fused)
    latest = frame.sort_values("timestamp_utc").groupby("basin_id", as_index=False).tail(1)
    as_of = str(pd.Timestamp(latest["timestamp_utc"].max()).isoformat())
    return interop.to_risk_results(latest, computed_at=as_of)


def _health_banner(report: dict[str, Any] | None) -> None:
    sources = (report or {}).get("sources", {}).get("sources", {})
    if not sources:
        st.warning("Source health unavailable; running in degraded demo mode.")
        return
    stale = [name for name, block in sources.items() if block.get("stale")]
    if stale:
        st.warning(
            "Degraded inputs: stale sources "
            + ", ".join(sorted(stale))
            + ". Risk estimates may rely on fallback or older data."
        )
    else:
        st.success("All monitored ingest sources are fresh.")
    cols = st.columns(max(1, len(sources)))
    for column, (name, block) in zip(cols, sorted(sources.items()), strict=False):
        lag = block.get("lag_hours")
        lag_text = "unknown" if lag is None else f"{lag:.1f} h lag"
        state = "stale" if block.get("stale") else "fresh"
        column.metric(name, state, lag_text)


def _overview(results: list[interop.RiskResult], thresholds_config: dict[str, Any]) -> None:
    table = interop.risk_frame(results)
    if table.empty:
        st.info("No risk rows are available for this run.")
        return
    display = table[
        [
            "area_id",
            "name",
            "district",
            "level",
            "tier",
            "score",
            "compound",
            "prob_a",
            "prob_b",
            "reasons",
        ]
    ].copy()
    display["score"] = display["score"].map(lambda value: f"{float(value):.3f}")
    st.dataframe(display, width="stretch", hide_index=True)

    cuts = thresholds_config.get("modules", {}).get(schema.MODULE_FUSED, {}).get("tiers", {})
    st.subheader("Why not higher?")
    for result in results:
        card = explain.explain(result, cuts)
        label = result.level.name
        with st.expander(
            f"{result.area_id} — {label} ({result.score:.3f})",
            expanded=result.level >= Level.ORANGE,
        ):
            st.write(f"Tier: `{result.tier}`")
            st.write(f"Gap to next tier: **{card.gap_to_next:.3f}** → `{card.next_tier}`")
            st.write(f"Binding constraint: {card.binding_constraint}")
            if result.compound:
                st.caption("Compound event: cloudburst and snowmelt mechanisms are both active.")
            st.caption("Drivers: " + "; ".join(result.reasons))


def _triage_tab(results: list[interop.RiskResult]) -> None:
    st.write(
        "Paste a report below. The verdict is deterministic: it checks monitored basin names, "
        "current risk levels and committed advisories. It never labels a message fake or true."
    )
    levels = {result.area_id: result.level for result in results}
    advisories = triage.load_advisories()
    text = st.text_area("Message text", height=160, placeholder="Example: Kalam is flooding now")
    if not st.button("Assess message"):
        return
    result = triage.triage(text, levels=levels, advisories=advisories)
    st.markdown(f"**Verdict:** `{result.verdict.value}`")
    if result.matched_basins:
        st.write("Matched basins: " + ", ".join(result.matched_basins))
    if result.extracted_date:
        st.write(f"Extracted date: {result.extracted_date.isoformat()}")
    if result.level is not None:
        st.write(f"Current level: {result.level.name}")
    for line in result.evidence:
        st.write("- " + line)


def _run_view(run_id: str) -> None:
    run_dir = RUNS_ROOT / run_id
    report = _load_json(run_dir / "run_report.json")
    st.caption(f"Run: `{run_id}` · offline replay of committed artefacts")
    if report and report.get("synthetic_data"):
        st.info("Demo data is synthetic and labelled as such in the run report.")

    _health_banner(report)

    fused_path = run_dir / "predictions_fused.parquet"
    if not fused_path.exists():
        st.error(f"Missing predictions_fused.parquet in {run_dir}")
        return
    fused = pd.read_parquet(fused_path)
    thresholds_config = thresholds.load_thresholds()
    thresholds.validate_thresholds(thresholds_config)
    results = _latest_results(fused)

    overview_tab, explain_tab, triage_panel = st.tabs(
        ["Basin overview", "Alerts & decisions", "Message triage"]
    )
    with overview_tab:
        _overview(results, thresholds_config)
    with explain_tab:
        alerts = _load_json(run_dir / "alerts.json") or []
        outbound = _load_json(run_dir / "outbound.json") or []
        left, right = st.columns(2)
        left.subheader("Alert decisions")
        left.json(alerts)
        right.subheader("Verified outbound messages")
        right.json(outbound)
        if report:
            st.subheader("Run report summary")
            st.json(
                {
                    "evaluated_at": report.get("evaluated_at"),
                    "degraded": report.get("degraded"),
                    "fusion": report.get("fusion"),
                    "decisions": report.get("decisions"),
                    "delivery": report.get("delivery"),
                    "verify": report.get("verify"),
                    "thresholds": report.get("thresholds"),
                }
            )
    with triage_panel:
        _triage_tab(results)


def main() -> None:
    st.set_page_config(page_title="FloodWatch KP", layout="wide")
    st.title("FloodWatch KP")
    st.caption(
        "Shadow-mode flood decision support for monitored KP basins. "
        "Alerts are verified before delivery; the committed demo runs fully offline."
    )
    runs = _discover_runs()
    if not runs:
        st.error(
            "No runs found under `runs/`. Generate one with `python scripts/06_run.py --run-id demo --offline`."
        )
        return
    selected = st.sidebar.selectbox("Run", runs, index=0)
    st.sidebar.caption("The demo uses synthetic data and committed thresholds.")
    _run_view(selected)


if __name__ == "__main__":
    main()
