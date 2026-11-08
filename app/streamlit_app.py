"""FloodWatch KP — operational dashboard and offline demo viewer.

This module is import-safe in the lightweight runtime: it never imports torch or
scikit-learn. The Streamlit UI loads committed ``runs/demo`` artefacts by
default and degrades gracefully when a source is missing.
"""

from __future__ import annotations

import json
import subprocess
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
from floodwatch.data_sources import historical_events  # noqa: E402
from floodwatch.models import eval_cases  # noqa: E402
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


def _historical_cases_view() -> None:
    st.header("Historical Case Evaluation (Pakistan 2010 & 2022 Floods)")
    st.caption(
        "Evaluates Model A and Model B against external benchmark events. "
        "Source-confirmed historical impacts from official disaster authorities (NDMA/PDMA/FFC) "
        "are reported strictly separately from model predictions."
    )

    # 1. Hydrological & Reanalysis Provenance Notice
    with st.expander("Hydrological Provenance & Reanalysis Notice", expanded=False):
        st.markdown(
            """
            - **Gridded Reanalysis Weather (ERA5 / ERA5-Land):** Historical weather inputs are obtained from
              Open-Meteo ERA5 atmospheric reanalysis (~9–30 km resolution). Reanalysis represents gridded spatial
              averages, **NOT local in-situ river gauge stage or discharge measurements**.
            - **Approximate Basin Control Points:** Monitored basins are approximate outlet/river gauging points,
              **NOT delineated catchment polygons**. A district-wide disaster declaration does not automatically mean
              every sub-basin or tributary inside that district experienced overbank flooding.
            - **Out-of-Sample Isolation:** Both the 2010 Super Floods and 2022 Monsoon Floods are strictly
              excluded from training, validation tuning, and decision threshold selection.
            """
        )

    # 2. Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        case_choice = st.selectbox(
            "Historical Event",
            [
                "2022 Pakistan Monsoon Floods (Aug 25 - Aug 30, 2022)",
                "2010 Pakistan Super Floods (Jul 28 - Aug 3, 2010)",
            ],
            index=0,
            key="hist_case_choice",
        )
        case_year = 2022 if "2022" in case_choice else 2010

    with ctrl_col2:
        model_choice = st.selectbox(
            "Evaluation Model",
            [
                "Model A (Cloudburst / Convective Rain Risk)",
                "Model B (Snowmelt & Rain-on-Snow Risk)",
            ],
            index=0,
            key="hist_model_choice",
        )
        is_model_a = "Model A" in model_choice
        model_name = "Model A" if is_model_a else "Model B"

    with ctrl_col3:
        threshold = st.slider(
            "Risk Decision Cutoff", min_value=0.10, max_value=0.90, value=0.50, step=0.05
        )

    cases = historical_events.load_historical_cases()
    case = cases.get(case_year)
    if not case:
        st.error(f"Historical case definition missing for {case_year}.")
        return

    st.info(
        f"**{case.title}** · Mechanism: `{case.primary_mechanism.upper()}` · Window: `{case.start_utc}` to `{case.end_utc}`"
    )
    st.write(case.description)

    # Check for precomputed report in artifacts, or evaluate live
    art_dir = ROOT / "artifacts" / ("a" if is_model_a else "b")
    manifest_p = art_dir / "manifest_real.json"
    report_dict = None
    if manifest_p.exists():
        try:
            m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
            reports = m_data.get("case_reports", {})
            report_dict = reports.get(str(case_year)) or reports.get(case_year)
        except Exception:
            report_dict = None

    if report_dict is None:
        # Fallback to computing from available predictions parquet or demo
        pred_p = art_dir / f"predictions_{'a' if is_model_a else 'b'}.parquet"
        if not pred_p.exists():
            pred_p = ROOT / "runs" / "demo" / "predictions_fused.parquet"
        if pred_p.exists():
            df_p = pd.read_parquet(pred_p)
            if "prob" not in df_p.columns:
                df_p["prob"] = df_p["prob_a"] if is_model_a else df_p["prob_b"]
            rep = eval_cases.evaluate_case(
                case_year=case_year, predictions_df=df_p, model_name=model_name, threshold=threshold
            )
            report_dict = rep.to_dict()

    if report_dict is None:
        st.warning(
            f"No trained {model_name} predictions or real evaluation reports found yet. Run `python scripts/train_{'model_a' if is_model_a else 'model_b'}_real.py` to generate real event evaluations."
        )
        return

    # 3. Two-Column Separate Reporting: Source-Confirmed Impact vs Model Predictions
    st.subheader("Source-Confirmed Impacts vs. Model-Predicted Risk")
    st.caption(
        "Official ground truth is reported separately from model predictions. Model predictions are never presented as confirmed historical impacts."
    )

    col_truth, col_pred = st.columns(2)

    comparisons = report_dict.get("comparisons", [])
    truth_rows = []
    pred_rows = []

    for c in comparisons:
        truth_rows.append(
            {
                "Basin ID": c["basin_id"],
                "Official Observed Status": c["confirmed_status"].replace("_", " ").title(),
                "Spatial Granularity": c["spatial_granularity"].replace("_", " ").title(),
                "Official Citation": c["source_citation"],
            }
        )
        pred_rows.append(
            {
                "Basin ID": c["basin_id"],
                "Model Max Probability": f"{c['model_max_prob']:.3f}",
                "Predicted Risk (>= Cutoff)": "ALERT (Risk Detected)"
                if c["model_predicted_risk"]
                else "NORMAL (Below Cutoff)",
                "Risk Tier": c["model_tier"],
                "Comparison Status": c["comparison_outcome"],
            }
        )

    with col_truth:
        st.markdown("#### 🏛️ Official Source-Confirmed Impacts")
        st.dataframe(pd.DataFrame(truth_rows), hide_index=True, width="stretch")

    with col_pred:
        st.markdown(f"#### 🤖 {model_name} Predicted Risk")
        st.dataframe(pd.DataFrame(pred_rows), hide_index=True, width="stretch")

    # 4. Comparison & Agreement Metrics (where geography & labels support comparison)
    st.subheader("Comparison & Verification Metrics (Comparable Basins Only)")
    metrics = report_dict.get("metrics", {})
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Probability of Detection (POD)", f"{metrics.get('pod', 0.0):.3f}")
    m_col2.metric("False Alarm Ratio (FAR)", f"{metrics.get('far', 0.0):.3f}")
    m_col3.metric("Critical Success Index (CSI)", f"{metrics.get('csi', 0.0):.3f}")
    m_col4.metric("F1-Score", f"{metrics.get('f1', 0.0):.3f}")
    m_col5.metric("Brier Score", f"{metrics.get('brier_score', 0.0):.3f}")

    # Detailed Matching Matrix
    comp_df = pd.DataFrame(
        [
            {
                "Basin": c["basin_id"],
                "Confirmed Impact": c["confirmed_status"],
                "Model Alert": c["model_predicted_risk"],
                "Outcome Classification": c["comparison_outcome"],
                "Scientific / Spatial Caveat": c["caveat"],
            }
            for c in comparisons
        ]
    )
    st.dataframe(comp_df, hide_index=True, width="stretch")

    # 5. Citations & Attribution
    st.subheader("Official Citations & Reference Provenance")
    for cit in historical_events.list_citations():
        st.markdown(
            f"- **{cit['name']}** ({cit['date']}) — Granularity: `{cit['granularity']}`. Attribution: {cit['attribution']}. [URL]({cit['url']})"
        )


def _run_model_test(test_file: str, model_name: str) -> None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", test_file],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        st.error(f"{model_name} tests exceeded the 5-minute time limit.")
        return
    except OSError as exc:
        st.error(f"Could not start {model_name} tests: {exc}")
        return

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode == 0:
        st.success(f"{model_name} tests passed.")
    else:
        st.error(f"{model_name} tests failed. Check that development dependencies are installed.")
    if output:
        st.code(output, language="text")


def _model_test_view() -> None:
    st.caption(
        "Runs each model's unit and smoke tests separately. These checks do not measure "
        "accuracy on real flood events. Install requirements-dev.txt to run them."
    )
    model_a, model_b = st.columns(2)
    if model_a.button("Run Model A tests", key="run_model_a_tests"):
        with st.spinner("Running Model A tests..."):
            _run_model_test("tests/test_model_a.py", "Model A")
    if model_b.button("Run Model B tests", key="run_model_b_tests"):
        with st.spinner("Running Model B tests..."):
            _run_model_test("tests/test_model_b.py", "Model B")


def main() -> None:
    st.set_page_config(page_title="FloodWatch KP", layout="wide")
    st.title("FloodWatch KP")
    st.caption(
        "Shadow-mode flood decision support for monitored KP basins. "
        "Alerts are verified before delivery; the committed demo runs fully offline."
    )
    runs = _discover_runs()
    selected = st.sidebar.selectbox("Run", runs, index=0) if runs else None
    st.sidebar.caption("The demo uses synthetic data and committed thresholds.")

    dashboard_tab, historical_tab, model_tests_tab = st.tabs(
        ["Operational Dashboard", "Historical Case Testing (2010 & 2022)", "Model Unit Tests"]
    )
    with dashboard_tab:
        if selected is None:
            st.error(
                "No runs found under `runs/`. Generate one with "
                "`python scripts/06_run.py --run-id demo --offline`."
            )
        else:
            _run_view(selected)
    with historical_tab:
        _historical_cases_view()
    with model_tests_tab:
        _model_test_view()


if __name__ == "__main__":
    main()
