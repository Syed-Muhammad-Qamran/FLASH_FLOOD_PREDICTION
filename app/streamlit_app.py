"""FloodWatch demo app.

    uv run streamlit run app/streamlit_app.py

Reads the most recent (or selected) run directory. Never computes risk itself:
if the app and the pipeline disagree, the pipeline is right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from floodwatch.config import RUNS_DIR, basins as basin_config, thresholds_config  # noqa: E402
from floodwatch.schema import TIER_RANK, parse_drivers  # noqa: E402

TIER_COLORS = {
    "none": (110, 118, 129),
    "low": (250, 204, 21),
    "moderate": (249, 115, 22),
    "high": (220, 38, 38),
    "severe": (127, 29, 29),
}
TIER_LABEL = {
    "none": "No alert",
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
    "severe": "Severe",
}

st.set_page_config(page_title="FloodWatch KP", page_icon="🌊", layout="wide")


def list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    runs = [
        p
        for p in RUNS_DIR.iterdir()
        if p.is_dir() and (p / "predictions_fused.parquet").exists()
    ]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    fused = pd.read_parquet(run_dir / "predictions_fused.parquet")
    pred_a = pd.read_parquet(run_dir / "predictions_a.parquet")
    alerts_path = run_dir / "alerts.json"
    alerts = json.loads(alerts_path.read_text(encoding="utf-8")) if alerts_path.exists() else []
    report_path = run_dir / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return fused, pred_a, alerts, report


def basin_meta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "basin_id": b["basin_id"],
                "name": b["name"],
                "lon": b["lon"],
                "lat": b["lat"],
                "mechanism": b["primary_mechanism"],
                "snow_fraction": b["snow_fraction"],
            }
            for b in basin_config()
        ]
    )


def map_frame(fused_at_time: pd.DataFrame) -> pd.DataFrame:
    meta = basin_meta()
    df = fused_at_time.merge(meta, on="basin_id", how="left")
    df["color"] = df["tier"].map(lambda t: [*TIER_COLORS.get(str(t), (110, 118, 129)), 200])
    df["radius"] = df["prob"].clip(0.15, 1.0) * 26000
    return df


def draw_map(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No rows to map.")
        return
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.75,
        stroked=True,
        get_line_color=[255, 255, 255, 180],
        line_width_min_pixels=1,
    )
    view = pdk.ViewState(
        latitude=float(df["lat"].mean()),
        longitude=float(df["lon"].mean()),
        zoom=6.2,
        pitch=0,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            map_style=None,
            tooltip={
                "html": "<b>{name}</b><br/>tier: {tier}<br/>prob: {prob:.2f}<br/>{top_drivers}",
                "style": {"backgroundColor": "#1f2937", "color": "white"},
            },
        ),
        use_container_width=True,
    )


def risk_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk"] = out["tier"].map(lambda t: TIER_LABEL.get(str(t), str(t)))
    out["cause"] = out["top_drivers"].map(lambda d: "; ".join(parse_drivers(d)[:3]))
    out["prob"] = out["prob"].round(3)
    out = out.merge(basin_meta()[["basin_id", "name", "mechanism"]], on="basin_id", how="left")
    return out[["name", "basin_id", "mechanism", "risk", "prob", "cause"]].sort_values(
        "prob", ascending=False
    )


def main() -> None:
    runs = list_runs()
    if not runs:
        st.title("FloodWatch KP")
        st.error("No runs found.")
        st.code(
            "uv run python scripts/02_build_demo_data.py\n"
            "uv run python scripts/03_train_baselines.py\n"
            "uv run python scripts/06_run_demo.py",
            language="powershell",
        )
        return

    st.sidebar.title("FloodWatch KP")
    labels = [p.name for p in runs]
    choice = st.sidebar.selectbox("Run", labels, index=0)
    run_dir = runs[labels.index(choice)]

    fused, pred_a, alerts, report = load_run(run_dir)
    fused["timestamp_utc"] = pd.to_datetime(fused["timestamp_utc"], utc=True)
    times = sorted(fused["timestamp_utc"].unique())
    peak_time = fused.loc[fused["prob"].idxmax(), "timestamp_utc"]

    mode = report.get("mode", "unknown")
    engine = report.get("engine", "unknown")
    status = report.get("health", {}).get("status", "unknown")

    st.sidebar.caption(f"mode: {mode} · engine: {engine}")
    st.sidebar.metric("Run status", status.upper())

    if mode == "demo":
        st.warning(
            "Synthetic demo run. Every number on this page is a pipeline test, not a forecast.",
            icon="🧪",
        )
    st.caption(
        f"Run `{run_dir.name}` · created {report.get('created_utc', '?')} · "
        "Research prototype in shadow mode. Not an official warning."
    )

    tab_now, tab_replay, tab_alerts, tab_report, tab_about = st.tabs(
        ["Situation", "Replay", "Alerts", "Run report", "Method & caveats"]
    )

    with tab_now:
        latest_time = pd.Timestamp(times[-1])
        current = fused[fused["timestamp_utc"] == latest_time]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest step (UTC)", latest_time.strftime("%Y-%m-%d %H:%M"))
        c2.metric("Basins at high or above", int(current["tier"].isin(["high", "severe"]).sum()))
        c3.metric("Peak probability", f"{current['prob'].max():.2f}")
        c4.metric("Alerts this run", len(alerts))

        left, right = st.columns([1.25, 1])
        with left:
            draw_map(map_frame(current))
        with right:
            st.subheader("Risk by basin")
            st.dataframe(risk_table(current), use_container_width=True, hide_index=True)

        st.subheader("Probability over the run")
        pivot = fused.pivot_table(index="timestamp_utc", columns="basin_id", values="prob")
        st.line_chart(pivot)

    with tab_replay:
        st.subheader("Replay")
        st.caption("Step through the run to see how the two mechanisms build and decay.")
        default_index = times.index(peak_time) if peak_time in times else len(times) - 1
        idx = st.slider("Timestep", 0, len(times) - 1, default_index, format="index %d")
        t = pd.Timestamp(times[idx])
        st.write(f"**{t:%Y-%m-%d %H:%M UTC}**  ·  {t.tz_convert('Asia/Karachi'):%H:%M} PKT")
        frame = fused[fused["timestamp_utc"] == t]
        left, right = st.columns([1.25, 1])
        with left:
            draw_map(map_frame(frame))
        with right:
            st.dataframe(risk_table(frame), use_container_width=True, hide_index=True)

        st.subheader("Module contributions")
        a_at = pred_a.copy()
        a_at["timestamp_utc"] = pd.to_datetime(a_at["timestamp_utc"], utc=True)
        a_at = a_at[a_at["timestamp_utc"] == t][["basin_id", "prob"]].rename(
            columns={"prob": "cloudburst_A"}
        )
        b_at = frame[["basin_id", "prob", "tier"]].rename(columns={"prob": "fused"})
        comp = a_at.merge(b_at, on="basin_id", how="outer").sort_values("fused", ascending=False)
        st.dataframe(comp, use_container_width=True, hide_index=True)

    with tab_alerts:
        st.subheader("Alert log (shadow mode)")
        if not alerts:
            st.info("No basin reached the alert tier in this run.")
        else:
            for a in alerts:
                icon = "🔴" if a["tier"] == "severe" else "🟠"
                compound = " · COMPOUND" if a.get("compound") else ""
                with st.expander(
                    f"{icon} {a['tier'].upper()} · {a['basin_id']} · p={a['peak_prob']:.2f}{compound}"
                ):
                    st.write(f"First alert: `{a['first_alert_utc']}`")
                    st.write(f"Peak: `{a['peak_utc']}`")
                    st.write("Drivers:")
                    for d in a.get("drivers", []):
                        st.write(f"- {d}")
            st.download_button(
                "Download alerts.json",
                data=json.dumps(alerts, indent=2),
                file_name=f"alerts_{run_dir.name}.json",
                mime="application/json",
            )

    with tab_report:
        st.subheader("Run report")
        st.json(report)

    with tab_about:
        st.subheader("What this is")
        st.markdown(
            """
**Two mechanisms, two modules, one fused tier.**

| | Model A | Model B |
|---|---|---|
| Mechanism | monsoon cloudburst, hill-torrent flash flood | seasonal snow / glacier melt, high flow |
| Inputs | 3-hourly rainfall, soil wetness, seasonality | degree-day melt, temperature, recent flow |
| Horizon | 6 hours | 24 hours |
| Baseline | gradient-boosted trees | gradient-boosted trees |

**Fusion.** The higher probability wins; if both modules are at moderate or above,
the event is flagged *compound* and the tier is escalated one level.

**Split.** Train 2010–2021, validation 2022–2023, test 2024–2025, with a 30-day
guard band at each boundary. The test years are scored once.

**Caveats that matter.**
- Rainfall is satellite/model gridded. It under-resolves individual cloudburst cells.
- Model B's labels come from *modelled* GloFAS discharge, not gauges.
- Glacial lake outburst floods (GLOFs) are **not** predicted by either module.
- This is shadow mode: nothing here is sent to the public.
            """
        )


if __name__ == "__main__":
    main()
