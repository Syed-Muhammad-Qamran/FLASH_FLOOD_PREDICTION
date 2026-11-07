# FloodWatch KP

An **operational-grade prototype** for flash-flood early warning in Khyber
Pakhtunkhwa, Pakistan. Two mechanisms, two modules, one fused risk tier:

- **Model A — cloudburst / hill-torrent.** 3-hourly rainfall, soil wetness,
  seasonality. 6-hour horizon. Catches monsoon cloudbursts in Swat, Panjkora,
  Kohat Toi and the Gomal.
- **Model B — snow / glacier melt.** Degree-day melt, temperature, recent flow.
  24-hour horizon. Catches seasonal high flow in Chitral and the upper Indus.
- **Fusion.** Higher probability wins; if both are at *moderate* or above the
  event is flagged **compound** and the tier is escalated.

> Status: **research prototype, shadow mode.** It runs unattended, writes
> auditable runs, and never sends a public warning. See `docs/PLAN.md` for what
> has to be true before that changes.

---

## Quickstart (works offline, no credentials)

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.10+.

```powershell
cd floodwatch
uv sync --extra app --extra dev

uv run python scripts/01_make_basins.py      # placeholder basins.geojson
uv run python scripts/02_build_demo_data.py  # synthetic 2010-2025 dataset
uv run python scripts/03_train_baselines.py  # train + score on validation
uv run python scripts/04_evaluate.py         # validation report
uv run python scripts/06_run_demo.py         # replay run, Aug 2022

uv run streamlit run app/streamlit_app.py
```

The first two commands need no network. `02_build_demo_data.py` generates a
synthetic dataset so the entire pipeline can be exercised before a single
credential exists. **Every number derived from it is a pipeline test, not a
scientific result**, and the app says so on every page.

## Live run (real forecast, still no API key)

```powershell
uv run python scripts/07_run_live.py
```

This pulls 2 days of past + 3 days of forecast precipitation and temperature for
every basin centroid from Open-Meteo, runs the transparent rule engine, fuses,
writes a run directory, and appends alerts in shadow mode. It is a real
operational pass over real forecast data — the models are the only synthetic part
of the demo path.

## Tests

```powershell
uv run pytest
```

The suite covers the things that quietly break: split leakage, naive timestamps,
probability range, fusion staleness, compound escalation, lead-time accounting,
and an end-to-end run.

---

## What is real and what is not

| Piece | Demo path | Real path |
|---|---|---|
| Basins | buffered control points | HydroSHEDS / MERIT delineation |
| Rainfall | synthetic generator | IMERG Final (research) / Early (live) |
| Temperature, soil | synthetic generator | ERA5 / ERA5-Land |
| Discharge | synthetic generator | GloFAS v4 |
| Forecast | synthetic generator | Open-Meteo (live) |
| Labels | rainfall / discharge proxies | same proxies + **observed flood inventory** (missing) |
| Models | gradient-boosted baselines | same + CNN-LSTM / LSTM in weeks 4-6 |

The one gap that no amount of modelling closes is **observed flood labels**. The
prototype is honest about using proxies. Phase 3 in `docs/PLAN.md` is about
building the verification dataset, not adding layers to the network.

## Layout

```
configs/          split.yaml (single source of truth), region, basins, thresholds, models
src/floodwatch/
  schema.py       the prediction contract
  splits.py       chronological split with guard bands
  labels.py       proxy labels, train-only thresholds
  features.py     shared feature engineering
  baselines.py    gradient-boosted baselines
  fusion.py       max / noisy-or, compound escalation
  metrics.py      POD, FAR, CSI, PR-AUC, event scores, lead time, bootstrap
  evaluate.py     split-aware scoring
  calibrate.py    validation-only threshold proposals
  operational.py  Open-Meteo ingest + transparent rule engine
  runner.py       one command -> one auditable run directory
  alerts.py       shadow-mode alerts, optional Telegram
  synthetic.py    demo data generator
app/streamlit_app.py
scripts/          01..07 numbered pipeline
tests/
docs/             PLAN, label rules, interfaces, runbook, log, limitations
runs/             one directory per run (gitignored)
```

## The rules that make the results believable

1. **One split file.** `configs/split.yaml` is the only place years are defined.
2. **Train-only fitting.** Thresholds, scalers and tier cutoffs come from
   training years. The test split is scored once, and
   `scripts/04_evaluate.py --split test` refuses to run without `--confirm-test`.
3. **One schema.** Every module writes `basin_id, timestamp_utc, module, prob,
   tier, top_drivers`. Naive timestamps are a hard error.
4. **Every run is auditable.** `runs/<run_id>/` holds the predictions, alerts,
   and a report with data source, engine, threshold provenance and health checks.
5. **Shadow mode by default.** Delivery is a file, not a message.

## Disclaimer

This is a research prototype. It is not an official warning service and must not
be used as one. See `docs/limitations.md`.
