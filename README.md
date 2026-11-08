# FloodWatch KP — Northern KP Flash-Flood Early Warning Prototype

Two independent mechanisms (A: cloudburst, B: snowmelt) score the same canonical
grid, fuse through validated Parquet files, and pass every outbound text through
a deterministic verifier before delivery. Shadow mode is the default.

> This repository is a 3-day prototype. Metrics on synthetic data are pipeline
> tests, never scientific results. See `docs/FEATURE_BACKLOG.md`.

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements-dev.txt
pip install -e .

make demo            # rebuild the committed offline demo run
make check           # ruff + offline tests
streamlit run app/streamlit_app.py
```

The Streamlit app installs from `requirements.txt` alone: no `torch`, no
`scikit-learn`.

## Layout and ownership

Every file has exactly one owner (`OWNERS.yaml`). The two models meet only at
`artifacts/{a,b}/predictions_*.parquet`; `fusion.py` never imports model code.
Frozen contracts are listed in `OWNERS.yaml` under `contract`.

## What is real and what is demo

- `data/demo/` and `src/floodwatch/demo_data.py` are a **synthetic, labelled**
  fixture used to prove the pipeline offline.
- `artifacts/a/predictions_a.parquet` in this repo is a deterministic
  placeholder produced by `scripts/00_build_demo.py` until the Model A lane
  publishes M2a. It is labelled synthetic in its manifest.
- Real evaluation is gated by M7 (`scripts/07_test_eval_once.py`); exactly one
  test-set run is allowed and the marker `runs/TEST_EVALUATED.json` enforces it.

## Safety model

- Alert at ORANGE and above only; one all-clear per drop below ORANGE.
- Every message is verified (`verify.check_risk_alert`) before it is written or sent.
- Delivery needs three locks: config, `FLOODWATCH_SEND=1`, and no operator
  override. `--dry-run` writes the exact payload and never opens a socket.
- Triage never says "fake" or "false"; it reports `no_signal_found` and cites
  the evidence it has.
