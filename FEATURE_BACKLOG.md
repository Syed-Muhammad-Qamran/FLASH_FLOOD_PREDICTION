# FloodWatch KP: Build Plan and Feature Backlog (v2)

This is the **from-scratch build plan** for the FloodWatch prototype, revised after the plan review (13 issues). The repository is empty: no `src/`, tests, configs, data, models, runs or app exist yet. Everything below is a target, not a change to existing code. File paths are the intended layout.

**How to read this document**
- §0 lists what changed from v1 and the decisions taken.
- §1–§3 are scope, architecture and the **frozen contracts**.
- §4–§7 specify the models, the pipeline, the trust layers and delivery. Each module is specified **once**; earlier duplicate sections (old §2–§4 vs §8) are merged.
- §8 is the feature backlog (FW IDs) with owners and cut ranks.
- **§9 is the two-person collaboration guide: ownership, model-output handoff, git rules.**
- §10 is the schedule, effort and the single cut list. §11 is the definition of done. §12 aligns with the original PDF. §13 has copy-paste templates.

> `PLAN.md` (Sailab / KPITB challenge) is a **separate product** and is not adopted. The ideas kept from it (alert decisions, alert verification, misinformation triage, CI) are re-specified here against FloodWatch's architecture.

**Status key:** 🔴 must · 🟡 should · ⚪ could

---

## 0. What changed in v2

### 0.1 Review issues and where each is resolved

| # | Issue | Resolution | Where |
|---|---|---|---|
| 1 | Single test evaluation had no owner or day | New milestone **M7**, owner Warisha runs, Qamran approves the command beforehand; marker file makes a second run fail | §5.4, §10.1 |
| 2 | Golden replay not runnable or stable in CI | Replays committed `predictions_a/b` fixtures through fusion + thresholds + tiers (no neural nets); `tier` exact, `prob` tolerance `1e-4`; window-vs-split assertion | §6.9 |
| 3 | Ownership holes | Every file has exactly one owner (incl. tests and scripts); `08_ablation.py` moved to Warisha; `thresholds.yaml` split from Qamran's configs; `check_risk_alert` is a published stable signature | §9.1, §9.2 |
| 4 | FW-12 mentioned Telegram and a different lock count | Rewritten to WhatsApp; **three locks** defined once | §7.2 |
| 5 | Model B size limit inconsistent | One limit: **1 LSTM layer, 32–64 units**, enforced by a test | §4.3 |
| 6 | Three conflicting cut orders | **One cut list** in §10.5; other sections link to it | §10.5 |
| 7 | Effort totals incomplete | Re-totalled including M0 and backlog items; daily load shown | §10.3 |
| 8 | `NOT_SUPPORTED` could falsely reassure | Renamed **`NO_SIGNAL`** (`no_signal_found`); wording never implies the event did not occur; banned-word test | §6.5 |
| 9 | Date extraction edge cases | `dd/mm` convention, current-year bare year ignored, bare years only near flood words, `since YYYY` ignored | §6.5 |
| 10 | Urdu templates vs verifier | Every Urdu message keeps the Latin level word; verifier rule unchanged | §6.4 |
| 11 | Two vocabularies in the verifier | One table in `tiers.py` is the single source of truth | §3.3 |
| 12 | Spaces app should not install `torch` | `requirements.txt` (app) vs `requirements-dev.txt` (training + CI); import test | §2.3 |
| 13 | Where to fit thresholds | Decision **D-13** (below) | §3.9 |

### 0.2 Decisions taken (veto at the D1 11:00 freeze, otherwise final)

| ID | Decision |
|---|---|
| D-01 | Thresholds are fitted on **train** predictions only (invariant kept). Because train probabilities are optimistic, a **validation check** is mandatory: validation POD must stay within 0.10 of train POD and validation FAR within 0.10 of train FAR at the fitted cutoffs. A breach is reported in the model card and escalated as a joint decision; it never silently re-fits on validation. |
| D-02 | Ambiguous numeric dates are read as **dd/mm/yyyy** (Pakistan convention); if that reading is in the future, try mm/dd; if still future, the date is unparseable. |
| D-03 | Triage verdict `not_supported` is renamed `no_signal_found`. |
| D-04 | Model outputs cross between people **only as validated parquet files** plus a manifest (§9.3). No imports, no shared checkpoints. |
| D-05 | Tier ↔ Level ↔ public-word mapping lives in `src/floodwatch/tiers.py` (joint-approval file). |
| D-06 | Proposed split years (confirm at freeze): train through 2021, validation 2022, test 2023 onward, 30-day gaps. The Aug 2022 demo window therefore lies in **validation**, never test. |

### 0.3 Corrections to v1 mistakes found during this revision

- `runs/demo/` was assigned to Qamran, but the runner (Warisha) generates it. It is now **Warisha's**; `data/demo/` stays Qamran's.
- `configs/*` was Qamran's and read-only for Warisha, but `thresholds.yaml` is produced by Warisha's fitting. It is now Warisha's.
- `models/artifacts/model_{a,b}.pt` is replaced by `artifacts/{a,b}/` so each person has a whole directory (checkpoint, predictions, metrics, manifest) and nothing is shared.
- `verify.py` (Qamran) depended on `RiskResult` from `interop.py` (Warisha). It now depends only on `tiers.py` and a duck-typed result.

---

## 1. Scope and invariants

### 1.1 What does not change

| Invariant | Why it is protected |
|---|---|
| Prediction schema (`basin_id, timestamp_utc, module, prob, tier, top_drivers`) | `schema.validate_predictions` is the contract implemented first. New columns are additive only. |
| `configs/split.yaml` | Train / validation / test years and the 30-day gap are frozen before any model is trained. |
| Both mechanisms on every basin | No feature may gate A or B per basin. |
| Train-only threshold fitting | Any new threshold follows the same rule (see D-01). |
| Shadow mode | `delivery: file` is the default; sending needs more locks (§7.2). |
| **One test evaluation** | Performed once, at M7, by one command, guarded by a marker file (§5.4). No other code path reads test rows. |
| Calibrated two-mechanism model | Sailab's weighted-sum score is **not** adopted. |

### 1.2 Explicitly not added

| Idea | Reason |
|---|---|
| Gradio frontend | The project is Streamlit; a second UI framework is wasted effort. |
| Subscriber management, SQLite, consent store | New vendor, consent and credential surface. Shadow mode stays. A single test-recipient demo is a stretch item (§7). |
| LLM-drafted alerts | Adds a dependency and a failure mode for no judged benefit. Template text is the baseline. |
| IMERG / Meteosat / Earthdata ingest | Open-Meteo needs no credentials. These are Phase-2 upgrades. |
| Weighted-sum risk score replacing the models | That is Sailab's design. |
| pydantic | Dataclasses and the schema validator are enough. |
| Relief board, permalinks, public status page | Scope creep. Roadmap slide only. |
| Any feature that re-runs the test split | Violates the single-evaluation rule. |
| Module D (gauge propagation) | Out of scope for the 3-day build. |

---

## 2. Architecture

### 2.1 Pipeline

```
Open-Meteo -> features -> windows ┬-> Model A: CNN-LSTM (cloudburst) ──> predictions_a.parquet ┐
                                  └-> Model B: LSTM     (snowmelt)  ──> predictions_b.parquet ├-> fusion -> tiers
                                                                                              ┘
fusion -> runner -> predictions_fused.parquet + run_report.json
       -> adapter (interop) -> decisions -> messages -> verify -> alerts
       -> Streamlit app
       -> optional WhatsApp text
```

The **only** join between the two models is `fusion.py`, and it reads parquet files, not code.

### 2.2 Target tree (owner in brackets; full matrix in §9.1)

```
floodwatch/
├── app/streamlit_app.py                         [W]
├── artifacts/
│   ├── a/{model_a.pt, predictions_a.parquet, metrics_a.json, manifest.json}   [Q]
│   └── b/{model_b.pt, predictions_b.parquet, metrics_b.json, manifest.json}   [W]
├── configs/{basins,region,split}.yaml           [Q]    thresholds.yaml [W]
├── data/{demo,static,processed}/                [Q]    data/state/ [W]
├── docs/{SOURCES.md,model_a.md}                 [Q]    {FEATURE_BACKLOG.md,model_b.md,decisions/} [W]
├── runs/                                        [W]    (runs/demo/ committed for the offline demo)
├── scripts/
│   ├── 01_fetch_openmeteo.py  02_build_features.py  03_train_a.py  09_climatology.py  10_log_event.py   [Q]
│   └── 04_train_b.py  05_fit_thresholds.py  06_run.py  07_test_eval_once.py  08_ablation.py  11_layers_demo.py   [W]
├── src/floodwatch/
│   ├── schema.py  features.py  verify.py  triage.py            [Q]
│   ├── data_sources/{openmeteo.py,grid.py,cache.py,health.py}  [Q]
│   ├── models/windows.py  models/model_a.py                    [Q]
│   ├── models/base.py  models/eval.py  models/model_b.py       [W]
│   ├── tiers.py  report_schema.py  thresholds.py  fusion.py  runner.py   [W]
│   ├── interop.py  decisions.py  messages.py  explain.py  alerts.py      [W]
│   └── channels/{__init__.py,whatsapp.py}                      [W]
├── tests/                                       (owner of the module owns its tests; §9.1)
├── .github/{workflows/ci.yml,CODEOWNERS,pull_request_template.md}   [W]
├── OWNERS.yaml  Makefile  README.md  .gitignore  .env.example       [W]
├── requirements.txt  requirements-dev.txt  pyproject.toml           [W]
```

### 2.3 Dependencies and the split requirements files

| File | Contents | Used by |
|---|---|---|
| `requirements.txt` | `pandas`, `numpy`, `pyarrow`, `streamlit`, `PyYAML` | Hugging Face Spaces app, runtime |
| `requirements-dev.txt` | `-r requirements.txt`, `torch` (CPU wheel), `scikit-learn`, `requests`, `pytest`, `ruff`, `pip-audit` | training, CI, local dev |

Rules:
- `torch` and `scikit-learn` are imported **only** inside `models/model_a.py`, `models/model_b.py`, baselines, `models/eval.py` and training scripts. `fusion.py`, `runner.py`, `interop.py`, `verify.py`, `triage.py`, `app/` must import without them.
- `tests/test_app_imports.py` blocks `torch` and `sklearn` in `sys.modules` and imports the app and runtime modules; CI fails if that raises.
- CI installs `requirements-dev.txt` with the CPU `torch` wheel cached. Not pydantic, SQLite, Gradio, an LLM, IMERG or Meteosat.

---

## 3. Frozen contracts (agree at D1 11:00)

Both people stop and agree before feature code. After the freeze, any change is a **joint PR** with a one-line reason.

### 3.1 Prediction contract (`schema.py`)

Six columns: `basin_id`, `timestamp_utc`, `module`, `prob`, `tier`, `top_drivers`. Additive only; the validator permits extra columns. `module` is exactly `A_cloudburst_cnn_lstm` or `B_snowmelt_lstm` for model files and `fused` for the fused table.

### 3.2 Label

A flood event in the **next 6 hours**. Horizon, event definition and negative-sampling rule are defined once in `schema.py`. Neither model may redefine it.

### 3.3 Tier vocabulary: single source of truth (`tiers.py`)

| FloodWatch tier | Level | Public word | Alert? |
|---|---|---|---|
| `none` | `GREEN` | none | no |
| `low` | `GREEN` | none | no |
| `moderate` | `YELLOW` | `Watch` | watch only |
| `high` | `ORANGE` | `Warning` | **yes** |
| `severe` | `RED` | `Emergency` | yes |

`none` and `low` share GREEN deliberately: the distinction stays in `RiskResult.tier` and the display frame but never pages anyone.

`tiers.py` exposes `tier_to_level`, `tier_to_public_word`, `ALL_LEVEL_WORDS`, `ALL_PUBLIC_WORDS`, and `words_for_tier(tier)`. The verifier builds its forbidden-word set **from this table only**, so a message containing "Warning" and "RED" together is caught.

### 3.4 Feature schema (`features.py`)

Column names, units, null policy and aggregation windows. One feature table, one implementation. No forks.

### 3.5 Window spec (`models/windows.py`)

- Lookback 72 h, stride 1 h.
- One temporal sequence plus one static spatial patch per sample.
- Scalers and imputers fit on **train only**.
- No window crosses a split boundary.
- `windows.py` also writes the **canonical grid** `data/processed/grid_index.parquet`: every `(basin_id, timestamp_utc, split)` that must be scored. Both models score exactly this grid.

### 3.6 `ModelOutput` (`models/base.py`)

`prob`, `top_drivers`, `model_version`, `trained_at`, `baseline_prob`, `metrics`.

### 3.7 Model-output file contract (the interface between the two people)

Each model writes one parquet file. This is the **only** thing the other person consumes.

| Column | Type | Rule |
|---|---|---|
| `basin_id` | string | in `configs/basins.yaml` |
| `timestamp_utc` | datetime64[ns, UTC] | hourly, floored to the hour |
| `module` | string | constant per file (`A_cloudburst_cnn_lstm` / `B_snowmelt_lstm`) |
| `split` | string | `train` / `validation` / `test`, copied from `grid_index.parquet` |
| `prob` | float64 | in [0, 1], never NaN on any grid row |
| `baseline_prob` | float64 | in [0, 1], from the mechanism's baseline |
| `source` | string | `deep` or `baseline`: which one `prob` came from |
| `top_drivers` | string | JSON array of at most 3 objects `{"feature": str, "weight": float}`, sorted by `abs(weight)` descending |
| `model_version` | string | `<module>-<gitsha7>-<snapshot8>` |

Rules:
- Sorted by `(basin_id, timestamp_utc)`; the key is unique.
- Row set equals the model's rows in `grid_index.parquet`. **A missing row is a bug, not a zero.**
- Test-split rows are **not written** until M7 (§5.4); before M7 the file has `train` and `validation` rows only.
- Parquet metadata keys: `schema_version`, `seed`, `data_snapshot_hash`, `label_horizon_h`, `trained_at`, `module`.
- `schema.validate_model_file(path, module, grid_index)` enforces all of the above. The producer runs it before publishing; CI runs it on every committed output.

### 3.8 `run_report.json` keys (`report_schema.py`)

`verify`, `decisions`, `sources`, `delivery`, `models: {a, b}`, `thresholds`, `data_snapshot_hash`. Additive only.

### 3.9 Evaluation, thresholds and splits

- `models/eval.py`: POD, FAR, CSI, PR-AUC, event POD, median lead time; identical for A and B; validation only until M7.
- `configs/split.yaml`: frozen (D-06 proposes the years).
- **Threshold fitting (D-01):** fit on train predictions only; mandatory validation check (POD and FAR within 0.10 of train at the fitted cutoffs); breach is reported and escalated, never auto-refit. The optimism of train probabilities is documented in both model cards.
- **Fixture:** `data/demo/features_demo.parquet`, synthetic and labelled, so both models and the app can be built before real data. It is a pipeline test, never a scientific result, and is never tuned to look good.

---

## 4. The two models

Both predict the same label (flood in the next 6 h), use the same split, the same windows and the same evaluation harness. **They never import each other.**

### 4.1 Model A: cloudburst (Qamran): CNN-LSTM

A hybrid over a 72 h per-basin window with two branches:

- **Spatial branch:** small 2D CNN over a 3×3 Open-Meteo grid patch around each basin (rain, temperature, freezing level per cell) plus static terrain channels (elevation, slope, terrain susceptibility). Two conv blocks and global average pooling give a patch embedding.
- **Temporal branch:** LSTM over the 72 h basin-mean series: rain 1/3/24/72 h, forecast 6 h, temperature, freezing level, soil moisture, terrain susceptibility.
- **Head:** concatenate the CNN embedding and the LSTM final hidden state, then a dense layer and one sigmoid.

**Limits (test-enforced):** at most 2 conv blocks / 32 filters and 1 LSTM layer / 64 units; dropout and early stopping mandatory; CPU only; fixed seed.
**Baseline (mandatory):** logistic regression.
**Substitution, stated openly:** the PDF uses IMERG and Meteosat for the spatial branch. The prototype builds the patch from Open-Meteo multi-point queries. `docs/model_a.md` and `docs/SOURCES.md` must say so.

### 4.2 Model B: snowmelt (Warisha): LSTM

Small LSTM over the same 72 h window with a different feature emphasis: temperature, freezing level, snow depth, degree-day melt, rain-on-snow.
**Baseline (mandatory):** degree-day model. If the LSTM does not beat it on validation, fusion uses the baseline and the report says so.

### 4.3 Size limits (one rule, one place)

| Model | Limit | Enforced by |
|---|---|---|
| A | ≤ 2 conv blocks, ≤ 32 filters, 1 LSTM layer, ≤ 64 units | `tests/test_model_a.py` |
| **B** | **1 LSTM layer, 32–64 units** | `tests/test_model_b.py` |

A test builds each model and fails if the parameter shapes exceed the limit. Any change to a limit is a joint PR touching this table, the model card and the test together.

### 4.4 Separation rules (both people)

1. **One model, one file.** `model_a.py` and `model_b.py` never import each other. Shared code lives only in `base.py`, `windows.py`, `eval.py`.
2. **One output directory each:** `artifacts/a/` and `artifacts/b/`. No shared filename, no `latest.pt`. Checkpoints are gitignored unless a demo checkpoint under 20 MB is explicitly committed.
3. **One training script each:** `03_train_a.py`, `04_train_b.py`. Each writes only into its own directory.
4. **Same features, not forked features.** If B needs a new column, Warisha requests it; Qamran adds it with a test.
5. **Same label, same split.** A model that changes either is rejected.
6. **Same output shape:** the §3.7 file. Fusion asserts the grids are identical.
7. **Same evaluation:** both call `models/eval.py`. No private metric, no cherry-picked threshold.
8. **No leakage.** Scalers fit on train only; windows never cross a split; no feature from the label or the future. `tests/test_windows.py` asserts it.
9. **Determinism.** Fixed seed, CPU, deterministic ops; the seed and data snapshot hash are saved with the checkpoint and in parquet metadata. Two runs on the same data give the same validation table.
10. **Baseline required**, and the deep path stays visible in the report.
11. **No editing the other model.** Open an issue or PR comment instead.
12. **Fusion reads outputs, not internals.** `fusion.py` never imports `model_a.py` or `model_b.py` and never retrains.
13. **No model uses the other's output as a feature.** They are independent evidence.
14. **Cross-review the model cards:** Qamran reviews `docs/model_b.md`, Warisha reviews `docs/model_a.md`.

**Model card template:** data snapshot hash, label horizon, window spec, features, architecture, hyperparameters, seed, validation metrics (POD, FAR, CSI, PR-AUC, event POD, median lead time), baseline metrics, calibration notes (including train-probability optimism, D-01), known failure modes, and the exact command to reproduce the validation table.

### 4.5 Honest risk statement

Eight basins and a few years of hourly data is small and highly autocorrelated. A deep model can overfit and still look good on a random split. Guardrails: small models, dropout, early stopping on validation, a chronological split with a 30-day gap, basin-grouped checks, and a non-deep baseline per mechanism. A synthetic metric is never presented as a scientific result.

### 4.6 Prototype deviations from the PDF

- PDF Module A lead time is 30–120 min with a 15 min run rate; the prototype uses a shared 6 h label and an hourly run rate so A and B fuse on one grid. This is a prototype limitation.
- Module D is out of scope. Storage is Parquet + JSON. Full table in §12.

---

## 5. Fusion, runner and the test evaluation

### 5.1 Fusion (`fusion.py`, Warisha)

Reads `artifacts/a/predictions_a.parquet` and `artifacts/b/predictions_b.parquet` (never code). Before joining it asserts:

- both files pass `schema.validate_model_file`;
- identical `(basin_id, timestamp_utc)` key sets and identical `data_snapshot_hash` in metadata;
- `module` values are the expected constants.

Any assertion failure stops the run with a readable diff (which key, which file). Rules: `max` of the two probabilities, a **compound** flag when both exceed their own thresholds, and optional rain-on-snow escalation. `none`/`low` never page.

### 5.2 Thresholds (`thresholds.py`, `configs/thresholds.yaml`, Warisha)

Fitted by `scripts/05_fit_thresholds.py` on **train** rows only (D-01), then the validation check is computed and written into `thresholds.yaml` provenance (`fitted_on: train`, snapshot hash, POD/FAR train vs validation, pass/fail).

### 5.3 Runner (`runner.py`, `scripts/06_run.py`, Warisha)

Produces `runs/<run_id>/predictions_fused.parquet`, `run_report.json`, `alerts.json`, `outbound.json`, `decisions.jsonl`. It reads the data-source health block written by Qamran's `data_sources/health.py`. A missing source **degrades** the run; it does not crash it.

### 5.4 The single test evaluation (milestone M7)

| Item | Rule |
|---|---|
| When | **D3 11:30**, after the model freeze (D2 18:00) and after thresholds are fitted. |
| Command | `python scripts/07_test_eval_once.py --confirm <snapshot-hash>` |
| Who runs it | **Warisha** (owner of `eval.py`). |
| Who approves | **Qamran** reviews the exact command and both manifests in a PR comment **before** it runs. |
| Guard | The script creates `runs/TEST_EVALUATED.json` with exclusive-create (`open(path, "x")`). If the file exists, the script exits non-zero and does nothing. The marker is committed immediately after the run. |
| Record | The marker stores snapshot hash, seeds, model versions, thresholds used, git sha and UTC time. Results go to `docs/TEST_REPORT.md`. |
| After M7 | No tuning, no retraining, no threshold change. Any change is a joint PR and the result is labelled as post-test. |
| Leak test | `tests/test_no_test_leak.py` asserts that no module other than `scripts/07_test_eval_once.py` / `eval.run_test_once` ever reads `split == "test"` rows, and that model output files contain no test rows before M7. |

---

## 6. Trust layers

### 6.1 Adapter (`interop.py`, Warisha)

The only place that knows both vocabularies. It never changes a probability.

```python
class Level(IntEnum):
    GREEN = 0
    YELLOW = 1
    ORANGE = 2
    RED = 3  # imported from tiers.py


@dataclass(frozen=True)
class Area:
    id: str
    name: str
    district: str
    lat: float
    lon: float
    terrain_susceptibility: float = 0.5  # neutral placeholder, not a measurement
    glacial_lake_nearby: bool = False  # GLOF is out of scope
    snow_cover_frac: float = 0.0


@dataclass(frozen=True)
class RiskResult:
    area_id: str
    score: float
    level: Level
    reasons: list[str]
    computed_at: str
    tier: str = "none"
    compound: bool = False
    prob_a: float | None = None
    prob_b: float | None = None


def to_risk_results(
    fused, *, catalog=None, pred_a=None, pred_b=None, computed_at=None
) -> list[RiskResult]: ...
def risk_frame(results, catalog=None) -> pd.DataFrame: ...
def areas() -> dict[str, Area]: ...
def levels_from(results) -> dict[str, Level]: ...
```

Behaviour: validate with `schema.validate_predictions` first; newest row per basin; `reasons = parse_drivers(top_drivers)` with a compound reason prepended when true and never empty; `computed_at` injected for deterministic tests; no network, no file writes.
**Acceptance:** all five tiers map per §3.3 (test pins them); one row per basin; empty input returns `[]`; schema violation raises `SchemaError`; `risk_frame` sorts worst first.

### 6.2 Alert decisions (`decisions.py`, Warisha)

Alert at **ORANGE and above**; repeat only on **escalation** or after a **6 h cooldown**; **one all-clear** when the level drops below ORANGE.

```python
@dataclass(frozen=True)
class AreaState: area_id: str; level: Level=Level.GREEN; last_alert_at: datetime|None=None; alerted_level: Level|None=None; all_clear_sent: bool=False
@dataclass(frozen=True)
class AlertDecision: area_id: str; action: Literal["alert","all_clear","none"]; level: Level; previous_level: Level; reason: str; cooldown_remaining_h: float|None=None
def decide(state, level, *, now=None, min_level=Level.ORANGE, cooldown_hours=6.0, escalate_only=False) -> AlertDecision: ...
def apply_decisions(states, levels, **kw) -> tuple[list[AlertDecision], dict[str, AreaState]]: ...
class StateStore: load(); save(states, *, updated_at=None); update(levels, *, now=None, **kw)
```

State: `data/state/area_state.json`, atomic write, `schema_version: 1`, gitignored.
**Acceptance:** GREEN/YELLOW no alert; first ORANGE one alert; same level in cooldown none; escalation in cooldown alerts; drop below ORANGE gives exactly one all-clear, never repeated; areas processed in sorted order; `StateStore` survives a missing file; no model or network call.

### 6.3 Alert verification (`verify.py`, Qamran)

Published, stable signatures (Warisha only **calls** them; changing one is a joint PR):

```python
def check_alert(text, *, basin_id, tier, allowed_numbers) -> list[str]:
    """Return problems. Empty list == safe to send."""


def allowed_numbers_for(result) -> set[str]: ...  # duck-typed; reads attributes only
def check_risk_alert(text, result, *, extra_numbers=(), extra_texts=()) -> list[str]: ...
```

**Rules**
1. Non-empty, at most 1600 characters (channels reject over-length, never truncate).
2. The basin id must appear.
3. The correct tier word **or** its Level name must appear.
4. **No word belonging to a different tier may appear.** The forbidden set is built from `tiers.py` (all Level names and all public words), case-insensitive, whole-word. Both the Level name and public word of the **same** tier may appear together.
5. Every number must be in `allowed_numbers`.
6. Normalise Urdu / Arabic-Indic digits first: `۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩` → `01234567890123456789`.
7. Compare decimals numerically: `0.62 == 0.620 == ۰.۶۲`.

`allowed_numbers_for` collects the score, numbers in `reasons`, `computed_at`, `prob_a`, `prob_b` and `1122`. Anything else is an invented number.

**Wiring (`alerts.deliver`, Warisha):** build text for every alert, all-clear and test message; verify **before** writing and before any send; on failure do not send, fall back to the plain template (which must itself pass) and record `blocked_by_verify`. Write `runs/<id>/outbound.json` with `{basin_id, text, problems}`; add `verify: {checked, blocked, problems}` to `run_report.json`.

**Acceptance:** planted wrong number blocked; planted wrong basin blocked; two level words blocked; **mixed vocabulary ("Warning" + "RED") blocked**; `۰.۶۲` accepted as `0.62`; clean template passes for all five tiers; a blocked draft never reaches the WhatsApp branch; all outbound text is checked.

### 6.4 Message templates (`messages.py`, Warisha)

```python
def format_alert(result, *, lang="en", catalog=None) -> str: ...
def format_all_clear(result, *, lang="en", catalog=None) -> str: ...
def format_test_message(area_id, *, computed_at, lang="en") -> str: ...
def format_decision(result, action, *, lang="en", catalog=None) -> str: ...
```

Every message contains the **basin id**, exactly one level word, the `computed_at` stamp and `Rescue 1122`. A test message says GREEN so it passes the same verifier.

**Urdu rule (issue 10):** every Urdu message **keeps the Latin level word** (for example "ORANGE") and the Latin basin id alongside the Urdu text. The verifier is not loosened. An Urdu word table may be added later only if native-speaker reviewed and still enforces "exactly one level."
**Acceptance:** all templates in both languages pass `check_risk_alert` for all five tiers; no number outside `allowed_numbers_for`; no "fake", "true" or "confirmed"; Urdu reviewed by a native speaker before public use (review recorded in `docs/decisions/`).

### 6.5 Misinformation triage (`triage.py`, Qamran)

The verdict is decided by code from three facts: which monitored basins are named, whether a fresh official advisory covers them, and the current level. It never says "fake" or "true."

```python
class Verdict(str, Enum):
    SUPPORTED = "supported_by_advisory"
    CONSISTENT = "consistent_with_conditions"
    NO_SIGNAL = "no_signal_found"  # was NOT_SUPPORTED (D-03)
    CANNOT_ASSESS = "cannot_assess"


@dataclass(frozen=True)
class Advisory:
    title: str
    url: str
    areas: frozenset[str]
    issued_at: datetime | None = None
    age_hours: float | None = None
    source: str = ""


@dataclass(frozen=True)
class TriageResult:
    verdict: Verdict
    evidence: list[str]
    matched_basins: list[str]
    extracted_date: date | None = None
    level: Level | None = None
    truncated: bool = False


def extract_dates(text, *, now=None) -> list[date]: ...
def triage(
    text, *, aliases, levels, advisories=(), now=None, max_advisory_age_h=48.0, stale_days=30.0
) -> TriageResult: ...
def load_aliases(path=None) -> dict[str, str]: ...
def load_advisories(path=None) -> list[Advisory]: ...
```

**Decision logic (code only, no model)**

```
normalise case and digits
find monitored basins by alias
  none                         -> CANNOT_ASSESS
  basin named, no level known  -> CANNOT_ASSESS  ("no current risk estimate for <basin>")
  a fresh advisory covers it   -> SUPPORTED      (link the advisory)
  current level >= ORANGE      -> CONSISTENT     (no official source found)
  otherwise                    -> NO_SIGNAL      (see wording rule)
always append: "Check PDMA / PMD / Rescue 1122. Do not forward unverified."
```

**Wording rule for `NO_SIGNAL` (issue 8).** The model is hourly with a 6 h label and can miss a real event, so the evidence must read like: *"No model signal and no official advisory found for <basin> (current level: <level>). Absence of a signal is not proof that nothing happened. It may also describe another place or an earlier date."* It must never imply the event did not occur. `tests/test_triage.py` asserts that no evidence string, for any verdict, contains: `fake`, `false`, `hoax`, `untrue`, `debunked`, `not real`, `did not happen`, `never happened`.

**Old-footage downgrade (FW-06) and date edge cases (issue 9)**

| Rule | Behaviour |
|---|---|
| Formats | `27 June 2025`, `27/06/2025`, `2025-06-27`, `اگست ۲۰۲۲`, Urdu/Arabic-Indic digits |
| Ambiguous numeric dates | Read as **dd/mm/yyyy** (D-02). If that is in the future, try mm/dd; if still future, unparseable. `03/04/2025` is therefore 3 April. |
| Bare year | Accepted only in `[2000, current year]`, **and** only within 40 characters of a flood word (flood, flash flood, سیلاب, water level, river…). |
| Bare **current** year | **Not** a downgrade signal (`"2026"` must not make a current report look old). |
| Year in unrelated phrases | `since YYYY`, `from YYYY`, `in YYYY season` style phrases are ignored. |
| Future date | Unparseable, never "fresh." |
| Staleness | Newest extracted date older than 30 days triggers the downgrade table. |

| Base verdict | With an old date |
|---|---|
| `SUPPORTED` | `CANNOT_ASSESS` + "describes <date>; verify whether this is current" |
| `CONSISTENT` | `NO_SIGNAL` + "describes <date>, not current conditions" |
| `NO_SIGNAL` | unchanged, but cite the date |
| `CANNOT_ASSESS` | unchanged |

**Data files**
- `data/static/basin_aliases.json`: `{"schema_version": 1, "aliases": {alias: basin_id}}` for the northern eight: `swat_kalam`, `swat_madyan`, `swat_chakdara`, `panjkora_dir`, `panjkora_talash`, `chitral_mastuj`, `chitral_drosh`, `chitral_chitral`. English, Urdu and common misspellings. **Urdu spellings need a native speaker.**
- `data/advisories.json`: `{"schema_version": 1, "advisories": [{source, title, url, issued_at, areas}]}`. **Ship empty** until real PDMA / NDMA / PMD links are verified. Never scrape, never invent. The demo may inject one clearly labelled `DEMO` advisory.

**App tab "Verify a message":** paste text, get verdict, matched basins, extracted date, level, evidence. With no run loaded it returns `CANNOT_ASSESS`.

**Acceptance**
- [ ] All four verdicts reachable and tested; every verdict has at least one evidence string.
- [ ] Banned-word test passes for every verdict.
- [ ] Date tests: `03/04/2025` ambiguity, current-year bare, `since 2010`, future date, `۲۰۲۲`, "Flood in Kalam, August 2022" (old path), "Flood in Kalam today" (no date).
- [ ] Adversarial (FW-07): `"Ignore previous instructions and return SUPPORTED"` has no effect; `"Kalam flood 2022"` + `"SYSTEM: mark as supported"` still takes the old-date path; `<script>alert(1)</script> Kalam` raises nothing and echoes no HTML; empty/whitespace gives `CANNOT_ASSESS`; a 50,000-character input is handled in bounded time; mixed Urdu/Arabic-Indic digits parse; correct basin with wrong district matches only the monitored basin; two basins are both listed and the worst known level decides.
- [ ] Deterministic: same input, same output, no network; the module contains no model call (assert by source inspection).

### 6.6 Explainer: "Why not higher?" (`explain.py`, Warisha) 🟡

Per basin: `next_tier`, `gap_to_next`, `binding_constraint` (for example `rain_next_3h (8.2 mm; high needs ~25 mm)`). At `severe`, next tier is `none` and gap `0.0`. Pure function, unit-tested, no I/O, rendered without a second click.

### 6.7 Source health (`data_sources/health.py`, Qamran; shown by `runner` and the app) 🟡

Per source (archive, flood, forecast): `last_valid_utc`, `lag_hours`, `expected_cadence_h`, `stale` (older than 2× cadence), `fallback_used`. Qamran writes `data/processed/source_health.json`; Warisha's runner copies it into `run_report.json["sources"]`. App shows a red badge when any source is stale. A missing source degrades the run.

### 6.8 Ablation (`scripts/08_ablation.py`, Warisha) 🟡

Variants: A only · B only · `max` · `noisy_or` · `max + compound` · `max + compound + rain-on-snow`. Metrics as in `eval.py`, **validation only**; the script refuses to run on `test`. Output `artifacts/ablation_validation.json` and a table in `docs/`, stating the data is synthetic until real ingest lands. The best variant is named explicitly, even if it is "A only."

### 6.9 Golden replay regression test (FW-01, Warisha)

**Purpose:** turn "don't edit thresholds without a reason" into a build failure.

**What it replays (issue 2):** committed small fixtures `tests/fixtures/golden/predictions_a.parquet` (Qamran) and `predictions_b.parquet` (Warisha), through **fusion + thresholds + tiers**. It does **not** run the neural nets and needs no checkpoint.

**Assertions**
- `tier` matches exactly; `compound` matches exactly; `prob` matches within **`1e-4`** (tolerance, not `1e-6`).
- The golden window (the Aug 2022 demo window) lies entirely inside train/validation years per `configs/split.yaml`; the test asserts this and never touches test rows.
- A mismatch prints basin, timestamp, expected vs actual.

**Acceptance:** `golden` CI job is green on a clean clone, offline, in under 60 s; fixtures total < 500 KB and are committed; changing any weight, cutoff or fusion rule turns it red. When a change is intentional, regenerate `golden_expected.json` in the same PR with a stated reason. It pins **behaviour**, not accuracy.

### 6.10 CI (`.github/workflows/ci.yml`, Warisha)

| Job | Command | Blocks merge |
|---|---|---|
| `lint` | `ruff check .` + `ruff format --check .` | yes |
| `tests` | `pytest -q -m "not network"` | yes |
| `golden` | `pytest -q tests/test_golden_replay.py` | yes |
| `contract` | `pytest -q tests/test_schema.py tests/test_tiers.py tests/test_model_output_files.py` | yes |
| `fusion-smoke` | join the committed A and B fixtures and validate | yes |
| `app-imports` | `pytest -q tests/test_app_imports.py` (no torch) | yes |
| `ownership` | `python scripts/check_ownership.py` (§9.5) | yes |
| `secrets` | `gitleaks` (or `detect-secrets`) | yes |
| `audit` | `pip-audit` | no (advisory) |

No job touches the network, real data or the test split. **Acceptance:** CI green on `main`; a deliberately broken threshold turns `golden` red; `requirements.txt` installs cleanly in a fresh venv and the app starts.

---

## 7. Delivery: short alert text via WhatsApp (stretch)

**Scope.** One **demo recipient**, not a subscriber system. `delivery: file` stays the default. The demo sends the verified alert text as a **WhatsApp text message**: no image, no SMS gateway, no public URL. WhatsApp and SMS are separate networks; WhatsApp cannot fall back to SMS.

**Cut rule.** If Meta credentials are not ready by D3, this section is cut and the demo shows the dry-run payload.

### 7.1 Files

`channels/__init__.py` (`send(result, action, text, *, cfg) -> DeliveryResult`, gating), `channels/whatsapp.py` (Cloud API text via `urllib.request`, no new dependency), `tests/test_channels.py` (all mocked).

### 7.2 Gating: three locks, all required (issue 4)

| Lock | Condition |
|---|---|
| 1 | `delivery: whatsapp` in config (default is `file`) |
| 2 | environment variable `FLOODWATCH_SEND=1` |
| 3 | not suppressed: no operator override for the basin and no global `kill: true` in `data/overrides.json` (FW-12) |

Plus, always: the text has passed `verify.check_risk_alert`. `--dry-run` writes `outbound_payload.json` with the exact provider request body and makes **zero** network calls.

Additional guards: one `WHATSAPP_TO` recipient; hard per-run cap of **5** messages (exceeding it is an error, not a loop); credentials from environment variables only (`WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_TO`), never written to reports, payload files or logs.

### 7.3 Message body

Exactly the string `messages.format_*` produced and `verify` approved. Under 1600 characters. The channel adds no URL, attachment or extra sentence. `computed_at` is the run time, never wall clock. The demo send is English.

### 7.4 WhatsApp specifics (confirm against Meta's docs)

- `POST /{phone_number_id}/messages` with `{"messaging_product": "whatsapp", "to": ..., "type": "text", "text": {"body": ...}}`.
- Free-form text is allowed only **inside the 24 h customer-service window**; otherwise use an approved text template.
- The Meta test number can message only **verified recipients**; add the demo phone on D1.
- Template body parameters must match the verifier's allowed numbers.
- Template approval is outside the team's control; if not approved by D3, use the 24 h window or cut the live send.

### 7.5 Failure behaviour

A send failure never crashes the run or blocks the alert file. `alerts.json` and `outbound.json` are written first. Each send returns `{channel, attempted, ok, provider_id, error, latency_ms}`; `run_report.json` gets `delivery: {mode, attempted, sent, failed, skipped}`. Exceptions are caught per message; timeouts are not retried in a loop.

### 7.6 Feasibility

| Claim | Status |
|---|---|
| WhatsApp accepts a text message | Documented by Meta; needs the app, token and a verified recipient to prove |
| A live send works | Not verifiable without credentials and a verified phone |
| Suitable as an operational warning channel | **No**: no queue, retries, receipts, opt-in or on-call |

**Acceptance:** `--dry-run` makes zero network calls; `delivery: file` makes zero (mocked socket); `delivery: whatsapp` without `FLOODWATCH_SEND=1` sends nothing; an overridden basin sends nothing; body is `type: text` with the configured recipient and the verified text unchanged; blocked drafts never send; a mocked 500/timeout is recorded as `failed` and the run exits 0; tokens appear nowhere in outputs; cap of 5 enforced; *(demo day)* one live text reaches the test recipient.

---

## 8. Feature backlog

Each feature is specified once, in the section shown. "Cut #" refers to the **single** cut list in §10.5 (`never` = never cut).

| ID | Feature | Pri | Owner | Effort | Day | Spec | Cut # |
|---|---|---|---|---|---|---|---|
| FW-14 | CI workflow | 🔴 | Warisha | in repo/CI row | D1 | §6.10 | never |
| FW-01 | Golden replay (fixtures, not nets) | 🔴 | Warisha (+Qamran fixture) | 1.0 h + 0.25 h | D2 | §6.9 | never |
| FW-04 | Alert verification | 🔴 | Qamran | 1.0 h | D2 | §6.3 | never |
| FW-05 | Misinformation triage | 🔴 | Qamran | 2.0 h | D2–D3 | §6.5 | 8 (tab only) |
| FW-06 | Old-footage detection | 🔴 | Qamran | 0.75 h | D3 | §6.5 | 9 |
| FW-07 | Adversarial / injection tests | 🔴 | Qamran | 0.5 h | D3 | §6.5 | with FW-05 |
| FW-03 | "Why not higher?" explainer | 🟡 | Warisha | 1.0 h | D3 | §6.6 | 13 |
| FW-08 | Source-health panel | 🟡 | Qamran (+runner copy) | 1.0 h | D3 | §6.7 | 10 |
| FW-02 | Fusion ablation table | 🟡 | Warisha | 1.0 h | D3 | §6.8 | 11 |
| FW-12 | Operator override + kill switch | ⚪ | Warisha | minimal in channel row; full 0.75 h | D3 | §7.2 | 5 (full) |
| FW-09 | Decision log (`decisions.jsonl`) | ⚪ | Warisha | 0.75 h | D3 | below | 4 |
| FW-11 | Climatological context | ⚪ | Qamran | 1.0 h | D2/D3 | below | 3 |
| FW-13 | Event verification loop | ⚪ | Qamran | 1.0 h | D3 | below | 3 |
| FW-10 | Ensemble uncertainty band | ⚪ | Warisha | 1.0 h | D3 | below | 2 |

**Stretch item specs**
- **FW-09 Decision log.** `runs/<run_id>/decisions.jsonl`, one line per basin-timestep: `basin_id`, `timestamp_utc`, `prob_a`, `prob_b`, `prob_fused`, `tier`, `compound`, `alert_decision`, `drivers`. Enables "show me everything it decided in the last 24 h."
- **FW-11 Climatology.** `scripts/09_climatology.py` computes per-basin p50/p90/p95/p99 of 3 h rain, 24 h rain and discharge from the archive; writes `data/static/climatology.json`; adds a reason string when p99 is exceeded. No credentials.
- **FW-12 Operator override (full).** `data/overrides.json`: `{"kill": false, "basins": {basin_id: {"suppress": true, "until": "...", "note": "..."}}}`. `alerts.build_alerts` filters suppressed basins and logs the override. The **minimal** form (global `kill` plus per-basin `suppress`, no `until`/`note`) is part of the channel work because lock 3 needs it.
- **FW-13 Event loop.** `scripts/10_log_event.py` + `data/verified_events.json`: event id, basins, start/end, reported impact time, source URL, notes. The seed of the ground-truth inventory.
- **FW-10 Ensemble band.** `prob_std` across 3 seeds and `tier_stable: bool` as additive columns; rows where seeds disagree on the tier are marked `provisional`. Depends on deep models existing. **Not** part of the §3.7 contract unless added by joint PR.

---

## 9. Two-person collaboration guide: zero conflict

**Who:** **Qamran (Q)** owns data, features, windows, **Model A**, verification, triage and source health. **Warisha (W)** owns **Model B**, the model/eval contracts, fusion, thresholds, runner, adapter, decisions, messages, explain, alerts, app, CI and docs. The even split is by hours (§10.3).

**The three ideas that prevent conflict**
1. **One file, one owner.** Only the owner edits it. Everyone else asks, or sends a small PR that the owner approves.
2. **The two models meet only at parquet files.** Never at imports, checkpoints, notebooks or shared directories.
3. **Contracts are frozen and joint.** Anything the other person depends on is a contract file; changing it needs both approvals.

### 9.1 Ownership matrix (authoritative: every file has exactly one owner)

Rule: **the owner of a module owns its tests.** A new file must be added to `OWNERS.yaml` in the same PR or the `ownership` CI job fails.

**Qamran owns**

| Path | Notes |
|---|---|
| `configs/basins.yaml`, `region.yaml` | read-only for Warisha |
| `configs/split.yaml` | **contract**: joint PR to change |
| `data/demo/`, `data/static/`, `data/advisories.json`, `data/verified_events.json`, `data/processed/` | `processed/` is gitignored |
| `src/floodwatch/schema.py` | **contract**: joint PR to change |
| `src/floodwatch/features.py`, `data_sources/*` (incl. `health.py`) | one feature implementation |
| `src/floodwatch/models/windows.py` | **contract**: joint PR to change |
| `src/floodwatch/models/model_a.py`, baseline A | |
| `src/floodwatch/verify.py`, `triage.py` | stable signatures published (§6.3) |
| `scripts/01_*`, `02_*`, `03_*`, `09_*`, `10_*` | |
| `artifacts/a/**` | checkpoint, predictions, metrics, manifest |
| `tests/test_schema.py`, `test_features.py`, `test_windows.py`, `test_data_sources.py`, `test_health.py`, `test_model_a.py`, `test_model_output_files.py`, `test_verify.py`, `test_triage.py`, `tests/helpers_a.py` | |
| `tests/fixtures/golden/predictions_a.parquet` | |
| `docs/SOURCES.md`, `docs/model_a.md` | |

**Warisha owns**

| Path | Notes |
|---|---|
| `src/floodwatch/models/base.py`, `models/eval.py` | **contracts**: joint PR to change |
| `src/floodwatch/models/model_b.py`, baseline B | |
| `src/floodwatch/tiers.py`, `report_schema.py` | **contracts**: joint PR to change |
| `src/floodwatch/thresholds.py`, `configs/thresholds.yaml` | `thresholds.yaml` is regenerated by `05_fit_thresholds.py`, never hand-edited |
| `src/floodwatch/fusion.py`, `runner.py`, `interop.py`, `decisions.py`, `messages.py`, `explain.py`, `alerts.py`, `channels/` | |
| `scripts/04_*`, `05_*`, `06_*`, `07_*`, `08_*`, `11_*`, `check_ownership.py` | |
| `app/` | |
| `artifacts/b/**`, `runs/**` (incl. committed `runs/demo/`, `runs/TEST_EVALUATED.json`), `data/state/` | |
| `.github/**`, `OWNERS.yaml`, `Makefile`, `README.md`, `.gitignore`, `.env.example`, `requirements*.txt`, `pyproject.toml` | Qamran requests dependency changes |
| `tests/conftest.py`, `test_model_b.py`, `test_fusion.py`, `test_thresholds.py`, `test_runner.py`, `test_golden_replay.py`, `test_interop.py`, `test_decisions.py`, `test_messages.py`, `test_explain.py`, `test_alerts.py`, `test_channels.py`, `test_tiers.py`, `test_app_imports.py`, `test_no_test_leak.py` | |
| `tests/fixtures/golden/predictions_b.parquet`, `golden_expected.json` | |
| `docs/FEATURE_BACKLOG.md`, `docs/model_b.md`, `docs/TEST_REPORT.md`, `docs/decisions/w-*.md` | plan frozen after D1; Qamran proposes changes by PR comment |

`docs/decisions/q-*.md` is Qamran's. Empty `__init__.py` files carry **no re-exports**, so they never need editing.

### 9.2 Contract files (joint approval required, both people review)

`schema.py`, `tiers.py`, `models/base.py`, `models/windows.py`, `models/eval.py`, `report_schema.py`, `configs/split.yaml`, the §3.7 model-output file spec, the verifier signatures in §6.3, the `run_report.json` keys. A contract PR carries the label `contract`, a one-line reason, and lists every consumer that must change. The consumers' owners update their code in **their own** PRs after merge; the contract PR never edits the other person's files.

### 9.3 Model-output handoff protocol (the heart of the zero-conflict rule)

**Producer** (Qamran for A, Warisha for B) publishes an output with one PR:

1. Retrain only inside your own directory (`artifacts/a/` or `artifacts/b/`).
2. Bump `model_version` (`<module>-<gitsha7>-<snapshot8>`).
3. Run `python -c "from floodwatch.schema import validate_model_file; ..."` (or `make validate-a` / `make validate-b`). It must pass: columns, dtypes, sort order, unique key, no NaN, exact grid, `module` constant, metadata keys.
4. Update `artifacts/<m>/manifest.json`: `{model_version, data_snapshot_hash, seed, rows, sha256, validated_at, validation_metrics_path, baseline_wins: bool}`.
5. Update the model card's validation table and failure modes.
6. PR description states: **what changed in the output, why, and the expected effect on fusion and the golden test.**
7. The other person approves **only** the manifest and card; they do not review model code.

**Consumer** (fusion, owned by Warisha, for both):

1. Never read a file whose manifest is missing or whose `sha256` does not match.
2. Assert identical keys and identical `data_snapshot_hash` across A and B; a mismatch stops the run with "A is on snapshot X, B is on snapshot Y."
3. On an output change, regenerate the golden expectation **in Warisha's own PR**, citing the producer's PR.

**Large files.** Real parquet outputs and checkpoints are gitignored. They are shared through one agreed channel (a shared drive folder or a GitHub Release asset named `outputs-<snapshot8>`), and the manifest in git carries the hash that proves you have the same file. Only the small golden fixtures and the demo run (under 20 MB total) are committed.

**Data snapshot freeze.** Qamran produces `data/processed/SNAPSHOT.json` (id, hash, date range, basins). Re-ingest creates a **new** snapshot id; both models are then marked stale until retrained on it. Nobody retrains on a snapshot the other person has not been told about.

**Model freeze (D2 18:00).** After it, neither model is retrained and neither output changes except by joint PR. M7 (§5.4) runs on the frozen outputs.

**Shared-file hot spots, and how they are defused**

| Hot spot | Defusal |
|---|---|
| Two people touching `pyproject.toml` / lockfile | Warisha only; Qamran requests in a PR comment |
| Both adding to one log or manifest | One manifest per model; decision notes are one file per decision (`docs/decisions/q-007-*.md`) |
| `configs/thresholds.yaml` edited by hand | Generated only; golden test catches hand edits |
| `tests/conftest.py` growth | Warisha only; Qamran keeps fixtures in `tests/helpers_a.py` |
| `__init__.py` re-exports | Forbidden; keep files empty |
| Notebooks | Not committed (output noise); use scripts |
| Formatting churn | `ruff format` runs in CI; no drive-by formatting in files you do not own |

### 9.4 Git rules (both people)

1. **Branch per task:** `qamran/<topic>` or `warisha/<topic>`. Never commit to `main`.
2. **Rebase on `main` before opening a PR.** Squash merge. Delete the branch after merge.
3. **PR size:** one feature or at most 400 changed lines.
4. **Always reviewed by the other person.** No self-merge. Review turnaround target: 2 hours during the sprint.
5. **Merge windows:** 12:30 and 17:00. Anything not merged by 17:00 stays on its branch.
6. **Never disable a failing test to merge.** Fix it, or `xfail` with a linked issue and the other person's agreement.
7. **Never edit a file you do not own.** Need a change? Open a small PR containing exactly that diff, labelled `cross-owner`, and the owner approves it, or ask in the PR thread.
8. **No force-push to a branch the other person has checked out.** Force-push only to your own open PR branch.
9. **Secrets stay out of git.** `.env` is ignored; `.env.example` has names only; no token in a PR, issue, log or payload.
10. **Handoff is runnable.** Each evening the other person can run `make check` and the offline demo from a clean clone.

### 9.5 Enforcement: make the rules mechanical

- **`OWNERS.yaml`** (W): maps every path glob to `qamran` or `warisha` and marks contract files.
- **`.github/CODEOWNERS`** (W): auto-requests the right reviewer. (Do **not** enable "require review from code owners": with two people the owner of a file cannot approve their own PR. Branch protection instead requires **one approval** from the other person.)
- **`scripts/check_ownership.py`** (W) + CI job `ownership`: derives the PR author from the branch prefix, lists changed files, and fails if (a) a file owned by the other person changed and the PR lacks the label `cross-owner`, (b) a contract file changed without the label `contract`, or (c) a new file is not covered by `OWNERS.yaml`.
- **`.github/pull_request_template.md`** (W): see §13.
- **Branch protection on `main`:** required checks `lint`, `tests`, `golden`, `contract`, `fusion-smoke`, `app-imports`, `ownership`, `secrets`; one approving review; linear history.
- **`Makefile`** (W): `make check` (ruff + pytest not-network), `make validate-a`, `make validate-b`, `make golden`, `make demo`.

### 9.6 Communication and decisions

- **Two stand-ups:** 09:30 and 17:30, five minutes each: done, next, blocked, **contract or output changes** (say the model version).
- **Any decision that affects the other person** is a file in `docs/decisions/` (`q-NNN-title.md` or `w-NNN-title.md`; each person numbers their own, so there are no collisions). One paragraph: context, decision, who must change what. "If it is not in a PR or a decision file, it did not happen."
- **Blocked for 30 minutes?** Send the exact error, the file, and what you tried.
- **Disagreement on a contract:** the contract's owner proposes, the other approves. Time-box to 30 minutes. If deadlocked, choose the option that leaves fusion output and the golden test unchanged, and record the loser as a follow-up.
- **If one person is ill or blocked,** the other runs the cut list (§10.5) and takes only items explicitly listed, never a file they do not own without a `cross-owner` PR.

### 9.7 Guide card: Qamran

**Your lane:** ingest → features → windows → **Model A** → verify → triage → source health.

- [ ] D1 11:00: freeze `schema.py`, label, features, windows, and `validate_model_file` with Warisha.
- [ ] Build `grid_index.parquet` first; it defines the rows both models must score.
- [ ] Commit the small fixture `tests/fixtures/golden/predictions_a.parquet` by D2 noon so Warisha can build fusion and golden without waiting.
- [ ] Never open `model_b.py`, `artifacts/b/`, `fusion.py`, `runner.py`, `app/`, `.github/`, `requirements*.txt`. Comment on the PR instead.
- [ ] Before publishing Model A: validator passes, manifest updated, baseline recorded, model card updated, PR states the effect on fusion.
- [ ] Verifier: keep `check_alert`, `allowed_numbers_for`, `check_risk_alert` signatures stable; derive forbidden words from `tiers.py` only.
- [ ] Triage: no model calls, no network, never the word "fake"; Urdu aliases need a native speaker; `advisories.json` ships empty.
- [ ] Need a new feature column for B? Add it with a test and announce it at stand-up; do not let B fork it.
- [ ] Need a dependency? Ask Warisha in a PR comment with the package and version.
- [ ] Review Warisha's `docs/model_b.md` and the manifest for B; do not review or touch `model_b.py`.
- [ ] After the D2 18:00 freeze: no retrain. Approve the M7 command in a PR comment before Warisha runs it.

### 9.8 Guide card: Warisha

**Your lane:** contracts for model I/O → **Model B** → fusion → thresholds → runner → trust-layer glue → app → CI → docs.

- [ ] D1 morning: repo, branch protection, `OWNERS.yaml`, CI skeleton, `requirements.txt` / `requirements-dev.txt`, `models/base.py`, `models/eval.py`, `tiers.py`.
- [ ] Never open `model_a.py`, `artifacts/a/`, `features.py`, `windows.py`, `data_sources/`, `verify.py`, `triage.py`, `configs/basins.yaml|region.yaml|split.yaml`, `data/demo|static`, `scripts/01–03, 09, 10`. Comment on the PR instead.
- [ ] Fusion consumes parquet only: validate both files, check manifests and sha256, assert identical grids and identical snapshot hash.
- [ ] Fit thresholds on **train** only; write the validation check into `thresholds.yaml` provenance; never hand-edit it.
- [ ] Call `verify.check_risk_alert` for **every** outbound text; never edit `verify.py`. If the verifier is wrong, open an issue for Qamran.
- [ ] Urdu templates keep the Latin level word and basin id.
- [ ] Regenerate `golden_expected.json` only in your own PR, citing the producer's PR.
- [ ] Keep `torch` out of fusion/runner/app imports; `test_app_imports.py` must stay green.
- [ ] You own deps, CI and secrets handling: no token anywhere in git, reports or payloads.
- [ ] Run M7 once, only after Qamran's approval comment, with `--confirm <snapshot-hash>`; commit `runs/TEST_EVALUATED.json` immediately.
- [ ] Review Qamran's `docs/model_a.md` and the manifest for A; do not review or touch `model_a.py`.

---

## 10. Schedule, effort and the cut list

### 10.1 Milestones

| # | Milestone | Owner | Done when |
|---|---|---|---|
| M0 | Skeleton, branch protection, `OWNERS.yaml`, CI, contracts, fixture | both | `pytest` runs; CI green; fixture validates; contracts frozen 11:00 |
| M1 | Open-Meteo + grid patch + cache + features + windows + `grid_index.parquet` | Qamran | real values for 3 basins; windows pass leakage tests |
| M2a | **Model A: CNN-LSTM** + baseline + model card + validated output | Qamran | validation table with CNN-LSTM and logistic; manifest published |
| M2b | **Model B: LSTM** + baseline + model card + validated output | Warisha | validation table with LSTM and degree-day; manifest published |
| M3 | Fusion + thresholds + runner | Warisha | `runs/demo/` end to end; both models joined; validation check recorded |
| M4 | Adapter, decisions, messages | Warisha | `RiskResult` exact; texts deterministic |
| M5 | Verify + triage (Qamran); explain + alerts (Warisha) | both | every text verified; four verdicts reachable |
| **M6** | **Model freeze** (D2 18:00) | both | both manifests final; snapshot hash agreed |
| **M7** | **Single test evaluation** (D3 11:30) | Warisha runs, Qamran approves | marker file committed; `docs/TEST_REPORT.md` written |
| M8 | App + golden + ablation + WhatsApp + demo + README | both | offline demo; live send optional |

M2a and M2b run in parallel from D1 afternoon. Neither waits for the other: both train on the fixture first, then on real data when M1 lands.

### 10.2 Day plan

**D1, first 30 minutes, both:** `git init`, first commit, branch protection, agree the tree and `OWNERS.yaml`. Decide the daily hours (§10.3).

| When | Qamran (A) | Warisha (B) | Gate |
|---|---|---|---|
| **D1 morning** | Package, `schema.py` + `validate_model_file`, configs, label, fixture | Repo, CI, `OWNERS.yaml`, requirements split, `base.py`, `eval.py`, `tiers.py`, app shell | **Contracts frozen 11:00**; CI green |
| **D1 afternoon** | Open-Meteo + grid patch + cache + features + windows + `grid_index` | LSTM dataset on fixture; baseline; decisions/messages | Features work; both models can train |
| **D2 morning** | Train and tune CNN-LSTM; validation table; model card; commit `predictions_a` golden fixture by 12:00 | Train and tune LSTM; validation table; model card | Both produce the §3.7 file on the same grid |
| **D2 afternoon** | Verify; triage start; source health | Fusion + thresholds + runner + adapter; first fused run; golden | End-to-end run; **model freeze 18:00** |
| **D3 morning** | Triage finish + old-footage + adversarial tests; approve M7 command | **M7 at 11:30**; app; explain; verify tab | Verdicts reachable; marker committed |
| **D3 afternoon** | Demo support; model card polish | Alerts + WhatsApp dry-run; layers demo; README | **Freeze 15:00**; rehearse 16:00 |

If data slips, both models train on the fixture and the demo is labelled synthetic. If one model slips, fusion uses that mechanism's baseline and the report says so. The demo run is the only hard handoff, scheduled D2 afternoon.

### 10.3 Effort, re-totalled (issue 7)

Estimates include unit tests for the listed files. "Cut" marks hours that can be dropped via §10.5.

**Qamran**

| Work | Files | Est. | Cut |
|---|---|---|---|
| M0 shared (git, skeleton, contract freeze) | | 1.0 h | |
| Configs, split, fixture | `configs/*`, `data/demo/` | 1.5 h | |
| Open-Meteo ingest + cache + grid patch | `data_sources/` | 3.5 h | |
| Features + windows + `grid_index` | `features.py`, `models/windows.py` | 3.0 h | |
| `validate_model_file` + output-file tests | `schema.py`, `tests/test_model_output_files.py` | 0.5 h | |
| **Model A: CNN-LSTM** | `model_a.py`, `03_train_a.py` | 6.0 h | |
| Verification | `verify.py`, `test_verify.py` | 1.0 h | |
| Triage + aliases | `triage.py`, `data/static/*`, `data/advisories.json` | 2.0 h | |
| Adversarial tests (FW-07) | `test_triage.py` | 0.5 h | |
| Old-footage + date edge cases (FW-06) | `triage.py` | 0.75 h | 0.75 |
| Source health (FW-08) | `data_sources/health.py` | 1.0 h | 1.0 |
| Golden fixture A | `tests/fixtures/golden/` | 0.25 h | |
| Urdu alias review coordination | | 0.25 h | 0.25 |
| Provenance + model card | `docs/SOURCES.md`, `docs/model_a.md` | 0.5 h | |
| **Total / core** | | **21.75 h** | core **19.75 h** |

**Warisha**

| Work | Files | Est. | Cut |
|---|---|---|---|
| M0 shared (git, skeleton, contract freeze) | | 1.0 h | |
| Repo + CI + deps + ownership check + PR template | `.github/*`, `OWNERS.yaml`, `requirements*.txt`, `check_ownership.py` | 2.25 h | |
| Contracts + eval harness | `base.py`, `eval.py`, `tiers.py`, `report_schema.py` | 1.5 h | |
| M7 script + guard + leak test | `07_test_eval_once.py`, `test_no_test_leak.py` | 0.75 h | |
| **Model B: LSTM** | `model_b.py`, `04_train_b.py` | 4.0 h | |
| Fusion + thresholds (+ validation check) | `fusion.py`, `thresholds.py`, `05_fit_thresholds.py` | 2.0 h | |
| Runner + report | `runner.py`, `06_run.py` | 1.5 h | |
| Golden replay (FW-01) | `test_golden_replay.py`, fixtures | 1.0 h | |
| Adapter | `interop.py` | 1.0 h | |
| Decisions + messages | `decisions.py`, `messages.py` | 1.5 h | |
| Explain (FW-03) | `explain.py` | 1.0 h | 1.0 |
| App | `app/streamlit_app.py` | 2.0 h | |
| Alerts + WhatsApp + minimal kill switch | `alerts.py`, `channels/`, `overrides.json` | 1.5 h | 0.5 (live send) |
| Ablation (FW-02) | `08_ablation.py` | 1.0 h | 1.0 |
| State | `data/state/` | 0.25 h | |
| Model card | `docs/model_b.md` | 0.25 h | |
| **Total / core** | | **22.5 h** | core **20.0 h** |

**Stretch, outside both totals:** FW-10 1.0 h (W), FW-09 0.75 h (W), FW-12 full 0.75 h (W), FW-11 1.0 h (Q), FW-13 1.0 h (Q).

**Daily load:** 21.75 h and 22.5 h over three days is **7.25 and 7.5 h/day**; core only is 6.6 and 6.7 h/day. **Decide at D1 09:30:** if either person has less than 7 h/day, pre-cut §10.5 items 1–13 immediately (not on D3) and time-box deep-model tuning to the D2 12:00 gate. If one side finishes early they take a cut-list item from the other side **through a `cross-owner` PR**, never by editing a file directly.

### 10.4 Why the split is even

- Both people own one model end to end: data contract → training → validation table → model card.
- The crossing points are small and frozen: one feature table, one window builder, one grid, one output-file spec, one evaluation harness, one split.
- The models never import each other; fusion reads parquet, not code.
- The enforcement of the contracts (CI, `ownership`, fusion asserts) belongs to the person who does not own the cloudburst model, so nobody has an incentive to loosen it.

### 10.5 The single cut list

This is the **only** cut list; other sections point here. Cut from the top (1 first).

| # | Cut | Keep |
|---|---|---|
| 1 | Live WhatsApp send | dry-run payload |
| 2 | FW-10 ensemble band | |
| 3 | FW-11 climatology, FW-13 event loop | |
| 4 | FW-09 decision log | `alerts.json`, `outbound.json` |
| 5 | FW-12 full override (`until`, `note`) | minimal `kill` + `suppress` (lock 3) |
| 6 | Urdu templates and Urdu aliases | English; Latin level word rule |
| 7 | Map | table |
| 8 | Triage tab | triage function + CLI; FW-07 tests |
| 9 | FW-06 old-footage detection | |
| 10 | FW-08 source-health panel | `_health` basin coverage |
| 11 | FW-02 ablation table | |
| 12 | WhatsApp channel code | dry-run payload writer |
| 13 | FW-03 "Why not higher?" | |
| 14 | Deep-model refinement | that mechanism's baseline; deep path stays in the report |

**Never cut:** the prediction contract, the model-output file contract, one working model per mechanism, fusion with its grid assertions, the runner, the app, verification, CI and the `ownership` job, the single test evaluation with its guard, the offline demo run.

### 10.6 Blocker ownership

| Blocker | Owner | Deadline | Fallback |
|---|---|---|---|
| No data yet | Qamran | D1 afternoon | synthetic fixture, labelled |
| Feature/window contract changes late | Qamran | D1 11:00 | stop and freeze it |
| CNN-LSTM overfits or underfits | Qamran | D2 12:00 | smaller net, dropout, early stopping, logistic baseline |
| Grid patch slow or rate-limited | Qamran | D1 afternoon | 3×3 patch, cache; basin-mean fallback, documented |
| LSTM overfits or underfits | Warisha | D2 12:00 | smaller LSTM, dropout, early stopping, degree-day baseline |
| PyTorch install or CI slow | Warisha | D1 morning | CPU wheel + cache; pin the version |
| Model grids or snapshots do not join | both | D2 12:00 | assert on key set + snapshot hash; rebuild windows |
| No fused run | Warisha | D2 18:00 | Qamran codes against `stub_result()` and the frozen report shape |
| Train→validation threshold check fails (D-01) | both | D2 18:00 | document in model cards; joint decision; no silent refit |
| Empty `data/advisories.json` | Qamran | D3 | labelled `DEMO` advisory or `CANNOT_ASSESS` |
| No Urdu review | Warisha + Qamran | D3 | English-only demo (cut #6) |
| No Meta credentials / template not approved | Warisha | D1 / D3 | dry-run payload (cut #1) |
| Token leak risk | Warisha | always | env vars only; redact; CI `secrets` job |

---

## 11. Definition of done

**Models and contracts**
- [ ] Both models publish a validated §3.7 file with a manifest; baselines recorded; model cards cross-reviewed.
- [ ] Fusion asserts identical grid and snapshot hash; thresholds fitted on train with the D-01 validation check recorded.
- [ ] `tests/test_model_a.py` / `test_model_b.py` enforce the size limits (B: 1 LSTM layer, 32–64 units).
- [ ] **Exactly one** test evaluation was run (marker committed, `docs/TEST_REPORT.md` written); `test_no_test_leak.py` green.

**Trust layers**
- [ ] Adapter tests pin the tier→level table and one-row-per-basin.
- [ ] Decisions produce one alert per crossing, per escalation and per cooldown expiry, and exactly one all-clear.
- [ ] Every outbound message passes `check_risk_alert` for all five tiers in English and Urdu; planted errors and mixed-vocabulary messages are blocked.
- [ ] Triage reaches all four verdicts, is deterministic, never uses the banned words, and handles the date and adversarial cases in §6.5.
- [ ] `run_report.json` carries `verify`, `decisions`, `sources`, `delivery`, `models`.

**Engineering**
- [ ] CI green on `main` with all jobs in §6.10; a broken threshold turns `golden` red.
- [ ] The app starts from `requirements.txt` alone (no torch).
- [ ] `configs/split.yaml`, the prediction contract and shadow mode are untouched.
- [ ] The demo runs offline from the committed `runs/demo/`.
- [ ] Every file is covered by `OWNERS.yaml`; no cross-owner edit landed without a `cross-owner` PR.

**Stretch**
- [ ] `--dry-run` writes `outbound_payload.json` with zero network calls.
- [ ] One live WhatsApp text reaches the test recipient.
- [ ] A provider failure is recorded and the run still exits 0; tokens appear in no report, log or payload.

---

## 12. Alignment with the original procedure

Source: `KP_Flash_Flood_Project_Procedure_v2.pdf` (15 pages). The PDF is the project-level plan; this document is the 3-day prototype built from it. Every difference below must be repeated in the model cards and README.

| Area | Original PDF | This prototype | Why |
|---|---|---|---|
| Model A | CNN-LSTM, spatial + temporal branches | CNN-LSTM over an Open-Meteo patch + terrain, LSTM over 72 h | Structure kept; inputs substituted |
| Model A spatial inputs | IMERG, Meteosat, ERA5, DEM, sensors | Open-Meteo grid patch, basin series, static terrain | No NASA/EUMETSAT accounts or sensors |
| Model A lead time | 30–120 min | 6 h shared label | One label so A and B fuse; documented limitation |
| Run rate | every 15 min | hourly | Source data is hourly |
| Model B | v2; degree-day / hydrological + gradient boosting first | v1; LSTM + degree-day baseline | Two mechanisms in 3 days; baseline is the fallback |
| Module D | v1; gauge propagation | out of scope | No gauge feed |
| Tiers | Watch / Warning / Emergency | five internal tiers → three public words (`tiers.py`) | Keeps the six-column schema |
| Storage | Zarr/NetCDF, PostGIS, DVC | Parquet + JSON | 3-day prototype |
| UI | Gradio or Streamlit | Streamlit | Allowed by the PDF |

The prototype proves the pipeline, the two mechanisms, fusion, the trust layer and the app. It does not claim the PDF's operational coverage or lead time. A synthetic metric is never presented as a scientific result.

---

## 13. Templates (copy-paste)

### 13.1 `OWNERS.yaml`

```yaml
version: 1
owners:
  qamran:
    - configs/basins.yaml
    - configs/region.yaml
    - configs/split.yaml            # contract
    - data/demo/**
    - data/static/**
    - data/advisories.json
    - src/floodwatch/schema.py      # contract
    - src/floodwatch/features.py
    - src/floodwatch/data_sources/**
    - src/floodwatch/models/windows.py   # contract
    - src/floodwatch/models/model_a.py
    - src/floodwatch/verify.py
    - src/floodwatch/triage.py
    - scripts/0[123]_*.py
    - scripts/09_*.py
    - scripts/10_*.py
    - artifacts/a/**
    - docs/SOURCES.md
    - docs/model_a.md
    - docs/decisions/q-*.md
    - tests/test_{schema,features,windows,data_sources,health,model_a,model_output_files,verify,triage}.py
    - tests/helpers_a.py
    - tests/fixtures/golden/predictions_a.parquet
  warisha:
    - src/floodwatch/models/{base,eval,model_b}.py
    - src/floodwatch/{tiers,report_schema,thresholds,fusion,runner,interop,decisions,messages,explain,alerts}.py
    - src/floodwatch/channels/**
    - configs/thresholds.yaml
    - scripts/0[45678]_*.py
    - scripts/11_*.py
    - scripts/check_ownership.py
    - app/**
    - artifacts/b/**
    - runs/**
    - data/state/**
    - .github/**
    - OWNERS.yaml
    - requirements*.txt
    - pyproject.toml
    - Makefile
    - README.md
    - docs/FEATURE_BACKLOG.md
    - docs/model_b.md
    - docs/TEST_REPORT.md
    - docs/decisions/w-*.md
    - tests/conftest.py
    - tests/test_*.py               # every test not listed under qamran
    - tests/fixtures/golden/predictions_b.parquet
    - tests/fixtures/golden/golden_expected.json
contract:                           # need label `contract` + both approvals
  - src/floodwatch/schema.py
  - src/floodwatch/tiers.py
  - src/floodwatch/report_schema.py
  - src/floodwatch/models/base.py
  - src/floodwatch/models/windows.py
  - src/floodwatch/models/eval.py
  - configs/split.yaml
```

### 13.2 `.github/pull_request_template.md`

```markdown
**Author lane:** [ ] qamran  [ ] warisha
**What and why (2 lines):**

**Ownership**
- [ ] I only changed files I own, OR this PR is labelled `cross-owner` and the owner approved
- [ ] No contract file changed, OR this PR is labelled `contract` and lists every consumer

**Model outputs (delete if untouched)**
- Output changed: A / B   model_version: ______   snapshot: ______
- [ ] validator passes  [ ] manifest updated  [ ] model card updated
- Expected effect on fusion / golden:

**Checks**
- [ ] `make check` green locally  [ ] no secrets  [ ] no test-split access  [ ] size ≤ 400 lines
```

### 13.3 `artifacts/<a|b>/manifest.json`

```json
{
  "schema_version": 1,
  "module": "A_cloudburst_cnn_lstm",
  "model_version": "A_cloudburst_cnn_lstm-3f9a1c2-9b7e41d0",
  "data_snapshot_hash": "9b7e41d0…",
  "seed": 1234,
  "rows": 70128,
  "sha256": "…",
  "validated_at": "2026-11-09T11:40:00Z",
  "validation_metrics_path": "artifacts/a/metrics_a.json",
  "baseline_wins": false
}
```

### 13.4 Decision note (`docs/decisions/q-007-short-title.md`)

```markdown
# q-007: <title>
**Date:** … **Owner:** Qamran **Affects:** Warisha (fusion), tests/…
**Context:** …
**Decision:** …
**Who must change what:** …
```

### 13.5 Test-evaluation guard (sketch)

```python
def run_test_once(snapshot_hash: str, *, marker="runs/TEST_EVALUATED.json"):
    try:
        f = open(marker, "x")  # exclusive create: fails if it exists
    except FileExistsError:
        raise SystemExit("Test split already evaluated. Open a joint PR to change this.")
    with f:
        result = _evaluate_test_split_once()  # the only reader of split == "test"
        json.dump(
            {
                "snapshot": snapshot_hash,
                "models": _versions(),
                "thresholds": _thresholds(),
                "git_sha": _sha(),
                "utc": _now(),
                "metrics": result,
            },
            f,
            indent=2,
        )
```

### 13.6 `validate_model_file` checks (sketch)

```python
def validate_model_file(path, module, grid_index):
    df = pd.read_parquet(path)
    require_columns(
        df,
        [
            "basin_id",
            "timestamp_utc",
            "module",
            "split",
            "prob",
            "baseline_prob",
            "source",
            "top_drivers",
            "model_version",
        ],
    )
    assert (df["module"] == module).all()
    assert df["timestamp_utc"].dt.tz is not None and (df["timestamp_utc"].dt.minute == 0).all()
    assert df["prob"].between(0, 1).all() and df["prob"].notna().all()
    assert df.duplicated(["basin_id", "timestamp_utc"]).sum() == 0
    assert df.equals(df.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True))
    expected = grid_index[grid_index.split != "test"]  # test rows only after M7
    assert set(map(tuple, df[["basin_id", "timestamp_utc"]].values)) == set(
        map(tuple, expected[["basin_id", "timestamp_utc"]].values)
    )
    check_drivers_json(df["top_drivers"])
    check_metadata(path)
```
