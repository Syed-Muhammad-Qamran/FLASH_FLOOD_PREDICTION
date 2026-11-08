# PR plan — person-2 (Warisha) lane + shared gate/epic work

Status: all code is currently on `feature/person-b` in two large commits
(`1435198`, `bc56b2b`) plus the ensemble commit (`91ca43f`, PR #51).
The history is monolithic, so the plan below splits the work into
topic branches / PRs of ≤ 400 lines each, each referencing its issue.

Rules (from `guidelines/COLLABORATION_AND_CODING_GUIDELINES.md`):
- branch name `warisha/<topic>`, one topic per PR;
- PR ≤ 400 changed lines; no self-merge;
- contract files (§3) need the `contract` label and joint review;
- cross-owner files need the `cross-owner` label;
- `gh` authenticated as `Warisha-Bilal7`; issues on `Warisha-Bilal7/Flash-floods-monitoring`.

## Already executed
| Branch | PR | Issue | Scope |
|---|---|---|---|
| `warisha/ensemble` | #51 | #48 | `src/floodwatch/ensemble.py`, `interop.py` (+3 lines), `tests/test_ensemble.py` |

## Recommended split (execute by cherry-picking the listed files onto a fresh branch from `main`, then push + `gh pr create`)

### 1. `warisha/contracts` — issue #23 (contract label, joint review)
- `src/floodwatch/tiers.py`, `schema.py`, `models/base.py`, `models/eval.py`, `report_schema.py`
- `tests/test_tiers.py`, `tests/test_schema.py`

### 2. `warisha/skeleton-ci` — issue #24
- `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `.env.example`
- `OWNERS.yaml`, `Makefile`, `.github/**`, `scripts/check_ownership.py`
- `README.md`

### 3. `warisha/model-b` — issue #28
- `src/floodwatch/models/model_b.py`, `src/floodwatch/models/windows.py`
- `scripts/04_train_b.py`, `artifacts/b/metrics_b.json`, `artifacts/b/manifest.json`
- `docs/model_b.md`, `tests/test_model_b.py`

### 4. `warisha/fusion` — issue #30
- `src/floodwatch/fusion.py`, `src/floodwatch/interop.py`
- `tests/test_fusion.py`, `tests/test_interop.py`

### 5. `warisha/thresholds` — issue #31
- `src/floodwatch/thresholds.py`, `configs/thresholds.yaml`
- `scripts/05_fit_thresholds.py`, `tests/test_thresholds.py`

### 6. `warisha/runner` — issue #32
- `src/floodwatch/runner.py`, `scripts/06_run.py`
- `tests/test_runner.py`

### 7. `warisha/decisions` — (decision state machine)
- `src/floodwatch/decisions.py`, `tests/test_decisions.py`

### 8. `warisha/channels` — issue #46
- `src/floodwatch/verify.py`, `messages.py`, `alerts.py`
- `src/floodwatch/channels/router.py`, `channels/whatsapp.py`
- `tests/test_verify.py`, `tests/test_messages.py`, `tests/test_channels.py`

### 9. `warisha/triage` — issue #35 (cross-owner: triage.py is Q-owned)
- `src/floodwatch/triage.py`, `tests/test_triage.py`
- needs `cross-owner` label

### 10. `warisha/explain` — issue #43
- `src/floodwatch/explain.py`, `tests/test_explain.py`

### 11. `warisha/golden-replay` — issue #41
- `scripts/09_build_golden.py`, `tests/test_golden_replay.py`
- `tests/fixtures/golden/*` (predictions_a/b, labels, golden_expected)

### 12. `warisha/ablation` — issue #42
- `scripts/08_ablation.py`, `tests/test_ablation.py`
- `artifacts/ablation_validation.json`, `docs/ABLATION.md`

### 13. `warisha/test-eval-guard` — issue #40
- `scripts/07_test_eval_once.py`, `tests/test_no_test_leak.py`

### 14. `warisha/app` — issue #45
- `app/streamlit_app.py`, `app/__init__.py`, `tests/test_app_imports.py`

### 15. `warisha/data-sources` — issue #44 (cross-owner: Q-owned lane)
- `src/floodwatch/data_sources/openmeteo.py`, `cache.py`, `grid.py`, `terrain.py`, `health.py`
- `scripts/01_fetch_openmeteo.py`, `scripts/09_build_terrain.py`
- `docs/SOURCES.md`, `tests/test_data_sources.py`
- needs `cross-owner` label

### 16. `warisha/docs-freeze` — issue #29 (joint)
- `docs/model_b.md`, `docs/TEST_REPORT.md` (only after M7 guard runs)

## Execution template (per topic)
```powershell
git checkout main
git pull
git checkout -b warisha/<topic>
# copy only the listed files from feature/person-b, then:
git add <files>
git commit -m "<type>: <summary> (#issue)"
git push -u origin warisha/<topic>
gh pr create --base main --head warisha/<topic> --title "..." --body "Closes #issue"
```

## Notes
- `artifacts/*/*.pt` and `artifacts/*/predictions_*.parquet` are gitignored; manifests/metrics/json are committed.
- `data/demo/features_demo.parquet` (12.5 MB) is currently unignored; confirm before committing.
- `runs/demo/` is committed for offline demo; `runs/TEST_EVALUATED.json` only after the M7 guard.
- No self-merge: each PR waits for Qamran review/approval.
- CI gates: lint, tests, contract, golden, fusion-smoke, app-imports, ownership, secrets.
