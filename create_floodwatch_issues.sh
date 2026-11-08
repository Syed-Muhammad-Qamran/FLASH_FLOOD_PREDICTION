#!/usr/bin/env bash
# Creates the 13 plan-review issues for FloodWatch KP using the GitHub CLI.
#
# Usage:
#   gh auth login                                  # once
#   REPO=your-user/floodwatch ./create_floodwatch_issues.sh
#   DRY_RUN=1 REPO=your-user/floodwatch ./create_floodwatch_issues.sh   # print only
#
# Needs: gh (https://cli.github.com). No token is stored in this script.
set -euo pipefail

: "${REPO:?Set REPO=owner/name}"
DRY_RUN="${DRY_RUN:-0}"

# --- labels (ignored if they already exist) -------------------------------
make_label() {
  [ "$DRY_RUN" = "1" ] && { echo "label: $1"; return; }
  gh label create "$1" --color "$2" --description "$3" --repo "$REPO" 2>/dev/null || true
}
make_label "plan-review"     "5319e7" "Found while reviewing FEATURE_BACKLOG.md"
make_label "contract"        "d93f0b" "Touches a frozen contract; needs both approvals"
make_label "process"         "c5def5" "Ownership, scheduling or plan consistency"
make_label "docs"            "0075ca" "Plan or documentation fix"
make_label "risk"            "b60205" "Could mislead users or break the build"
make_label "owner:qamran"    "bfdadc" "Owned by Qamran (A / data / verify / triage)"
make_label "owner:warisha"   "fef2c0" "Owned by Warisha (B / fusion / app / CI)"
make_label "owner:both"      "e4e669" "Joint decision"

# --- helper ----------------------------------------------------------------
new_issue() {
  local title="$1" labels="$2" body="$3"
  if [ "$DRY_RUN" = "1" ]; then
    echo "-----"; echo "TITLE:  $title"; echo "LABELS: $labels"; echo "$body"; return
  fi
  gh issue create --repo "$REPO" --title "$title" --label "$labels" --body "$body"
}

# 1 ---------------------------------------------------------------------------
new_issue "Schedule the single test-split evaluation (owner and day)" \
  "plan-review,process,contract,owner:both" "$(cat <<'EOF'
**Problem.** The invariant "one test evaluation" is in §0, §8.1 and §9.8, but no milestone, owner or day performs it.

**Proposed fix**
- Add a milestone (after M3, before the demo freeze) that runs the test split once with `models/eval.py`.
- Name one owner and require the other to review the command beforehand.
- Record the run in the report (data snapshot hash, seed, thresholds used) so it is provably single.
- Add a guard (e.g. a `test_evaluated: true` marker file) that makes a second run fail loudly.

**Acceptance**
- [ ] Milestone and owner written into §9.4 / §9.5.
- [ ] Guard exists and is tested.
EOF
)"

# 2 ---------------------------------------------------------------------------
new_issue "Golden replay (FW-01) may not be runnable or stable in CI" \
  "plan-review,risk,owner:warisha,owner:qamran" "$(cat <<'EOF'
**Problem**
- The golden test replays the full pipeline, but checkpoints are gitignored unless a demo checkpoint < 20 MB is committed, so CI may have nothing to run.
- Matching `prob` to 1e-6 across machines with PyTorch is fragile even on CPU with a fixed seed.
- The "Aug 2022 demo window" must not fall inside the test years in `configs/split.yaml`, otherwise the golden test touches the test split.

**Proposed fix**
- Replay from committed `predictions_a/b.parquet` fixtures through fusion + thresholds + tiers (the part we actually want pinned), not through the neural nets.
- Keep exact match on `tier`; use a tolerance (e.g. 1e-4) for `prob` unless a deterministic checkpoint is committed.
- Assert in the test that the window is inside train/validation years.

**Acceptance**
- [ ] `golden` job green on a clean clone, offline, < 60 s.
- [ ] Window-vs-split assertion present.
EOF
)"

# 3 ---------------------------------------------------------------------------
new_issue "Ownership holes: unowned tests, ablation script, verify wiring" \
  "plan-review,process,owner:both" "$(cat <<'EOF'
**Problem** (§9.1 / §9.8)
- `tests/test_golden_replay.py`, `test_schema.py`, `test_triage.py`, `test_decisions.py`, `test_messages.py` etc. appear in neither "must not edit" list.
- `scripts/08_ablation.py` needs fusion code (Warisha) but sits in `scripts/01_*..09_*`, which Warisha may not edit. Same for `09_climatology.py`.
- `alerts.deliver()` (Warisha) must call `verify.py` (Qamran): the call site is a cross-owner dependency with no stated rule.

**Proposed fix**
- Add a test-file owner column to §9.1 (rule: the owner of the module owns its tests).
- Reassign `08_ablation.py` to Warisha; keep `09_climatology.py` with Qamran.
- State that Qamran publishes `check_risk_alert` as a stable function signature and Warisha only calls it.

**Acceptance**
- [ ] Every file in the target tree has exactly one owner.
EOF
)"

# 4 ---------------------------------------------------------------------------
new_issue "FW-12 references Telegram and a different lock count than §8.11" \
  "plan-review,docs,owner:warisha" "$(cat <<'EOF'
**Problem.** FW-12 says "`FLOODWATCH_SEND=1` on top of `delivery: telegram`, so two things must be true". Telegram is not part of the plan (§5, §8.11 use WhatsApp), and §8.11.4 describes three locks.

**Proposed fix.** Rewrite FW-12 to use `delivery: whatsapp` + `FLOODWATCH_SEND=1`, and use one consistent lock count in both sections.

**Acceptance**
- [ ] No mention of Telegram remains in the plan.
- [ ] FW-12 and §8.11.4 agree.
EOF
)"

# 5 ---------------------------------------------------------------------------
new_issue "Model B size limit inconsistent (§9.0 vs §9.3 rule 11)" \
  "plan-review,docs,contract,owner:warisha" "$(cat <<'EOF'
**Problem.** §9.0 says Model B is one LSTM layer, 32-64 units. §9.3 rule 11 allows up to 2 LSTM layers and 64 units.

**Proposed fix.** Pick one limit, update both places and `docs/model_b.md`, and add a test that fails if the model exceeds it.

**Acceptance**
- [ ] One limit stated everywhere; test enforces it.
EOF
)"

# 6 ---------------------------------------------------------------------------
new_issue "Three conflicting cut orders (§1, §8.9, §9.9)" \
  "plan-review,process,docs,owner:both" "$(cat <<'EOF'
**Problem.** The cut rank column in §1, the cut order in §8.9 and the cut line in §9.9 disagree (e.g. where Urdu, triage tab, ablation and source-health sit).

**Proposed fix.** Declare §9.9 canonical, delete or reference it from §1 and §8.9, and keep "never cut: contract, one model per mechanism, fusion, runner, app, verification, CI, offline demo run".

**Acceptance**
- [ ] One cut list in the plan; other sections link to it.
EOF
)"

# 7 ---------------------------------------------------------------------------
new_issue "Effort totals do not include shared M0, FW items or the 1.5 h channel work" \
  "plan-review,process,owner:both" "$(cat <<'EOF'
**Problem**
- 17.5 h per person excludes the shared M0 hour.
- "Alerts + WhatsApp" is 1.0 h in §9.1 but 1.5 h in §8.11.8.
- Warisha's table omits FW-01 (golden), FW-08 (source health), FW-02 (ablation) and test-writing for several modules.

**Proposed fix.** Re-total both columns including M0 and the backlog items; if over ~6 h/day each, move items to the cut line now rather than on D3.

**Acceptance**
- [ ] Both totals recomputed and the daily load shown.
EOF
)"

# 8 ---------------------------------------------------------------------------
new_issue "Triage NOT_SUPPORTED can falsely reassure on a real, missed event" \
  "plan-review,risk,owner:qamran" "$(cat <<'EOF'
**Problem.** `NOT_SUPPORTED` is returned whenever the tier is below `high`. An hourly model with a 6 h label can miss a real event, so "not supported" may be read as "did not happen".

**Proposed fix**
- Evidence text must say "no model signal and no official advisory found", never imply the event did not occur.
- Consider renaming the verdict to something like `no_signal_found`.
- Keep the "Check PDMA / PMD / Rescue 1122" line, and add a test that the evidence string never contains words like "false", "fake", "did not happen".

**Acceptance**
- [ ] Evidence wording reviewed and pinned by a test.
EOF
)"

# 9 ---------------------------------------------------------------------------
new_issue "FW-06 date extraction edge cases (dd/mm vs mm/dd, bare current year)" \
  "plan-review,risk,owner:qamran" "$(cat <<'EOF'
**Problem**
- `27/06/2025` is unambiguous but `03/04/2025` is not (dd/mm vs mm/dd).
- A bare current year ("2026") parses as an old date and wrongly downgrades a current report.
- Years inside unrelated phrases ("since 2010") can trigger the downgrade.

**Proposed fix**
- Treat ambiguous numeric dates as the older plausible reading, or assume dd/mm (Pakistan convention) and document it.
- A bare year equal to the current year is not a downgrade signal.
- Only bare years near flood words count; add tests for each case.

**Acceptance**
- [ ] Tests cover ambiguous date, current-year bare, and "since YYYY" phrasing.
EOF
)"

# 10 --------------------------------------------------------------------------
new_issue "Urdu templates must still contain the Latin level word for the verifier" \
  "plan-review,risk,owner:warisha,owner:qamran" "$(cat <<'EOF'
**Problem.** `check_alert` looks for the English tier word or Level name. A fully Urdu template without it fails verification, or tempts someone to loosen the verifier.

**Proposed fix.** Either keep the English level word (e.g. "ORANGE") inside every Urdu message, or extend the verifier with an explicit Urdu word table that is native-speaker reviewed. Do not weaken the "exactly one level word" rule.

**Acceptance**
- [ ] Urdu templates pass `check_risk_alert` for all five tiers.
- [ ] Native-speaker review recorded.
EOF
)"

# 11 --------------------------------------------------------------------------
new_issue "Two vocabularies (Watch/Warning/Emergency vs GREEN..RED) in the verifier" \
  "plan-review,contract,owner:both" "$(cat <<'EOF'
**Problem.** Public words (Watch, Warning, Emergency) and internal Levels (GREEN, YELLOW, ORANGE, RED) both exist. The "no other tier word" rule must cover every word in both sets, otherwise a message can contain "Warning" and "RED" together unnoticed.

**Proposed fix.** Define one table in `interop.py` mapping tier -> Level -> public word, have `verify.py` build its forbidden-word set from it, and test mixed-vocabulary messages.

**Acceptance**
- [ ] Mixed-vocabulary messages are blocked.
- [ ] Table is the single source of truth.
EOF
)"

# 12 --------------------------------------------------------------------------
new_issue "Split requirements: Spaces app should not install torch" \
  "plan-review,process,owner:warisha" "$(cat <<'EOF'
**Problem.** One `requirements.txt` serves CI and Hugging Face Spaces, but the app only reads run outputs and does not need `torch`.

**Proposed fix.** `requirements.txt` for the app (pandas, pyarrow, streamlit, PyYAML, numpy) and `requirements-dev.txt` / `pyproject` extras for training and CI (torch CPU wheel, scikit-learn, pytest, ruff).

**Acceptance**
- [ ] App starts in a fresh venv with only `requirements.txt`.
- [ ] CI installs the dev set and caches it.
EOF
)"

# 13 --------------------------------------------------------------------------
new_issue "Decide where tier thresholds are fitted (train vs validation)" \
  "plan-review,contract,owner:both" "$(cat <<'EOF'
**Problem.** The plan fixes "train-only threshold fitting", but probabilities on training data are optimistic (the model has seen those rows), which can set cutoffs too loose. Validation is already used for early stopping.

**Options**
1. Keep train-only and document the optimism, with a check that validation POD/FAR stay acceptable.
2. Use out-of-fold predictions on train.
3. Fit on validation and say so explicitly (changes the invariant).

**Acceptance**
- [ ] Decision recorded in the plan and `docs/model_*.md`.
- [ ] Invariant wording updated if option 3 is chosen.
EOF
)"

echo "Done."
