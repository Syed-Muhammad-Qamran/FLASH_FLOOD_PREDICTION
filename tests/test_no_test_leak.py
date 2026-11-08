"""The test split stays sealed until the M7 evaluation.

Only ``scripts/07_test_eval_once.py`` may select test rows. This test scans the
pipeline scripts and the runtime modules for any other test-split selection and
checks the committed golden fixtures.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RUNTIME = ROOT / "src" / "floodwatch"
FIXTURES = ROOT / "tests" / "fixtures" / "golden"

ALLOWED = {"07_test_eval_once.py"}
PATTERNS = (
    re.compile(r"""split["']?\s*==\s*["']test["']"""),
    re.compile(r"""["']test["']\s*==\s*.*split"""),
    re.compile(r"""include_test\s*=\s*True"""),
)


def test_only_the_m7_script_selects_test_rows() -> None:
    offenders: list[str] = []
    for path in sorted(SCRIPTS.glob("*.py")):
        if path.name in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.name}: {pattern.pattern}")
    assert not offenders, "test-split access outside 07_test_eval_once.py:\n" + "\n".join(offenders)


def test_runtime_never_selects_test_rows() -> None:
    offenders: list[str] = []
    for path in sorted(RUNTIME.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    assert not offenders, "test-split access in runtime code:\n" + "\n".join(offenders)


def test_golden_fixtures_are_train_or_validation_only() -> None:
    for name in ("predictions_a.parquet", "predictions_b.parquet"):
        frame = pd.read_parquet(FIXTURES / name)
        assert set(frame["split"]) <= {"train", "validation"}, name
        assert "test" not in set(frame["split"]), name


def test_committed_model_outputs_exclude_test_rows() -> None:
    checked = 0
    for path in (ROOT / "artifacts").glob("*/predictions_*.parquet"):
        frame = pd.read_parquet(path)
        assert set(frame["split"]) <= {"train", "validation"}, path
        checked += 1
    if checked == 0:
        pytest.skip("prediction parquets are gitignored; CI checks the golden fixtures instead")


def test_runner_reads_no_features_or_test_rows() -> None:
    text = (RUNTIME / "runner.py").read_text(encoding="utf-8")
    assert "features_demo" not in text or "FEATURES_PATH" not in text
    assert '"test"' not in text and "'test'" not in text
