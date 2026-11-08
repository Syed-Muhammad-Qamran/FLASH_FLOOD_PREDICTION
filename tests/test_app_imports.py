"""The app/runtime image must never load torch or scikit-learn."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

RUNTIME_MODULES = (
    "floodwatch",
    "floodwatch.tiers",
    "floodwatch.schema",
    "floodwatch.interop",
    "floodwatch.decisions",
    "floodwatch.verify",
    "floodwatch.messages",
    "floodwatch.explain",
    "floodwatch.triage",
    "floodwatch.report_schema",
    "floodwatch.thresholds",
    "floodwatch.fusion",
    "floodwatch.runner",
    "floodwatch.alerts",
    "floodwatch.channels.router",
    "floodwatch.data_sources.health",
)

HEAVY = {"torch", "sklearn"}


def _heavy_loaded() -> set[str]:
    return {name for name in sys.modules if name.split(".", 1)[0] in HEAVY}


def test_runtime_modules_import_without_torch_or_sklearn() -> None:
    before = _heavy_loaded()
    for module in RUNTIME_MODULES:
        importlib.import_module(module)
    assert _heavy_loaded() == before, (
        f"runtime imports loaded a heavy ML package: {sorted(_heavy_loaded() - before)}"
    )


def test_streamlit_app_imports_without_torch_or_sklearn() -> None:
    before = _heavy_loaded()
    try:
        importlib.import_module("app.streamlit_app")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".", 1)[0] == "streamlit":
            return
        raise
    assert _heavy_loaded() == before, (
        f"app import loaded a heavy ML package: {sorted(_heavy_loaded() - before)}"
    )
