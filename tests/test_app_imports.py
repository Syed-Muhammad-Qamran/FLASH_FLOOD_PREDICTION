"""The app/runtime image must never load torch or scikit-learn."""

from __future__ import annotations

import importlib
import sys

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


def _assert_no_heavy_ml() -> None:
    for name in list(sys.modules):
        root = name.split(".", 1)[0]
        assert root not in {"torch", "sklearn"}, f"heavy ML package loaded: {name}"


def test_runtime_modules_import_without_torch_or_sklearn() -> None:
    for module in RUNTIME_MODULES:
        importlib.import_module(module)
    _assert_no_heavy_ml()


def test_streamlit_app_imports_without_torch_or_sklearn() -> None:
    # streamlit is a runtime dependency, but the import-safety property holds
    # even when it is absent (the module structure must not reach for ML).
    try:
        importlib.import_module("app.streamlit_app")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".", 1)[0] == "streamlit":
            return
        raise
    _assert_no_heavy_ml()
