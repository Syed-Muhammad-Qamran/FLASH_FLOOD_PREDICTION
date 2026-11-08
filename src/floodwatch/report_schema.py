"""Invariant keys for ``run_report.json``.

Contract file: joint `contract` PR to change. New keys are additive.
"""

from __future__ import annotations

from typing import Any

REQUIRED_KEYS: tuple[str, ...] = (
    "verify",
    "decisions",
    "sources",
    "delivery",
    "models",
    "thresholds",
    "data_snapshot_hash",
)

MODEL_KEYS: tuple[str, ...] = ("a", "b")

DELIVERY_KEYS: tuple[str, ...] = ("mode", "attempted", "sent", "failed", "skipped")

VERIFY_KEYS: tuple[str, ...] = ("checked", "blocked", "problems")


class ReportSchemaError(ValueError):
    """Raised when run_report.json is missing an invariant key."""


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ReportSchemaError("run report must be a JSON object")
    missing = [key for key in REQUIRED_KEYS if key not in report]
    if missing:
        raise ReportSchemaError(f"run report is missing required keys: {missing}")
    if not isinstance(report["models"], dict):
        raise ReportSchemaError("report.models must be an object")
    missing_models = [key for key in MODEL_KEYS if key not in report["models"]]
    if missing_models:
        raise ReportSchemaError(f"report.models is missing: {missing_models}")
    if not isinstance(report["delivery"], dict):
        raise ReportSchemaError("report.delivery must be an object")
    missing_delivery = [k for k in DELIVERY_KEYS if k not in report["delivery"]]
    if missing_delivery:
        raise ReportSchemaError(f"report.delivery is missing: {missing_delivery}")
    if not isinstance(report["verify"], dict):
        raise ReportSchemaError("report.verify must be an object")
    missing_verify = [k for k in VERIFY_KEYS if k not in report["verify"]]
    if missing_verify:
        raise ReportSchemaError(f"report.verify is missing: {missing_verify}")
    return report
