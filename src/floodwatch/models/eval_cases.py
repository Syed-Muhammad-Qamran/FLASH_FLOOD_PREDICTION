"""Evaluation of historical flood cases (2010 and 2022 Pakistan Floods).

Evaluates Model A (cloudburst) and Model B (snowmelt/rain-on-snow) on external
historical cases, strictly separating source-confirmed affected areas from
model-predicted risk areas.

Does not import torch or scikit-learn so it remains lightweight and import-safe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..data_sources.historical_events import HistoricalCase, load_historical_cases


@dataclass(frozen=True)
class BasinCaseComparison:
    basin_id: str
    confirmed_status: str  # "confirmed_flood", "localized_minor", "no_confirmed_flood", "unmapped"
    source_citation: str
    spatial_granularity: str
    model_predicted_risk: bool
    model_max_prob: float
    model_tier: str
    is_comparable: bool
    comparison_outcome: str  # "TRUE_POSITIVE", "FALSE_NEGATIVE", "FALSE_POSITIVE", "TRUE_NEGATIVE", "NON_COMPARABLE", "UNMAPPED"
    caveat: str


@dataclass(frozen=True)
class CaseEvaluationReport:
    case_id: str
    year: int
    title: str
    mechanism: str
    model_evaluated: str
    threshold_used: float
    window_start: str
    window_end: str
    comparisons: list[BasinCaseComparison]
    metrics: dict[str, float]
    provenance_citations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "year": self.year,
            "title": self.title,
            "mechanism": self.mechanism,
            "model_evaluated": self.model_evaluated,
            "threshold_used": self.threshold_used,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "comparisons": [asdict(c) for c in self.comparisons],
            "metrics": self.metrics,
            "provenance_citations": self.provenance_citations,
        }


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def compute_metrics(
    tp: int, fp: int, fn: int, tn: int, y_true: list[int], y_prob: list[float]
) -> dict[str, float]:
    """Calculate POD/Recall, FAR, CSI, Precision, F1, Accuracy, Brier score."""
    pod = _safe_div(tp, tp + fn)
    far = _safe_div(fp, tp + fp)
    csi = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)

    # Brier score
    if y_true and len(y_true) == len(y_prob):
        brier = float(np.mean([(p - t) ** 2 for p, t in zip(y_prob, y_true, strict=False)]))
    else:
        brier = 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "pod": round(pod, 4),
        "far": round(far, 4),
        "csi": round(csi, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier, 4),
    }


def evaluate_case(
    *,
    case_year: int,
    predictions_df: pd.DataFrame,
    model_name: str = "Model A",
    threshold: float = 0.50,
) -> CaseEvaluationReport:
    """Evaluate external historical flood case against official confirmed ground truth.

    predictions_df requires: basin_id, timestamp_utc, prob
    """
    cases = load_historical_cases()
    if case_year not in cases:
        raise KeyError(f"Historical case {case_year} not found in configuration")

    case: HistoricalCase = cases[case_year]
    preds = predictions_df.copy()
    preds["timestamp_utc"] = pd.to_datetime(preds["timestamp_utc"], utc=True)

    start_ts = pd.to_datetime(case.start_utc, utc=True)
    end_ts = pd.to_datetime(case.end_utc, utc=True)

    # Filter predictions to case window
    window_preds = preds[(preds["timestamp_utc"] >= start_ts) & (preds["timestamp_utc"] <= end_ts)]

    # Compute max probability and predicted risk per basin
    basin_max_prob: dict[str, float] = {}
    for b_id, grp in window_preds.groupby("basin_id"):
        basin_max_prob[str(b_id)] = float(grp["prob"].max())

    is_model_a = "a" in model_name.lower()
    comparisons: list[BasinCaseComparison] = []

    tp = fp = fn = tn = 0
    y_true_eval: list[int] = []
    y_prob_eval: list[float] = []

    for b_id, gt in case.ground_truth.items():
        max_p = basin_max_prob.get(b_id, 0.0)
        pred_risk = max_p >= threshold

        # Determine tier
        if max_p >= 0.85:
            tier = "RED"
        elif max_p >= threshold:
            tier = "ORANGE"
        elif max_p >= 0.20:
            tier = "YELLOW"
        else:
            tier = "GREEN"

        comparable = gt.comparable_with_model_a if is_model_a else gt.comparable_with_model_b

        if not comparable or gt.confirmed_status == "unmapped":
            outcome = "NON_COMPARABLE" if not comparable else "UNMAPPED"
            caveat = (
                f"Mechanistic/geographic mismatch: {b_id} is in a different hydrological regime "
                f"or lacks point-to-point gauge resolution ({gt.spatial_granularity})."
            )
        else:
            actual_flood = gt.is_confirmed_flood
            if actual_flood and pred_risk:
                outcome = "TRUE_POSITIVE"
                tp += 1
            elif actual_flood and not pred_risk:
                outcome = "FALSE_NEGATIVE"
                fn += 1
            elif not actual_flood and pred_risk:
                outcome = "FALSE_POSITIVE"
                fp += 1
            else:
                outcome = "TRUE_NEGATIVE"
                tn += 1

            y_true_eval.append(1 if actual_flood else 0)
            y_prob_eval.append(max_p)
            caveat = f"Spatial granularity: {gt.spatial_granularity}. Direct comparison valid."

        comparisons.append(
            BasinCaseComparison(
                basin_id=b_id,
                confirmed_status=gt.confirmed_status,
                source_citation=gt.source_citation,
                spatial_granularity=gt.spatial_granularity,
                model_predicted_risk=pred_risk,
                model_max_prob=round(max_p, 4),
                model_tier=tier,
                is_comparable=comparable,
                comparison_outcome=outcome,
                caveat=caveat,
            )
        )

    metrics = compute_metrics(tp, fp, fn, tn, y_true_eval, y_prob_eval)

    citations = [gt.source_citation for gt in case.ground_truth.values() if gt.source_citation]
    unique_citations = sorted(set(citations))

    return CaseEvaluationReport(
        case_id=case.case_id,
        year=case.year,
        title=case.title,
        mechanism=case.primary_mechanism,
        model_evaluated=model_name,
        threshold_used=threshold,
        window_start=case.start_utc,
        window_end=case.end_utc,
        comparisons=comparisons,
        metrics=metrics,
        provenance_citations=unique_citations,
    )
