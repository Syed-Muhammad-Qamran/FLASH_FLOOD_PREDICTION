"""Multi-seed ensemble uncertainty for Model B (and optionally Model A).

Epistemic uncertainty is estimated from K independent seed checkpoints: the
standard deviation of their predicted probabilities and whether every seed
agrees on the assigned tier. Rows where the seeds disagree are flagged
``provisional`` so the UI and downstream consumers can treat them as unstable.

The module is pure (numpy/pandas only): it never trains, never touches the
network, and never reads the test split. It only *consumes* already-published
per-seed prediction frames.

Additive columns (non-breaking to the §3.7 model-file contract):
    ``prob_std``     float  standard deviation across seeds (0.0 for a single seed)
    ``tier_stable``  bool   True when every seed assigns the same tier
    ``provisional``  bool   True when the prediction is unstable
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import schema

DEFAULT_STD_THRESHOLD = 0.05


@dataclass(frozen=True)
class EnsembleSummary:
    mean: np.ndarray
    std: np.ndarray
    tier_stable: np.ndarray
    provisional: np.ndarray
    n_seeds: int
    seeds: tuple[int, ...] = field(default_factory=tuple)


def _align_frames(frames: Iterable[pd.DataFrame]) -> list[pd.DataFrame]:
    """Sort each frame to a common (basin_id, timestamp_utc) key order."""
    aligned: list[pd.DataFrame] = []
    for frame in frames:
        copy = frame.copy()
        copy["timestamp_utc"] = pd.to_datetime(copy["timestamp_utc"], utc=True)
        copy = copy.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)
        aligned.append(copy)
    return aligned


def ensemble_probabilities(probabilities: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) across K seed arrays of equal shape."""
    if not probabilities:
        raise ValueError("at least one seed prediction is required")
    stack = np.stack([np.asarray(p, dtype=float) for p in probabilities], axis=0)
    if stack.ndim != 2:
        raise ValueError("each seed prediction must be a 1-D array")
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=0)
    return mean, std


def tier_stability(
    probabilities: list[np.ndarray],
    cutoffs: dict[str, float],
) -> np.ndarray:
    """True where every seed assigns the same tier."""
    if not probabilities:
        raise ValueError("at least one seed prediction is required")
    tiers_per_seed = [
        np.asarray([schema.tier_for(float(p), cutoffs) for p in seed], dtype=object)
        for seed in probabilities
    ]
    stable = np.ones(len(probabilities[0]), dtype=bool)
    for seed_tiers in tiers_per_seed[1:]:
        stable = stable & (seed_tiers == tiers_per_seed[0])
    return stable


def provisional_flags(
    std: np.ndarray,
    tier_stable: np.ndarray,
    *,
    std_threshold: float = DEFAULT_STD_THRESHOLD,
) -> np.ndarray:
    """A row is provisional when seeds disagree on tier or spread is large."""
    std = np.asarray(std, dtype=float)
    tier_stable = np.asarray(tier_stable, dtype=bool)
    return (~tier_stable) | (std > float(std_threshold))


def build_ensemble(
    probabilities: list[np.ndarray],
    cutoffs: dict[str, float],
    *,
    std_threshold: float = DEFAULT_STD_THRESHOLD,
    seeds: tuple[int, ...] = (),
) -> EnsembleSummary:
    mean, std = ensemble_probabilities(probabilities)
    stable = tier_stability(probabilities, cutoffs)
    provisional = provisional_flags(std, stable, std_threshold=std_threshold)
    return EnsembleSummary(
        mean=mean,
        std=std,
        tier_stable=stable,
        provisional=provisional,
        n_seeds=len(probabilities),
        seeds=seeds,
    )


def attach_ensemble_columns(
    frame: pd.DataFrame,
    summary: EnsembleSummary,
    *,
    mean_column: str = "prob",
) -> pd.DataFrame:
    """Attach ``prob_std``, ``tier_stable`` and ``provisional`` to a frame.

    ``mean_column`` is overwritten with the ensemble mean when more than one seed
    is present; a single seed leaves it untouched.
    """
    out = frame.copy()
    if len(summary.mean) != len(out):
        raise ValueError(
            f"ensemble length {len(summary.mean)} does not match frame length {len(out)}"
        )
    if summary.n_seeds > 1:
        out[mean_column] = np.clip(summary.mean, 0.0, 1.0)
    out["prob_std"] = np.clip(summary.std, 0.0, 1.0)
    out["tier_stable"] = summary.tier_stable.astype(bool)
    out["provisional"] = summary.provisional.astype(bool)
    return out


def ensemble_from_frames(
    frames: list[pd.DataFrame],
    *,
    cutoffs: dict[str, float],
    std_threshold: float = DEFAULT_STD_THRESHOLD,
    seeds: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Stack K per-seed prediction frames into one ensemble frame.

    Each input frame must contain ``basin_id``, ``timestamp_utc`` and ``prob``.
    Frames are aligned on (basin_id, timestamp_utc); an error is raised if their
    key sets differ.
    """
    if not frames:
        raise ValueError("at least one prediction frame is required")
    aligned = _align_frames(frames)
    keys = [set(zip(f["basin_id"].astype(str), f["timestamp_utc"], strict=False)) for f in aligned]
    if len({frozenset(k) for k in keys}) != 1:
        raise ValueError("seed prediction frames do not share the same key set")
    base = aligned[0][["basin_id", "timestamp_utc"]].copy()
    probabilities = [f["prob"].to_numpy(dtype=float) for f in aligned]
    summary = build_ensemble(probabilities, cutoffs, std_threshold=std_threshold, seeds=seeds)
    base["prob"] = summary.mean
    base = attach_ensemble_columns(base, summary)
    # Carry through static metadata from the first seed frame.
    for column in ("module", "split", "baseline_prob", "source", "top_drivers", "model_version"):
        if column in aligned[0].columns:
            base[column] = aligned[0][column].to_numpy()
    base.attrs.update(aligned[0].attrs)
    return base.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)


__all__ = [
    "DEFAULT_STD_THRESHOLD",
    "EnsembleSummary",
    "attach_ensemble_columns",
    "build_ensemble",
    "ensemble_from_frames",
    "ensemble_probabilities",
    "provisional_flags",
    "tier_stability",
]
