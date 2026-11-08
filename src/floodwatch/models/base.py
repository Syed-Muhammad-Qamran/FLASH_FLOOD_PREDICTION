"""Shared model-output dataclasses.

Contract file: joint `contract` PR to change. No torch import lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tiers import Tier


@dataclass(frozen=True)
class ModelOutput:
    """One scored grid row from either mechanism."""

    prob: float
    top_drivers: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    model_version: str = ""
    trained_at: str = ""
    baseline_prob: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.prob) <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {self.prob!r}")
        if not 0.0 <= float(self.baseline_prob) <= 1.0:
            raise ValueError(f"baseline_prob must be in [0, 1], got {self.baseline_prob!r}")

    def as_row(
        self,
        *,
        basin_id: str,
        timestamp_utc: Any,
        module: str,
        split: str,
        tier: Tier | None = None,
    ) -> dict[str, Any]:
        import json

        row: dict[str, Any] = {
            "basin_id": basin_id,
            "timestamp_utc": timestamp_utc,
            "module": module,
            "split": split,
            "prob": float(self.prob),
            "baseline_prob": float(self.baseline_prob),
            "source": "deep",
            "top_drivers": json.dumps(
                [{"feature": d["feature"], "weight": float(d["weight"])} for d in self.top_drivers],
                ensure_ascii=False,
            ),
            "model_version": self.model_version,
        }
        if tier is not None:
            row["tier"] = tier
        return row


@dataclass(frozen=True)
class ModelSpec:
    """The frozen size limits for a mechanism, mirrored by its test."""

    name: str
    module: str
    max_lstm_layers: int
    min_units: int
    max_units: int
    max_conv_blocks: int = 0
    max_conv_filters: int = 0
