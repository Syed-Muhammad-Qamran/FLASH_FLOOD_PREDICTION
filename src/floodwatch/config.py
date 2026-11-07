"""Configuration loading. One place that knows where everything lives."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    env = os.environ.get("FLOODWATCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/floodwatch/config.py -> <repo>
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
CONFIG_DIR = ROOT / "configs"
DATA_DIR = Path(os.environ.get("FLOODWATCH_DATA", ROOT / "data")).expanduser().resolve()
RUNS_DIR = Path(os.environ.get("FLOODWATCH_RUNS", ROOT / "runs")).expanduser().resolve()
MODELS_DIR = Path(os.environ.get("FLOODWATCH_MODELS", ROOT / "models")).expanduser().resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = CONFIG_DIR / p
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=None)
def region() -> dict[str, Any]:
    return load_yaml("region.yaml")


@lru_cache(maxsize=None)
def split_config() -> dict[str, Any]:
    return load_yaml("split.yaml")


@lru_cache(maxsize=None)
def basins_config() -> dict[str, Any]:
    return load_yaml("basins.yaml")


@lru_cache(maxsize=None)
def models_config() -> dict[str, Any]:
    return load_yaml("models.yaml")


@lru_cache(maxsize=None)
def thresholds_config() -> dict[str, Any]:
    """Return thresholds with calibrated values taking precedence over defaults.

    Each module entry is flattened to a single dict and tagged with
    ``_provenance`` so the report can say whether a number was calibrated.
    """
    raw = load_yaml("thresholds.yaml")
    out: dict[str, Any] = {}
    for key in ("model_a", "model_b"):
        block = raw.get(key, {})
        calibrated = block.get("calibrated") or {}
        if calibrated:
            merged = {**block.get("defaults", {}), **calibrated}
            merged["_provenance"] = "calibrated"
        else:
            merged = dict(block.get("defaults", {}))
            merged["_provenance"] = "defaults"
        out[key] = merged
    out["fusion"] = dict(raw.get("fusion", {}))
    out["alerts"] = dict(raw.get("alerts", {}))
    return out


def basins() -> list[dict[str, Any]]:
    return list(basins_config()["basins"])


def basin_ids() -> list[str]:
    return [b["basin_id"] for b in basins()]


def basins_geojson_path() -> Path:
    p = Path(region()["basins_geojson"])
    return p if p.is_absolute() else ROOT / p


def ensure_dirs() -> None:
    for d in (DATA_DIR, RUNS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
