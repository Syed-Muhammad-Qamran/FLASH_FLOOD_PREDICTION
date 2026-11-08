"""Small on-disk cache for fetched raw/source payloads.

The cache is optional and lives under ``data/raw/`` (gitignored). It never stores
credentials because Open-Meteo requires none.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "data" / "raw" / "cache"


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    key: str
    created_at: datetime
    meta: dict[str, Any]


def cache_key(source: str, **parts: Any) -> str:
    payload = json.dumps({"source": source, **parts}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def write_frame(
    frame: pd.DataFrame, *, key: str, meta: dict[str, Any], cache_dir: Path | None = None
) -> CacheEntry:
    base = Path(cache_dir or DEFAULT_CACHE)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{key}.parquet"
    frame.to_parquet(path, index=False)
    meta_path = base / f"{key}.json"
    envelope = {
        "key": key,
        "created_at": datetime.now(UTC).isoformat(),
        "rows": int(len(frame)),
        **meta,
    }
    meta_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
    return CacheEntry(path=path, key=key, created_at=datetime.now(UTC), meta=envelope)


def read_frame(key: str, *, cache_dir: Path | None = None) -> pd.DataFrame | None:
    path = Path(cache_dir or DEFAULT_CACHE) / f"{key}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def read_meta(key: str, *, cache_dir: Path | None = None) -> dict[str, Any] | None:
    path = Path(cache_dir or DEFAULT_CACHE) / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None
