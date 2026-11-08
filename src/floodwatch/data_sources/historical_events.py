"""Historical ground-truth flood impacts, official citations, and case mapping.

Provides provenance-tracked reference datasets for the 2010 Pakistan Super Floods
and 2022 Monsoon Floods in northern Khyber Pakhtunkhwa.

Official sources:
  - NDMA Annual Flood Report 2010
  - NDMA Monsoon 2022 Daily Situation Reports (SitRep No. 74)
  - PDMA KP Post-Disaster Damage Assessment
  - Federal Flood Commission (FFC) River Discharge Records
  - UN OCHA Pakistan Floods Situation Reports
  - Dartmouth Flood Observatory (DFO) Satellite Inundation

Hydrological Caveats:
  - Open-Meteo ERA5 is gridded reanalysis, not gauge observations.
  - Basins are approximate control points, not delineated polygons.
  - District reports do not automatically label all sub-basins as flooded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "historical_cases.yaml"
DEFAULT_JSON_PATH = ROOT / "data" / "verified_events.json"


@dataclass(frozen=True)
class BasinGroundTruth:
    basin_id: str
    confirmed_status: str  # "confirmed_flood", "localized_minor", "no_confirmed_flood", "unmapped"
    source_citation: str
    spatial_granularity: (
        str  # "river_gauge_and_valley", "gauging_station_discharge", "district_report", etc.
    )
    gauge_discharge_cusecs: float | None
    notes: str
    comparable_with_model_a: bool
    comparable_with_model_b: bool

    @property
    def is_confirmed_flood(self) -> bool:
        return self.confirmed_status == "confirmed_flood"


@dataclass(frozen=True)
class HistoricalCase:
    case_id: str
    year: int
    title: str
    start_utc: str
    end_utc: str
    peak_utc: str
    primary_mechanism: str
    description: str
    ground_truth: dict[str, BasinGroundTruth]

    def get_basin(self, basin_id: str) -> BasinGroundTruth | None:
        return self.ground_truth.get(basin_id)

    def confirmed_basins(self) -> list[str]:
        return [b_id for b_id, gt in self.ground_truth.items() if gt.is_confirmed_flood]

    def non_confirmed_basins(self) -> list[str]:
        return [
            b_id
            for b_id, gt in self.ground_truth.items()
            if gt.confirmed_status in {"no_confirmed_flood", "localized_minor"}
        ]


def load_historical_cases(config_path: Path | str | None = None) -> dict[int, HistoricalCase]:
    """Load validated historical cases from YAML configuration."""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        # Fallback to JSON if yaml is missing
        json_p = Path(DEFAULT_JSON_PATH)
        if json_p.exists():
            return _load_from_json(json_p)
        raise FileNotFoundError(f"Historical cases config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases_dict: dict[int, HistoricalCase] = {}

    for year_key, case_data in raw.get("cases", {}).items():
        year = int(year_key)
        gt_dict: dict[str, BasinGroundTruth] = {}
        for b_id, b_info in case_data.get("ground_truth", {}).items():
            gt_dict[b_id] = BasinGroundTruth(
                basin_id=str(b_id),
                confirmed_status=str(b_info.get("confirmed_status", "unmapped")),
                source_citation=str(b_info.get("source_citation", "")),
                spatial_granularity=str(b_info.get("spatial_granularity", "unknown")),
                gauge_discharge_cusecs=b_info.get("gauge_discharge_cusecs"),
                notes=str(b_info.get("notes", "")),
                comparable_with_model_a=bool(b_info.get("comparable_with_model_a", True)),
                comparable_with_model_b=bool(b_info.get("comparable_with_model_b", True)),
            )

        win = case_data.get("window", {})
        cases_dict[year] = HistoricalCase(
            case_id=str(case_data.get("case_id", f"case_{year}")),
            year=year,
            title=str(case_data.get("title", f"{year} Flood Case")),
            start_utc=str(win.get("start", "")),
            end_utc=str(win.get("end", "")),
            peak_utc=str(win.get("peak", "")),
            primary_mechanism=str(case_data.get("primary_mechanism", "cloudburst")),
            description=str(case_data.get("description", "")).strip(),
            ground_truth=gt_dict,
        )

    return cases_dict


def _load_from_json(json_path: Path) -> dict[int, HistoricalCase]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: dict[int, HistoricalCase] = {}
    for ev in data.get("events", []):
        year = int(ev.get("year", 2000))
        gt: dict[str, BasinGroundTruth] = {}
        for b_id, b_data in ev.get("basins", {}).items():
            gt[b_id] = BasinGroundTruth(
                basin_id=str(b_id),
                confirmed_status=str(b_data.get("status", "unmapped")),
                source_citation=str(b_data.get("citation", "")),
                spatial_granularity=str(b_data.get("granularity", "unknown")),
                gauge_discharge_cusecs=b_data.get("discharge_cusecs"),
                notes=str(b_data.get("citation", "")),
                comparable_with_model_a=bool(b_data.get("comparable_model_a", True)),
                comparable_with_model_b=bool(b_data.get("comparable_model_b", True)),
            )
        out[year] = HistoricalCase(
            case_id=str(ev.get("case_id", f"case_{year}")),
            year=year,
            title=str(ev.get("title", "")),
            start_utc=str(ev.get("window_start", "")),
            end_utc=str(ev.get("window_end", "")),
            peak_utc=str(ev.get("peak_timestamp", "")),
            primary_mechanism=str(ev.get("primary_mechanism", "cloudburst")),
            description=str(data.get("description", "")),
            ground_truth=gt,
        )
    return out


def get_case(year: int) -> HistoricalCase:
    cases = load_historical_cases()
    if year not in cases:
        raise KeyError(
            f"No historical case recorded for year {year}. Available: {list(cases.keys())}"
        )
    return cases[year]


def list_citations() -> list[dict[str, str]]:
    """Return list of official citations and URLs for the reference data."""
    path = DEFAULT_CONFIG_PATH
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    citations = []
    for src_id, info in raw.get("sources", {}).items():
        citations.append(
            {
                "id": src_id,
                "name": info.get("name", ""),
                "url": info.get("url", ""),
                "date": info.get("date_published", ""),
                "granularity": info.get("granularity", ""),
                "attribution": info.get("attribution", ""),
                "license": info.get("license", ""),
            }
        )
    return citations
