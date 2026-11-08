"""Tests for Open-Meteo retry/cache behavior and refusal to synthesize fake data."""

from __future__ import annotations

import tempfile
import urllib.error
from pathlib import Path

import pytest

from floodwatch.data_sources import openmeteo


def test_openmeteo_retries_on_http_429() -> None:
    """Ensure HTTP 429 rate limits trigger bounded retry before failing or succeeding."""
    calls = []

    def mock_fetcher_429(url, params, timeout):
        calls.append(len(calls))
        if len(calls) < 3:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return {
            "latitude": 35.48,
            "longitude": 72.58,
            "hourly": {
                "time": ["2022-08-26T00:00:00Z"],
                "temperature_2m": [15.0],
                "precipitation": [5.0],
            },
        }

    # Should succeed after retrying
    res = openmeteo.fetch_archive(
        basin_id="swat_kalam",
        latitude=35.48,
        longitude=72.58,
        start_date="2022-08-26",
        end_date="2022-08-26",
        fetcher=mock_fetcher_429,
        initial_backoff_s=0.01,
        max_backoff_s=0.05,
    )
    assert len(calls) == 3
    assert len(res.frame) == 1
    assert res.frame["rain_mm"].iloc[0] == 5.0


def test_openmeteo_refuses_to_synthesize_data_on_failure() -> None:
    """Ensure that on persistent API failure, OpenMeteoError is raised, never fake data."""

    def fail_fetcher(url, params, timeout):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)

    with pytest.raises(openmeteo.OpenMeteoError) as exc_info:
        openmeteo.fetch_archive(
            basin_id="swat_kalam",
            latitude=35.48,
            longitude=72.58,
            start_date="2022-08-26",
            end_date="2022-08-26",
            fetcher=fail_fetcher,
            max_retries=2,
            initial_backoff_s=0.01,
            max_backoff_s=0.05,
        )
    assert "refusing to synthesize fake weather" in str(exc_info.value) or "HTTPError" in str(
        exc_info.value
    )


def test_openmeteo_caching_behavior() -> None:
    """Ensure disk caching stores and returns identical historical frames."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        call_count = [0]

        def sample_fetcher(url, params, timeout):
            call_count[0] += 1
            return {
                "latitude": 35.48,
                "longitude": 72.58,
                "hourly": {
                    "time": ["2022-08-26T00:00:00Z", "2022-08-26T01:00:00Z"],
                    "temperature_2m": [12.0, 14.0],
                    "precipitation": [3.5, 4.2],
                },
            }

        # First fetch: contacts fetcher and writes to cache
        res1 = openmeteo.fetch_archive(
            basin_id="swat_kalam",
            latitude=35.48,
            longitude=72.58,
            start_date="2022-08-26",
            end_date="2022-08-26",
            fetcher=sample_fetcher,
            use_cache=True,
            cache_dir=tmp_path,
        )
        assert call_count[0] == 1
        assert len(res1.frame) == 2

        # Second fetch: reads from disk cache, never contacts fetcher
        res2 = openmeteo.fetch_archive(
            basin_id="swat_kalam",
            latitude=35.48,
            longitude=72.58,
            start_date="2022-08-26",
            end_date="2022-08-26",
            fetcher=sample_fetcher,
            use_cache=True,
            cache_dir=tmp_path,
        )
        assert call_count[0] == 1  # Fetcher was NOT called again!
        assert len(res2.frame) == 2
        assert res2.frame["rain_mm"].tolist() == [3.5, 4.2]
