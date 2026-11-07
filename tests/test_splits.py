from __future__ import annotations

import pandas as pd
import pytest

from floodwatch.splits import assign_split, describe, load_windows


def test_windows_are_ordered_and_gapped():
    w = load_windows()
    assert w["train"].end <= w["validation"].start
    assert w["validation"].end <= w["test"].start
    # gap is actually applied
    assert w["validation"].start > pd.Timestamp("2022-01-01", tz="UTC")
    assert w["test"].start > pd.Timestamp("2024-01-01", tz="UTC")


def test_no_year_appears_in_two_splits():
    ts = pd.Series(pd.date_range("2009-01-01", "2026-12-31", freq="3h", tz="UTC"))
    labels = assign_split(ts)
    for split in ("train", "validation", "test"):
        years = set(ts[labels == split].dt.year.unique())
        assert years, f"{split} is empty"
    train_years = set(ts[labels == "train"].dt.year)
    test_years = set(ts[labels == "test"].dt.year)
    assert not (train_years & test_years)


def test_gap_rows_are_not_assigned_to_a_split():
    w = load_windows()
    ts = pd.Series([w["train"].end + pd.Timedelta(hours=1)])
    assert assign_split(ts).iloc[0] == "gap"


def test_describe_shape():
    df = describe()
    assert list(df["split"]) == ["train", "validation", "test"]
    assert (df["days"] > 0).all()
