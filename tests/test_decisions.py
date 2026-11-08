"""Alert state machine: threshold, cooldown, escalation and all-clear."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from floodwatch import decisions
from floodwatch.tiers import Level

NOW = datetime(2022, 8, 27, 12, tzinfo=UTC)


def _state(level: Level = Level.GREEN, **kwargs) -> decisions.AreaState:
    return decisions.AreaState(area_id="swat_kalam", level=level, **kwargs)


def test_green_and_yellow_never_alert() -> None:
    for level in (Level.GREEN, Level.YELLOW):
        decision = decisions.decide(_state(), level, now=NOW)
        assert decision.action == "none"


def test_first_orange_alerts() -> None:
    decision = decisions.decide(_state(), Level.ORANGE, now=NOW)
    assert decision.action == "alert"
    assert decision.level is Level.ORANGE


def test_same_level_inside_cooldown_is_suppressed() -> None:
    state = _state(Level.ORANGE, last_alert_at=NOW - timedelta(hours=2), alerted_level=Level.ORANGE)
    decision = decisions.decide(state, Level.ORANGE, now=NOW)
    assert decision.action == "none"
    assert decision.cooldown_remaining_h == pytest.approx(4.0)


def test_same_level_after_cooldown_alerts_again() -> None:
    state = _state(Level.ORANGE, last_alert_at=NOW - timedelta(hours=7), alerted_level=Level.ORANGE)
    assert decisions.decide(state, Level.ORANGE, now=NOW).action == "alert"


def test_escalation_alerts_inside_cooldown() -> None:
    state = _state(Level.ORANGE, last_alert_at=NOW - timedelta(hours=1), alerted_level=Level.ORANGE)
    decision = decisions.decide(state, Level.RED, now=NOW)
    assert decision.action == "alert"
    assert decision.reason.startswith("escalation")


def test_drop_below_orange_gives_exactly_one_all_clear() -> None:
    state = _state(Level.ORANGE, last_alert_at=NOW, alerted_level=Level.ORANGE)
    first = decisions.decide(state, Level.GREEN, now=NOW + timedelta(hours=1))
    assert first.action == "all_clear"
    quiet = decisions.AreaState(area_id="swat_kalam", level=Level.GREEN, all_clear_sent=True)
    second = decisions.decide(quiet, Level.GREEN, now=NOW + timedelta(hours=2))
    assert second.action == "none"


def test_areas_are_processed_in_sorted_order() -> None:
    decisions_out, _ = decisions.apply_decisions(
        {}, {"swat_madyan": Level.ORANGE, "chitral_chitral": Level.RED}, now=NOW
    )
    assert [item.area_id for item in decisions_out] == ["chitral_chitral", "swat_madyan"]


def test_store_survives_a_missing_file(tmp_path) -> None:
    store = decisions.StateStore(tmp_path / "missing.json")
    assert store.load() == {}
    saved = store.save({"swat_kalam": _state(Level.ORANGE, last_alert_at=NOW)}, updated_at=NOW)
    assert saved.exists()
    loaded = store.load()
    assert loaded["swat_kalam"].level is Level.ORANGE


def test_store_write_is_atomic(tmp_path) -> None:
    path = tmp_path / "area_state.json"
    store = decisions.StateStore(path)
    store.save({"swat_kalam": _state(Level.RED)}, updated_at=NOW)
    assert not path.with_suffix(".json.tmp").exists()
    payload = path.read_text(encoding="utf-8")
    assert '"schema_version": 1' in payload


def test_corrupt_basin_does_not_poison_the_store(tmp_path) -> None:
    path = tmp_path / "area_state.json"
    path.write_text(
        '{"schema_version": 1, "areas": {"good": {"level": "ORANGE"}, "bad": {"level": "PURPLE"}}}',
        encoding="utf-8",
    )
    loaded = decisions.StateStore(path).load()
    assert set(loaded) == {"good"}


def test_missing_area_state_is_created() -> None:
    _, updated = decisions.apply_decisions({}, {"swat_kalam": Level.ORANGE}, now=NOW)
    assert updated["swat_kalam"].level is Level.ORANGE
    assert updated["swat_kalam"].last_alert_at == NOW
