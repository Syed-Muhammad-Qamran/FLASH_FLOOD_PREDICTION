"""Delivery gating: three locks, dry-run payloads, cap and verify blocking."""

from __future__ import annotations

import json

import pytest

from floodwatch import messages
from floodwatch.channels import router, whatsapp
from floodwatch.interop import RiskResult
from floodwatch.tiers import Level

ENV_OPEN = {
    "FLOODWATCH_SEND": "1",
    "WHATSAPP_TOKEN": "token-value",
    "WHATSAPP_PHONE_NUMBER_ID": "123",
    "WHATSAPP_TO": "92300",
}


def _result(area_id: str = "swat_kalam", tier: str = "severe") -> RiskResult:
    return RiskResult(
        area_id=area_id,
        score=0.62,
        level=Level.RED if tier == "severe" else Level.ORANGE,
        reasons=["rain_3h (12.5)"],
        computed_at="2022-08-27T00:00:00+00:00",
        tier=tier,
    )


def _text(result: RiskResult) -> str:
    return messages.format_alert(result)


def test_gate_requires_all_three_locks() -> None:
    allowed, reasons = router.gate(
        "swat_kalam", cfg={"delivery": "file"}, environ={}, overrides={"kill": False, "basins": {}}
    )
    assert allowed is False
    assert any("lock1" in reason for reason in reasons)
    assert any("lock2" in reason for reason in reasons)

    allowed, reasons = router.gate(
        "swat_kalam",
        cfg={"delivery": "whatsapp"},
        environ={"FLOODWATCH_SEND": "1"},
        overrides={"kill": False, "basins": {}},
    )
    assert allowed is True, reasons


def test_kill_switch_blocks_every_basin() -> None:
    allowed, reasons = router.gate(
        "swat_kalam",
        cfg={"delivery": "whatsapp"},
        environ={"FLOODWATCH_SEND": "1"},
        overrides={"kill": True, "basins": {}},
    )
    assert allowed is False
    assert any("global kill switch" in reason for reason in reasons)


def test_basin_suppression_blocks_only_that_basin() -> None:
    overrides = {"kill": False, "basins": {"swat_kalam": {"suppress": True}}}
    blocked, reasons = router.gate(
        "swat_kalam",
        cfg={"delivery": "whatsapp"},
        environ={"FLOODWATCH_SEND": "1"},
        overrides=overrides,
    )
    open_, _ = router.gate(
        "chitral_drosh",
        cfg={"delivery": "whatsapp"},
        environ={"FLOODWATCH_SEND": "1"},
        overrides=overrides,
    )
    assert blocked is False and any("suppressed" in reason for reason in reasons)
    assert open_ is True


def test_file_mode_records_without_opening_a_socket(monkeypatch, tmp_path) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("file mode must never send")

    monkeypatch.setattr(router.whatsapp, "send_text", explode)
    result = _result()
    report = router.dispatch(
        [(result, "alert", _text(result))],
        cfg={"delivery": "file"},
        environ={},
        overrides={"kill": False, "basins": {}},
        out_dir=tmp_path,
    )
    assert report.recorded == 1
    assert report.sent == 0
    assert report.messages[0].blocked is False


def test_dry_run_writes_the_exact_payload_and_no_socket(monkeypatch, tmp_path) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("dry-run must never send")

    monkeypatch.setattr(router.whatsapp, "send_text", explode)
    result = _result()
    report = router.dispatch(
        [(result, "alert", _text(result))],
        cfg={"delivery": "whatsapp"},
        environ=ENV_OPEN,
        overrides={"kill": False, "basins": {}},
        dry_run=True,
        out_dir=tmp_path,
    )
    assert report.attempted == 1
    assert report.sent == 0
    assert report.payload_path
    payload = json.loads(open(report.payload_path, encoding="utf-8").read())
    assert payload["messages"][0]["type"] == "text"
    assert payload["messages"][0]["to"] == "92300"
    assert payload["messages"][0]["text"]["body"] == _text(result)
    assert "token-value" not in open(report.payload_path, encoding="utf-8").read()


def test_whatsapp_without_send_env_sends_nothing(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("lock 2 is closed")

    monkeypatch.setattr(router.whatsapp, "send_text", explode)
    result = _result()
    report = router.dispatch(
        [(result, "alert", _text(result))],
        cfg={"delivery": "whatsapp"},
        environ=ENV_OPEN | {"FLOODWATCH_SEND": "0"},
        overrides={"kill": False, "basins": {}},
    )
    assert report.sent == 0
    assert report.blocked == 1


def test_whatsapp_success_reaches_the_sender(monkeypatch) -> None:
    sent: list[str] = []

    def fake_send(text, cfg):
        sent.append(text)
        return {"messages": [{"id": "1"}]}

    monkeypatch.setattr(router.whatsapp, "send_text", fake_send)
    result = _result()
    text = _text(result)
    report = router.dispatch(
        [(result, "alert", text)],
        cfg={"delivery": "whatsapp"},
        environ=ENV_OPEN,
        overrides={"kill": False, "basins": {}},
    )
    assert report.sent == 1
    assert sent == [text]


def test_a_provider_failure_is_recorded_and_the_run_continues(monkeypatch) -> None:
    def fail(text, cfg):
        raise whatsapp.WhatsAppError("WhatsApp send failed: HTTPError")

    monkeypatch.setattr(router.whatsapp, "send_text", fail)
    result = _result()
    report = router.dispatch(
        [(result, "alert", _text(result))],
        cfg={"delivery": "whatsapp"},
        environ=ENV_OPEN,
        overrides={"kill": False, "basins": {}},
    )
    assert report.failed == 1
    assert report.sent == 0
    assert "HTTPError" in report.messages[0].reasons[0]


def test_cap_of_five_applies_to_a_broadcast(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("over-cap must not send")

    monkeypatch.setattr(router.whatsapp, "send_text", explode)
    result = _result()
    items = [(result, "alert", _text(result))] * (router.MAX_MESSAGES_PER_RUN + 1)
    report = router.dispatch(
        items,
        cfg={"delivery": "whatsapp"},
        environ=ENV_OPEN,
        overrides={"kill": False, "basins": {}},
    )
    assert report.sent == 0
    assert report.skipped == len(items)
    assert any("hard cap exceeded" in problem for problem in report.problems)


def test_a_blocked_draft_never_reaches_the_sender(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("a blocked draft must never be sent")

    monkeypatch.setattr(router.whatsapp, "send_text", explode)
    result = _result()
    bad_text = "Basin: swat_kalam\nLevel: RED\nRisk score: 0.999\nHotline: Rescue 1122"
    report = router.dispatch(
        [(result, "alert", bad_text)],
        cfg={"delivery": "whatsapp"},
        environ=ENV_OPEN,
        overrides={"kill": False, "basins": {}},
    )
    assert report.blocked == 1
    assert report.attempted == 0
    assert report.messages[0].problems


def test_payload_builder_keeps_the_text_unchanged() -> None:
    body = whatsapp.build_payload("hello", to="92300")
    assert body["type"] == "text"
    assert body["text"]["body"] == "hello"
    assert body["messaging_product"] == "whatsapp"


def test_config_from_env_never_echoes_the_token() -> None:
    cfg = whatsapp.WhatsAppConfig.from_env(ENV_OPEN)
    assert cfg.token == "token-value"
    assert cfg.redacted()["token"] == "***redacted***"


def test_transport_errors_do_not_leak_the_token(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise whatsapp.urllib.error.URLError("connection reset")

    monkeypatch.setattr(whatsapp.urllib.request, "urlopen", boom)
    cfg = whatsapp.WhatsAppConfig.from_env(ENV_OPEN)
    with pytest.raises(whatsapp.WhatsAppError) as excinfo:
        whatsapp.send_text("hello", cfg)
    assert "token-value" not in str(excinfo.value)
    assert "URLError" in str(excinfo.value)
