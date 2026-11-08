"""WhatsApp Cloud API text sender (standard library only).

Credentials come from the environment and are never logged, persisted in the
payload file or copied into a run report.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

API_VERSION = "v21.0"
GRAPH_URL = "https://graph.facebook.com"


class WhatsAppError(RuntimeError):
    pass


@dataclass(frozen=True)
class WhatsAppConfig:
    token: str
    phone_number_id: str
    to: str
    api_version: str = API_VERSION
    timeout_s: float = 15.0

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> WhatsAppConfig:
        env = environ or dict(os.environ)
        missing = [
            name
            for name in ("WHATSAPP_TOKEN", "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_TO")
            if not env.get(name)
        ]
        if missing:
            raise WhatsAppError(f"missing WhatsApp environment variables: {missing}")
        return cls(
            token=env["WHATSAPP_TOKEN"],
            phone_number_id=env["WHATSAPP_PHONE_NUMBER_ID"],
            to=env["WHATSAPP_TO"],
        )

    def redacted(self) -> dict[str, Any]:
        return {
            "phone_number_id": self.phone_number_id,
            "to": self.to,
            "api_version": self.api_version,
            "token": "***redacted***",
        }


def build_payload(text: str, *, to: str) -> dict[str, Any]:
    """The exact provider request body for a text message."""
    return {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }


def send_text(text: str, cfg: WhatsAppConfig) -> dict[str, Any]:
    """POST one text message. Never called by ``--dry-run``."""
    body = json.dumps(build_payload(text, to=cfg.to)).encode("utf-8")
    url = f"{GRAPH_URL}/{cfg.api_version}/{cfg.phone_number_id}/messages"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {cfg.token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout_s) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        # Never include the token or full request headers in the error.
        raise WhatsAppError(f"WhatsApp send failed: {type(exc).__name__}") from exc


__all__ = [
    "API_VERSION",
    "WhatsAppConfig",
    "WhatsAppError",
    "build_payload",
    "send_text",
]
