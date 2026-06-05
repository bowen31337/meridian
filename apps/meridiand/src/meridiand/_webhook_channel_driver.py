"""
Generic Webhook Channel Driver: meridian.webhook

Inbound HMAC verification is applied in the system channel router (see
_system_channel.py): callers must supply X-Meridian-Signature: sha256=<hex>
when the channel config carries hmac_secret_ref.

Outbound delivers a signed JSON payload via HTTP POST to the URL in channel
config outbound_url.  The payload is signed with HMAC-SHA256 and attached in
X-Meridian-Signature when hmac_secret_ref resolves to a non-None secret.

Every outbound send emits an OTel span "webhook.channel.send" and attaches a
structured invocation event.  On failure the error is surfaced as ChannelFailure
and an audit-log entry is written before re-raising.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import uuid

from core_errors import (
    AuditLog,
    AuditLogEntry,
    NoopAuditLog,
    StructuredEvent,
    get_tracer,
    record_invocation_event,
)
import httpx
from opentelemetry.trace import StatusCode
from sdk_channel import (
    ChannelCapabilities,
    ChannelDriver,
    ChannelFailure,
    SendRequest,
    SendResult,
    StartRequest,
    StopRequest,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Return the HMAC-SHA256 hex digest of payload_bytes keyed by secret."""
    mac = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256)
    return mac.hexdigest()


@runtime_checkable
class SecretResolver(Protocol):
    """Resolves a secret_ref to the raw secret value, or None if unavailable."""

    def resolve(self, secret_ref: str) -> str | None: ...


class NoopSecretResolver:
    """Always returns None — no signing performed."""

    def resolve(self, secret_ref: str) -> str | None:
        return None


class WebhookChannelDriver(ChannelDriver):
    """
    Generic webhook channel driver (kind: meridian.webhook).

    Outbound: HTTP POST to channel config["outbound_url"].  Payload is a JSON
    object signed with HMAC-SHA256 via X-Meridian-Signature when the channel
    config carries hmac_secret_ref and the resolver returns a non-None value.

    Channel config fields:
        outbound_url     (required) Destination URL for outbound delivery.
        hmac_secret_ref  (optional) Secret ref; when resolved, outbound payload
                         is signed with X-Meridian-Signature: sha256=<hex>.
    """

    kind = "meridian.webhook"

    def __init__(
        self,
        *,
        storage_root: Path,
        secret_resolver: SecretResolver | None = None,
        audit_log: AuditLog | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._storage_root = storage_root
        self._resolver = secret_resolver or NoopSecretResolver()
        self._audit_log = audit_log or NoopAuditLog()
        self._http_client = http_client

    async def start(self, request: StartRequest) -> None:
        pass

    async def send(self, request: SendRequest) -> SendResult:
        channel_file = self._storage_root / "channels" / f"{request.channel_id}.json"
        if not channel_file.exists():
            raise ChannelFailure(
                code="CHAN_CONFIG_NOT_FOUND",
                message=f"Channel config not found for '{request.channel_id}'",
                channel_id=request.channel_id,
                channel_kind=request.channel_kind,
                session_id=request.session_id,
                timestamp=_now(),
            )

        channel_config: dict[str, Any] = json.loads(channel_file.read_text())
        driver_config: dict[str, Any] = channel_config.get("config", {})
        outbound_url: str | None = driver_config.get("outbound_url")

        if outbound_url is None:
            raise ChannelFailure(
                code="CHAN_OUTBOUND_URL_MISSING",
                message=f"Channel '{request.channel_id}' config missing outbound_url",
                channel_id=request.channel_id,
                channel_kind=request.channel_kind,
                session_id=request.session_id,
                timestamp=_now(),
            )

        hmac_secret_ref: str | None = driver_config.get("hmac_secret_ref")
        delivery_id = f"wh_{uuid.uuid4().hex}"
        now = _now()

        payload: dict[str, Any] = {
            "delivery_id": delivery_id,
            "channel_id": request.channel_id,
            "session_id": request.session_id,
            "recipient": request.recipient,
            "content": request.content,
            "content_type": request.content_type,
            "timestamp": now,
        }
        if request.thread_id is not None:
            payload["thread_id"] = request.thread_id

        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        secret: str | None = None
        if hmac_secret_ref is not None:
            secret = self._resolver.resolve(hmac_secret_ref)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if secret is not None:
            sig = _sign_payload(payload_bytes, secret)
            headers["X-Meridian-Signature"] = f"sha256={sig}"

        tracer = get_tracer()
        with tracer.start_as_current_span(
            "webhook.channel.send",
            attributes={
                "channel.id": request.channel_id,
                "channel.kind": request.channel_kind,
                "session.id": request.session_id,
                "webhook.delivery_id": delivery_id,
                "webhook.url": outbound_url,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="webhook.channel.send.invocation",
                    code="webhook_channel_send",
                    timestamp=now,
                ),
            )

            try:
                if self._http_client is not None:
                    response = await self._http_client.post(
                        outbound_url, content=payload_bytes, headers=headers
                    )
                else:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(
                            outbound_url, content=payload_bytes, headers=headers
                        )

                if not response.is_success:
                    raise RuntimeError(f"HTTP {response.status_code}")

            except ChannelFailure:
                raise
            except Exception as exc:
                failure = ChannelFailure(
                    code="CHAN_SEND_FAILED",
                    message=f"Webhook delivery to '{outbound_url}' failed: {exc}",
                    channel_id=request.channel_id,
                    channel_kind=request.channel_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    cause=exc,
                )
                span.set_status(StatusCode.ERROR, failure.message)
                span.record_exception(exc)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="webhook.channel.send.failed",
                        code=failure.code,
                        timestamp=failure.timestamp,
                        detail={
                            "channel_id": request.channel_id,
                            "session_id": request.session_id,
                            "delivery_id": delivery_id,
                            "url": outbound_url,
                            "message": failure.message,
                        },
                    )
                )
                raise failure from exc

        return SendResult(
            message_id=delivery_id,
            timestamp=now,
            delivered=True,
        )

    async def stop(self, request: StopRequest) -> None:
        pass

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(can_send_text=True)
