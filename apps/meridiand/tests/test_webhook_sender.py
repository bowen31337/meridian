"""
Webhook sender conformance suite.

Tests cover:
  - deliver_webhook_event POSTs to the webhook URL.
  - POST Content-Type header is application/json.
  - POST payload contains webhook_id, session_id, event.
  - Event payload includes seq, ts, type, data fields.
  - Event thread_id included when present.
  - Delivery succeeds on HTTP 2xx.
  - X-Meridian-Signature header present when secret resolved.
  - X-Meridian-Signature has "sha256=" prefix.
  - X-Meridian-Signature value is HMAC-SHA256 of payload bytes.
  - No X-Meridian-Signature when secret_ref is None.
  - No X-Meridian-Signature when resolver returns None.
  - Audit entry "webhook.sender.delivered" written on success.
  - Audit entry level is "info" on success.
  - Audit detail has webhook_id, delivery_id, session_id, event_seq, event_type.
  - Delivery_id in audit detail has "delivery_" prefix.
  - OTel span "webhook.sender.deliver" emitted on success.
  - Span carries webhook.id, webhook.url, webhook.delivery_id, session.id, event.seq, event.type.
  - On 5xx response, retries up to max_retries.
  - On 5xx response with retries exhausted, raises WebhookDeliveryError.
  - On 4xx response, does not retry (single attempt then fail).
  - On network RequestError, retries.
  - Exponential backoff: delays double each retry.
  - Linear backoff: delays increase linearly each retry.
  - After successful retry, no DLQ entry written.
  - DLQ file written to storage_root/webhooks/dlq/{webhook_id}/{delivery_id}.json on failure.
  - DLQ record contains delivery_id, webhook_id, session_id, event, attempts, last_error, failed_at.
  - DLQ delivery_id has "delivery_" prefix.
  - Audit entry "webhook.sender.failed" written on permanent failure.
  - Audit entry level is "error" on permanent failure.
  - Audit detail has webhook_id, delivery_id, session_id, event_seq, event_type, message on failure.
  - OTel span has ERROR status on permanent failure.
  - run_webhook_sender_loop delivers events from matching sessions.
  - run_webhook_sender_loop skips events that don't match filter types.
  - run_webhook_sender_loop only scans the filtered session_id when set.
  - run_webhook_sender_loop persists watermark after delivery.
  - run_webhook_sender_loop skips events already before the watermark.
  - run_webhook_sender_loop skips inactive webhooks.
  - Watermark stored at storage_root/webhooks/watermarks/{webhook_id}.json.
  - DLQ event: watermark still advances past permanently-failed event.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from meridiand._audit import FileAuditLog
from meridiand._webhook_sender import (
    NoopSecretResolver,
    WebhookDeliveryError,
    _backoff_delay,
    _sign_payload,
    deliver_webhook_event,
    run_webhook_sender_loop,
)
import pytest
from storage_event_log import SessionEvent

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_webhook(
    webhook_id: str = "webhook_abc",
    url: str = "https://example.com/hook",
    max_retries: int = 0,
    backoff: str = "exponential",
    event_types: list[str] | None = None,
    session_id: str | None = None,
    secret_ref: str | None = None,
) -> dict[str, Any]:
    return {
        "id": webhook_id,
        "name": "test",
        "url": url,
        "secret_ref": secret_ref,
        "event_filter": {
            "types": event_types or ["session.created"],
            "session_id": session_id,
        },
        "max_retries": max_retries,
        "backoff": backoff,
        "status": "active",
    }


def _make_event(
    seq: int = 0,
    event_type: str = "session.created",
    data: dict | None = None,
    thread_id: str | None = None,
) -> SessionEvent:
    return SessionEvent(
        seq=seq,
        ts="2026-01-01T00:00:00Z",
        type=event_type,
        data=data or {"phase": "init"},
        thread_id=thread_id,
    )


class _FixedSecretResolver:
    """Resolves a single known secret_ref."""

    def __init__(self, ref: str, secret: str) -> None:
        self._ref = ref
        self._secret = secret

    def resolve(self, secret_ref: str) -> str | None:
        return self._secret if secret_ref == self._ref else None


class _MockTransport(httpx.AsyncBaseTransport):
    """Records requests; returns responses from a configured sequence."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return httpx.Response(200)


def _make_client(*responses: httpx.Response) -> tuple[httpx.AsyncClient, _MockTransport]:
    transport = _MockTransport(list(responses))
    client = httpx.AsyncClient(transport=transport)
    return client, transport


def _dlq_records(storage_root: Path, webhook_id: str) -> list[dict]:
    dlq_dir = storage_root / "webhooks" / "dlq" / webhook_id
    if not dlq_dir.exists():
        return []
    return [json.loads(f.read_text()) for f in sorted(dlq_dir.glob("delivery_*.json"))]


def _watermarks(storage_root: Path, webhook_id: str) -> dict:
    f = storage_root / "webhooks" / "watermarks" / f"{webhook_id}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _write_webhook(storage_root: Path, webhook: dict) -> None:
    d = storage_root / "webhooks"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{webhook['id']}.json").write_text(json.dumps(webhook))


def _write_event(storage_root: Path, session_id: str, event: SessionEvent) -> None:
    events_dir = storage_root / "events" / "2026" / "01" / "01"
    events_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "seq": event.seq,
        "ts": event.ts,
        "type": event.type,
        "data": event.data,
    }
    if event.thread_id is not None:
        record["thread_id"] = event.thread_id
    with (events_dir / f"{session_id}.ndjson").open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Unit: _sign_payload
# ---------------------------------------------------------------------------


class TestSignPayload:
    def test_returns_hex_string(self) -> None:
        sig = _sign_payload(b"payload", "secret")
        assert all(c in "0123456789abcdef" for c in sig)

    def test_matches_stdlib_hmac(self) -> None:
        payload = b'{"foo":"bar"}'
        secret = "mysecret"
        expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert _sign_payload(payload, secret) == expected

    def test_different_secrets_differ(self) -> None:
        assert _sign_payload(b"data", "s1") != _sign_payload(b"data", "s2")

    def test_different_payloads_differ(self) -> None:
        assert _sign_payload(b"a", "s") != _sign_payload(b"b", "s")


# ---------------------------------------------------------------------------
# Unit: _backoff_delay
# ---------------------------------------------------------------------------


class TestBackoffDelay:
    def test_exponential_attempt1(self) -> None:
        assert _backoff_delay("exponential", 1) == pytest.approx(1.0)

    def test_exponential_attempt2(self) -> None:
        assert _backoff_delay("exponential", 2) == pytest.approx(2.0)

    def test_exponential_attempt3(self) -> None:
        assert _backoff_delay("exponential", 3) == pytest.approx(4.0)

    def test_linear_attempt1(self) -> None:
        assert _backoff_delay("linear", 1) == pytest.approx(1.0)

    def test_linear_attempt2(self) -> None:
        assert _backoff_delay("linear", 2) == pytest.approx(2.0)

    def test_linear_attempt3(self) -> None:
        assert _backoff_delay("linear", 3) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# deliver_webhook_event: success
# ---------------------------------------------------------------------------


class TestDeliverWebhookSuccess:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_posts_to_webhook_url(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        assert len(transport.requests) == 1
        assert str(transport.requests[0].url) == "https://example.com/hook"

    async def test_post_method(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        assert transport.requests[0].method == "POST"

    async def test_content_type_header(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        assert transport.requests[0].headers["content-type"] == "application/json"

    async def test_payload_contains_webhook_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook(webhook_id="webhook_xyz")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        body = json.loads(transport.requests[0].content)
        assert body["webhook_id"] == "webhook_xyz"

    async def test_payload_contains_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-42",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        body = json.loads(transport.requests[0].content)
        assert body["session_id"] == "sess-42"

    async def test_payload_event_has_seq(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event(seq=7)
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        body = json.loads(transport.requests[0].content)
        assert body["event"]["seq"] == 7

    async def test_payload_event_has_type(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event(event_type="tool_call.result")
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        body = json.loads(transport.requests[0].content)
        assert body["event"]["type"] == "tool_call.result"

    async def test_payload_event_includes_thread_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event(thread_id="thread-99")
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        body = json.loads(transport.requests[0].content)
        assert body["event"]["thread_id"] == "thread-99"

    async def test_payload_event_omits_thread_id_when_none(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event(thread_id=None)
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        body = json.loads(transport.requests[0].content)
        assert "thread_id" not in body["event"]


# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------


class TestHmacSignature:
    async def test_signature_header_present_when_secret_resolved(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook(secret_ref="ref://my-secret")
        event = _make_event()
        resolver = _FixedSecretResolver("ref://my-secret", "supersecret")
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=resolver,
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        assert "x-meridian-signature" in transport.requests[0].headers

    async def test_signature_starts_with_sha256(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook(secret_ref="ref://my-secret")
        event = _make_event()
        resolver = _FixedSecretResolver("ref://my-secret", "supersecret")
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=resolver,
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        sig = transport.requests[0].headers["x-meridian-signature"]
        assert sig.startswith("sha256=")

    async def test_signature_is_correct_hmac(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook(secret_ref="ref://my-secret")
        event = _make_event()
        resolver = _FixedSecretResolver("ref://my-secret", "supersecret")
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=resolver,
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        req = transport.requests[0]
        body_bytes = req.content
        expected_hex = hmac.new(b"supersecret", body_bytes, hashlib.sha256).hexdigest()
        sig = req.headers["x-meridian-signature"]
        assert sig == f"sha256={expected_hex}"

    async def test_no_signature_header_when_secret_ref_none(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook(secret_ref=None)
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        assert "x-meridian-signature" not in transport.requests[0].headers

    async def test_no_signature_header_when_resolver_returns_none(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(200))
        webhook = _make_webhook(secret_ref="ref://unresolvable")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        assert "x-meridian-signature" not in transport.requests[0].headers


# ---------------------------------------------------------------------------
# Audit log on success
# ---------------------------------------------------------------------------


class TestDeliverAuditSuccess:
    async def test_writes_delivered_audit_entry(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        records = _audit_records(storage_root)
        assert any(r.get("event") == "webhook.sender.delivered" for r in records)

    async def test_audit_level_is_info(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.sender.delivered"
        )
        assert record["level"] == "info"

    async def test_audit_detail_has_webhook_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook(webhook_id="webhook_detail")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.sender.delivered"
        )
        assert record["detail"]["webhook_id"] == "webhook_detail"

    async def test_audit_detail_has_delivery_id_prefix(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.sender.delivered"
        )
        assert record["detail"]["delivery_id"].startswith("delivery_")

    async def test_audit_detail_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-77",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.sender.delivered"
        )
        assert record["detail"]["session_id"] == "sess-77"

    async def test_audit_detail_has_event_seq(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event(seq=5)
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.sender.delivered"
        )
        assert record["detail"]["event_seq"] == 5

    async def test_audit_detail_has_event_type(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event(event_type="message.added")
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.sender.delivered"
        )
        assert record["detail"]["event_type"] == "message.added"


# ---------------------------------------------------------------------------
# OTel on success
# ---------------------------------------------------------------------------


class TestDeliverOtelSuccess:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_emits_deliver_span(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "webhook.sender.deliver" in span_names

    async def test_span_has_webhook_id_attribute(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook(webhook_id="webhook_otel")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "webhook.sender.deliver"
        )
        assert span.attributes["webhook.id"] == "webhook_otel"

    async def test_span_has_webhook_url_attribute(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook(url="https://otel.example.com/hook")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "webhook.sender.deliver"
        )
        assert span.attributes["webhook.url"] == "https://otel.example.com/hook"

    async def test_span_has_delivery_id_attribute(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "webhook.sender.deliver"
        )
        assert span.attributes["webhook.delivery_id"].startswith("delivery_")

    async def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-span",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "webhook.sender.deliver"
        )
        assert span.attributes["session.id"] == "sess-span"

    async def test_span_has_event_seq_attribute(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event(seq=3)
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "webhook.sender.deliver"
        )
        assert span.attributes["event.seq"] == 3

    async def test_span_has_event_type_attribute(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(200))
        webhook = _make_webhook()
        event = _make_event(event_type="model_call.completed")
        dlq_dir = storage_root / "webhooks" / "dlq"

        await deliver_webhook_event(
            webhook,
            event,
            "sess-1",
            client=client,
            secret_resolver=NoopSecretResolver(),
            audit_log=audit,
            dlq_dir=dlq_dir,
        )

        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "webhook.sender.deliver"
        )
        assert span.attributes["event.type"] == "model_call.completed"


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetryBehaviour:
    async def test_retries_on_5xx(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(
            httpx.Response(500),
            httpx.Response(200),
        )
        webhook = _make_webhook(max_retries=1, backoff="exponential")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with patch("meridiand._webhook_sender.asyncio.sleep", new_callable=AsyncMock):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        assert len(transport.requests) == 2

    async def test_does_not_retry_on_4xx(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, transport = _make_client(httpx.Response(400))
        webhook = _make_webhook(max_retries=3, backoff="exponential")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with (
            pytest.raises(WebhookDeliveryError),
            patch("meridiand._webhook_sender.asyncio.sleep", new_callable=AsyncMock),
        ):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        assert len(transport.requests) == 1

    async def test_retries_on_network_error(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _MockTransport([])
        call_count = 0

        async def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        webhook = _make_webhook(max_retries=2, backoff="linear")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with patch("meridiand._webhook_sender.asyncio.sleep", new_callable=AsyncMock):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        assert call_count == 3

    async def test_raises_after_all_retries_exhausted(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(
            httpx.Response(503),
            httpx.Response(503),
        )
        webhook = _make_webhook(max_retries=1, backoff="exponential")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with (
            pytest.raises(WebhookDeliveryError),
            patch("meridiand._webhook_sender.asyncio.sleep", new_callable=AsyncMock),
        ):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

    async def test_success_after_retry_no_dlq(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(
            httpx.Response(500),
            httpx.Response(201),
        )
        webhook = _make_webhook(webhook_id="webhook_retry", max_retries=1)
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with patch("meridiand._webhook_sender.asyncio.sleep", new_callable=AsyncMock):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        assert _dlq_records(storage_root, "webhook_retry") == []

    async def test_exponential_backoff_delays(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(500),
        )
        webhook = _make_webhook(max_retries=2, backoff="exponential")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"
        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with (
            patch("meridiand._webhook_sender.asyncio.sleep", side_effect=_fake_sleep),
            pytest.raises(WebhookDeliveryError),
        ):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        assert sleep_calls == pytest.approx([1.0, 2.0])

    async def test_linear_backoff_delays(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(500),
        )
        webhook = _make_webhook(max_retries=2, backoff="linear")
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"
        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with (
            patch("meridiand._webhook_sender.asyncio.sleep", side_effect=_fake_sleep),
            pytest.raises(WebhookDeliveryError),
        ):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        assert sleep_calls == pytest.approx([1.0, 2.0])


# ---------------------------------------------------------------------------
# Dead-letter queue
# ---------------------------------------------------------------------------


class TestDeadLetterQueue:
    async def test_dlq_file_written_on_failure(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook(webhook_id="webhook_dlq", max_retries=0)
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        assert len(_dlq_records(storage_root, "webhook_dlq")) == 1

    async def test_dlq_path_under_webhook_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook(webhook_id="webhook_path", max_retries=0)
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        assert (storage_root / "webhooks" / "dlq" / "webhook_path").is_dir()

    async def test_dlq_record_has_delivery_id_prefix(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook(webhook_id="webhook_rec", max_retries=0)
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        record = _dlq_records(storage_root, "webhook_rec")[0]
        assert record["delivery_id"].startswith("delivery_")

    async def test_dlq_record_has_webhook_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook(webhook_id="webhook_wid", max_retries=0)
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        record = _dlq_records(storage_root, "webhook_wid")[0]
        assert record["webhook_id"] == "webhook_wid"

    async def test_dlq_record_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook(webhook_id="webhook_sid", max_retries=0)
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-dlq",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        record = _dlq_records(storage_root, "webhook_sid")[0]
        assert record["session_id"] == "sess-dlq"

    async def test_dlq_record_has_event(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook(webhook_id="webhook_ev", max_retries=0)
        event = _make_event(seq=9, event_type="error")
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        record = _dlq_records(storage_root, "webhook_ev")[0]
        assert record["event"]["seq"] == 9
        assert record["event"]["type"] == "error"

    async def test_dlq_record_has_attempts(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(500),
        )
        webhook = _make_webhook(webhook_id="webhook_att", max_retries=2)
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with (
            patch("meridiand._webhook_sender.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(WebhookDeliveryError),
        ):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        record = _dlq_records(storage_root, "webhook_att")[0]
        assert record["attempts"] == 3

    async def test_dlq_record_has_last_error(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(503))
        webhook = _make_webhook(webhook_id="webhook_err", max_retries=0)
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        record = _dlq_records(storage_root, "webhook_err")[0]
        assert "503" in record["last_error"]

    async def test_dlq_record_has_failed_at(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook(webhook_id="webhook_fat", max_retries=0)
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        record = _dlq_records(storage_root, "webhook_fat")[0]
        assert "failed_at" in record
        assert len(record["failed_at"]) > 0

    async def test_multiple_failures_create_separate_dlq_entries(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        dlq_dir = storage_root / "webhooks" / "dlq"
        webhook = _make_webhook(webhook_id="webhook_multi", max_retries=0)

        for seq in (0, 1):
            client, _ = _make_client(httpx.Response(500))
            event = _make_event(seq=seq)
            with pytest.raises(WebhookDeliveryError):
                await deliver_webhook_event(
                    webhook,
                    event,
                    "sess-1",
                    client=client,
                    secret_resolver=NoopSecretResolver(),
                    audit_log=audit,
                    dlq_dir=dlq_dir,
                )

        assert len(_dlq_records(storage_root, "webhook_multi")) == 2


# ---------------------------------------------------------------------------
# Audit log on failure
# ---------------------------------------------------------------------------


class TestDeliverAuditFailure:
    async def test_writes_failed_audit_entry(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        records = _audit_records(storage_root)
        assert any(r.get("event") == "webhook.sender.failed" for r in records)

    async def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.sender.failed"
        )
        assert record["level"] == "error"

    async def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "webhook.sender.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel on failure
# ---------------------------------------------------------------------------


class TestDeliverOtelFailure:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_emits_span_on_failure(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "webhook.sender.deliver" in span_names

    async def test_span_has_error_status_on_failure(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        audit = FileAuditLog(storage_root)
        client, _ = _make_client(httpx.Response(500))
        webhook = _make_webhook()
        event = _make_event()
        dlq_dir = storage_root / "webhooks" / "dlq"

        with pytest.raises(WebhookDeliveryError):
            await deliver_webhook_event(
                webhook,
                event,
                "sess-1",
                client=client,
                secret_resolver=NoopSecretResolver(),
                audit_log=audit,
                dlq_dir=dlq_dir,
            )

        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "webhook.sender.deliver"
        )
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------


class TestWebhookSenderLoop:
    async def test_loop_delivers_matching_events(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        webhook = _make_webhook(
            webhook_id="webhook_loop",
            event_types=["session.created"],
        )
        _write_webhook(storage_root, webhook)
        event = _make_event(seq=0, event_type="session.created")
        _write_event(storage_root, "sess-loop", event)

        received: list[dict] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.content))
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

        async def _run_once() -> None:
            loop_task = asyncio.create_task(
                run_webhook_sender_loop(
                    storage_root, audit, check_interval_seconds=9999, _http_client=client
                )
            )
            await asyncio.sleep(0.1)
            loop_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await loop_task

        await _run_once()
        assert len(received) == 1
        assert received[0]["session_id"] == "sess-loop"

    async def test_loop_skips_non_matching_event_types(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        webhook = _make_webhook(
            webhook_id="webhook_filter",
            event_types=["session.created"],
        )
        _write_webhook(storage_root, webhook)
        # Write a message.added event — should not be delivered
        _write_event(storage_root, "sess-x", _make_event(seq=0, event_type="message.added"))

        received: list[dict] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.content))
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

        loop_task = asyncio.create_task(
            run_webhook_sender_loop(
                storage_root, audit, check_interval_seconds=9999, _http_client=client
            )
        )
        await asyncio.sleep(0.1)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        assert received == []

    async def test_loop_filters_by_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        webhook = _make_webhook(
            webhook_id="webhook_sess_filter",
            event_types=["session.created"],
            session_id="sess-wanted",
        )
        _write_webhook(storage_root, webhook)
        _write_event(storage_root, "sess-wanted", _make_event(seq=0, event_type="session.created"))
        _write_event(storage_root, "sess-other", _make_event(seq=0, event_type="session.created"))

        received: list[dict] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.content))
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

        loop_task = asyncio.create_task(
            run_webhook_sender_loop(
                storage_root, audit, check_interval_seconds=9999, _http_client=client
            )
        )
        await asyncio.sleep(0.1)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        assert all(r["session_id"] == "sess-wanted" for r in received)
        assert len(received) == 1

    async def test_loop_persists_watermark(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        webhook = _make_webhook(
            webhook_id="webhook_wm",
            event_types=["session.created"],
        )
        _write_webhook(storage_root, webhook)
        _write_event(storage_root, "sess-wm", _make_event(seq=0, event_type="session.created"))

        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

        loop_task = asyncio.create_task(
            run_webhook_sender_loop(
                storage_root, audit, check_interval_seconds=9999, _http_client=client
            )
        )
        await asyncio.sleep(0.1)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        wm = _watermarks(storage_root, "webhook_wm")
        assert wm.get("sess-wm") == 0

    async def test_loop_does_not_redeliver_past_watermark(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        webhook = _make_webhook(
            webhook_id="webhook_no_redeliver",
            event_types=["session.created"],
        )
        _write_webhook(storage_root, webhook)
        _write_event(storage_root, "sess-nr", _make_event(seq=0, event_type="session.created"))

        # Pre-set watermark to 0 so seq=0 is already "delivered"
        wm_dir = storage_root / "webhooks" / "watermarks"
        wm_dir.mkdir(parents=True, exist_ok=True)
        (wm_dir / "webhook_no_redeliver.json").write_text(json.dumps({"sess-nr": 0}))

        received: list[dict] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.content))
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

        loop_task = asyncio.create_task(
            run_webhook_sender_loop(
                storage_root, audit, check_interval_seconds=9999, _http_client=client
            )
        )
        await asyncio.sleep(0.1)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        assert received == []

    async def test_loop_skips_inactive_webhooks(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        webhook = _make_webhook(
            webhook_id="webhook_inactive",
            event_types=["session.created"],
        )
        webhook["status"] = "inactive"
        _write_webhook(storage_root, webhook)
        _write_event(storage_root, "sess-ia", _make_event(seq=0, event_type="session.created"))

        received: list[dict] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.content))
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

        loop_task = asyncio.create_task(
            run_webhook_sender_loop(
                storage_root, audit, check_interval_seconds=9999, _http_client=client
            )
        )
        await asyncio.sleep(0.1)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        assert received == []

    async def test_loop_advances_watermark_after_dlq(self, storage_root: Path) -> None:
        """Permanently failed event is DLQ'd; watermark still advances past it."""
        audit = FileAuditLog(storage_root)
        webhook = _make_webhook(
            webhook_id="webhook_dlq_wm",
            event_types=["session.created"],
            max_retries=0,
        )
        _write_webhook(storage_root, webhook)
        _write_event(storage_root, "sess-dlqwm", _make_event(seq=0, event_type="session.created"))

        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

        loop_task = asyncio.create_task(
            run_webhook_sender_loop(
                storage_root, audit, check_interval_seconds=9999, _http_client=client
            )
        )
        await asyncio.sleep(0.1)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        # Watermark should be advanced even though event went to DLQ
        wm = _watermarks(storage_root, "webhook_dlq_wm")
        assert wm.get("sess-dlqwm") == 0

        # DLQ entry should be written
        assert len(_dlq_records(storage_root, "webhook_dlq_wm")) == 1
