"""
Tests for the generic webhook channel driver (meridian.webhook).

Covers:
  - Outbound delivery: POST to outbound_url, HMAC signing, SendResult shape.
  - Outbound failures: missing config, HTTP errors; audit log written, ChannelFailure raised.
  - Inbound HMAC verification via POST /v1/channels/{id}/inbound: valid, invalid,
    missing signature, no hmac_secret_ref (pass-through).
  - OTel spans emitted on outbound send and inbound HMAC failure.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._webhook_channel_driver import (
    NoopSecretResolver,
    WebhookChannelDriver,
    _sign_payload,
)
from opentelemetry.trace import StatusCode
from sdk_channel import (
    ChannelCapabilities,
    ChannelFailure,
    ChannelRuntime,
    SendRequest,
    StartRequest,
    StopRequest,
)

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEBHOOK_KIND = "meridian.webhook"


class FixedSecretResolver:
    """Returns a fixed secret for any ref."""

    def __init__(self, secret: str) -> None:
        self._secret = secret

    def resolve(self, secret_ref: str) -> str | None:
        return self._secret


def _make_channel_file(
    storage_root: Path,
    *,
    channel_id: str = "ch_wh_1",
    outbound_url: str | None = "http://target.example/hook",
    hmac_secret_ref: str | None = None,
    inbound_policy: str = "open",
    egress_policy: str = "enabled",
) -> str:
    channels_dir = storage_root / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"token_vault_ref": "vaults/main/tok"}
    if outbound_url is not None:
        config["outbound_url"] = outbound_url
    if hmac_secret_ref is not None:
        config["hmac_secret_ref"] = hmac_secret_ref
    record: dict[str, Any] = {
        "id": channel_id,
        "kind": _WEBHOOK_KIND,
        "config": config,
        "inbound_policy": inbound_policy,
        "egress_policy": egress_policy,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    (channels_dir / f"{channel_id}.json").write_text(json.dumps(record))
    return channel_id


def _make_send_request(channel_id: str = "ch_wh_1") -> SendRequest:
    return SendRequest(
        channel_id=channel_id,
        channel_kind=_WEBHOOK_KIND,
        session_id="sess_test",
        recipient="ext-user-1",
        content="hello webhook",
    )


def _make_driver(
    storage_root: Path,
    *,
    secret: str | None = None,
    audit_log=None,
    http_client: httpx.AsyncClient | None = None,
) -> WebhookChannelDriver:
    resolver = FixedSecretResolver(secret) if secret is not None else NoopSecretResolver()
    return WebhookChannelDriver(
        storage_root=storage_root,
        secret_resolver=resolver,
        audit_log=audit_log,
        http_client=http_client,
    )


def _make_runtime(driver: WebhookChannelDriver) -> ChannelRuntime:
    rt = ChannelRuntime()
    rt.register(driver)
    return rt


def _make_client(
    storage_root: Path,
    *,
    driver: WebhookChannelDriver | None = None,
    secret_resolver=None,
) -> TestClient:
    d = driver or _make_driver(storage_root)
    rt = _make_runtime(d)
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        channel_runtime=rt,
        secret_resolver=secret_resolver,
    )
    return TestClient(app, raise_server_exceptions=False)


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hmac_signature(body: bytes, secret: str) -> str:
    digest = _sign_payload(body, secret)
    return f"sha256={digest}"


# ---------------------------------------------------------------------------
# Driver: capabilities
# ---------------------------------------------------------------------------


class TestDriverCapabilities:
    def test_kind_is_meridian_webhook(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root)
        assert driver.kind == "meridian.webhook"

    def test_capabilities_can_send_text(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert isinstance(caps, ChannelCapabilities)
        assert caps.can_send_text is True

    def test_start_and_stop_are_noops(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root)
        req = StartRequest(channel_id="c", channel_kind=_WEBHOOK_KIND, session_id="s")
        stop_req = StopRequest(channel_id="c", channel_kind=_WEBHOOK_KIND, session_id="s")

        import asyncio

        asyncio.get_event_loop().run_until_complete(driver.start(req))
        asyncio.get_event_loop().run_until_complete(driver.stop(stop_req))


# ---------------------------------------------------------------------------
# Driver: outbound delivery — happy path
# ---------------------------------------------------------------------------


class TestDriverSendSuccess:
    @pytest.fixture(autouse=True)
    def _otel_clear(self) -> None:
        _otel_exporter.clear()

    async def test_posts_to_outbound_url(self, storage_root: Path) -> None:
        captured: list[httpx.Request] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root, outbound_url="http://target.example/hook")
            driver = _make_driver(storage_root, http_client=client)
            result = await driver.send(_make_send_request())

        assert len(captured) == 1
        assert str(captured[0].url) == "http://target.example/hook"
        assert result.delivered is True

    async def test_result_has_message_id(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            result = await driver.send(_make_send_request())

        assert result.message_id.startswith("wh_")
        assert len(result.message_id) > 3

    async def test_result_has_timestamp(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            result = await driver.send(_make_send_request())

        assert isinstance(result.timestamp, str)
        assert len(result.timestamp) > 0

    async def test_payload_includes_channel_and_session(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root, channel_id="ch_abc")
            driver = _make_driver(storage_root, http_client=client)
            req = SendRequest(
                channel_id="ch_abc",
                channel_kind=_WEBHOOK_KIND,
                session_id="sess_xyz",
                recipient="user-1",
                content="hi",
            )
            await driver.send(req)

        payload = json.loads(bodies[0])
        assert payload["channel_id"] == "ch_abc"
        assert payload["session_id"] == "sess_xyz"
        assert payload["recipient"] == "user-1"
        assert payload["content"] == "hi"

    async def test_unsigned_when_no_secret(self, storage_root: Path) -> None:
        headers_seen: list[dict] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            headers_seen.append(dict(request.headers))
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            await driver.send(_make_send_request())

        assert "x-meridian-signature" not in headers_seen[0]

    async def test_signed_when_secret_configured(self, storage_root: Path) -> None:
        _SECRET = "mysecret"
        headers_seen: list[dict] = []
        body_seen: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            headers_seen.append(dict(request.headers))
            body_seen.append(request.content)
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root, hmac_secret_ref="vault/secret")
            driver = _make_driver(storage_root, secret=_SECRET, http_client=client)
            await driver.send(_make_send_request())

        sig = headers_seen[0].get("x-meridian-signature", "")
        assert sig.startswith("sha256=")
        expected = f"sha256={_sign_payload(body_seen[0], _SECRET)}"
        assert sig == expected

    async def test_emits_otel_span(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            await driver.send(_make_send_request())

        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "webhook.channel.send" in span_names

    async def test_span_has_channel_id_attribute(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root, channel_id="ch_span_test")
            driver = _make_driver(storage_root, http_client=client)
            req = SendRequest(
                channel_id="ch_span_test",
                channel_kind=_WEBHOOK_KIND,
                session_id="sess_s",
                recipient="r",
                content="c",
            )
            await driver.send(req)

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("webhook.channel.send")
        assert span is not None
        assert span.attributes["channel.id"] == "ch_span_test"

    async def test_span_has_invocation_event(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            await driver.send(_make_send_request())

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("webhook.channel.send")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names


# ---------------------------------------------------------------------------
# Driver: outbound delivery — failures
# ---------------------------------------------------------------------------


class TestDriverSendFailures:
    @pytest.fixture(autouse=True)
    def _otel_clear(self) -> None:
        _otel_exporter.clear()

    async def test_missing_outbound_url_raises_channel_failure(self, storage_root: Path) -> None:
        _make_channel_file(storage_root, outbound_url=None)
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(_make_send_request())
        assert exc_info.value.code == "CHAN_OUTBOUND_URL_MISSING"

    async def test_config_not_found_raises_channel_failure(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root)
        req = SendRequest(
            channel_id="ch_nonexistent",
            channel_kind=_WEBHOOK_KIND,
            session_id="s",
            recipient="r",
            content="c",
        )
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(req)
        assert exc_info.value.code == "CHAN_CONFIG_NOT_FOUND"

    async def test_http_error_raises_channel_failure(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            with pytest.raises(ChannelFailure) as exc_info:
                await driver.send(_make_send_request())
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_http_error_writes_audit_log(self, storage_root: Path) -> None:
        from core_errors import NoopAuditLog

        class CapturingAuditLog:
            def __init__(self) -> None:
                self.entries: list = []

            def write(self, entry) -> None:
                self.entries.append(entry)

        captured_log = CapturingAuditLog()

        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, audit_log=captured_log, http_client=client)
            with pytest.raises(ChannelFailure):
                await driver.send(_make_send_request())

        assert len(captured_log.entries) == 1
        entry = captured_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "webhook.channel.send.failed"

    async def test_http_error_span_marked_error(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            with pytest.raises(ChannelFailure):
                await driver.send(_make_send_request())

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("webhook.channel.send")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    async def test_network_error_raises_channel_failure(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            with pytest.raises(ChannelFailure) as exc_info:
                await driver.send(_make_send_request())
        assert exc_info.value.code == "CHAN_SEND_FAILED"


# ---------------------------------------------------------------------------
# Inbound HMAC verification via POST /v1/channels/{id}/inbound
# ---------------------------------------------------------------------------


_INBOUND_SECRET = "inbound-secret-42"
_INBOUND_SECRET_REF = "vault/inbound_hmac"


def _make_inbound_client(
    storage_root: Path,
    *,
    secret: str | None = None,
    channel_id: str = "ch_hmac",
    hmac_secret_ref: str | None = None,
    inbound_policy: str = "open",
) -> tuple[TestClient, str]:
    resolver = FixedSecretResolver(secret) if secret is not None else NoopSecretResolver()
    _make_channel_file(
        storage_root,
        channel_id=channel_id,
        hmac_secret_ref=hmac_secret_ref,
        inbound_policy=inbound_policy,
    )
    driver = WebhookChannelDriver(
        storage_root=storage_root,
        secret_resolver=resolver,
    )
    rt = ChannelRuntime()
    rt.register(driver)
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        channel_runtime=rt,
        secret_resolver=resolver,
    )
    client = TestClient(app, raise_server_exceptions=False)
    return client, channel_id


class TestInboundHmacVerification:
    @pytest.fixture(autouse=True)
    def _otel_clear(self) -> None:
        _otel_exporter.clear()

    def _body(self) -> bytes:
        return json.dumps({"sender_id": "ext-1", "content": "hi"}).encode()

    def test_valid_signature_accepted(self, storage_root: Path) -> None:
        client, channel_id = _make_inbound_client(
            storage_root,
            secret=_INBOUND_SECRET,
            hmac_secret_ref=_INBOUND_SECRET_REF,
        )
        body = self._body()
        sig = _hmac_signature(body, _INBOUND_SECRET)
        resp = client.post(
            f"/v1/channels/{channel_id}/inbound",
            content=body,
            headers={"Content-Type": "application/json", "X-Meridian-Signature": sig},
        )
        assert resp.status_code == 200

    def test_invalid_signature_rejected(self, storage_root: Path) -> None:
        client, channel_id = _make_inbound_client(
            storage_root,
            secret=_INBOUND_SECRET,
            hmac_secret_ref=_INBOUND_SECRET_REF,
        )
        body = self._body()
        resp = client.post(
            f"/v1/channels/{channel_id}/inbound",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Meridian-Signature": "sha256=deadbeef",
            },
        )
        assert resp.status_code == 401

    def test_invalid_signature_error_code(self, storage_root: Path) -> None:
        client, channel_id = _make_inbound_client(
            storage_root,
            secret=_INBOUND_SECRET,
            hmac_secret_ref=_INBOUND_SECRET_REF,
        )
        body = self._body()
        data = client.post(
            f"/v1/channels/{channel_id}/inbound",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Meridian-Signature": "sha256=deadbeef",
            },
        ).json()
        assert data["error"]["code"] == "channel_inbound_hmac_invalid"

    def test_missing_signature_rejected(self, storage_root: Path) -> None:
        client, channel_id = _make_inbound_client(
            storage_root,
            secret=_INBOUND_SECRET,
            hmac_secret_ref=_INBOUND_SECRET_REF,
        )
        body = self._body()
        resp = client.post(
            f"/v1/channels/{channel_id}/inbound",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    def test_no_hmac_secret_ref_no_verification(self, storage_root: Path) -> None:
        client, channel_id = _make_inbound_client(
            storage_root,
            secret=_INBOUND_SECRET,
            hmac_secret_ref=None,  # no HMAC configured
        )
        body = self._body()
        resp = client.post(
            f"/v1/channels/{channel_id}/inbound",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

    def test_hmac_failure_writes_audit(self, storage_root: Path) -> None:
        client, channel_id = _make_inbound_client(
            storage_root,
            secret=_INBOUND_SECRET,
            hmac_secret_ref=_INBOUND_SECRET_REF,
        )
        body = self._body()
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Meridian-Signature": "sha256=bad",
            },
        )
        records = _audit_records(storage_root)
        assert any(
            r.get("event") == "channel.inbound.failed"
            and r.get("code") == "channel_inbound_hmac_invalid"
            for r in records
        )

    def test_hmac_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client, channel_id = _make_inbound_client(
            storage_root,
            secret=_INBOUND_SECRET,
            hmac_secret_ref=_INBOUND_SECRET_REF,
        )
        body = self._body()
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Meridian-Signature": "sha256=bad",
            },
        )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "channel.inbound.failed"
        )
        assert record["level"] == "error"

    def test_hmac_failure_span_has_error_status(self, storage_root: Path) -> None:
        client, channel_id = _make_inbound_client(
            storage_root,
            secret=_INBOUND_SECRET,
            hmac_secret_ref=_INBOUND_SECRET_REF,
        )
        body = self._body()
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Meridian-Signature": "sha256=bad",
            },
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.inbound")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_non_webhook_channel_no_hmac_check(self, storage_root: Path) -> None:
        """Non-webhook channels are not subject to HMAC verification."""
        from sdk_channel import ChannelCapabilities, ChannelDriver, SendResult

        class OtherDriver(ChannelDriver):
            kind = "test.other"

            async def start(self, request: StartRequest) -> None:
                pass

            async def send(self, request: SendRequest) -> SendResult:
                return SendResult(message_id="m", timestamp="t", delivered=True)

            async def stop(self, request: StopRequest) -> None:
                pass

            def capabilities(self) -> ChannelCapabilities:
                return ChannelCapabilities()

        channels_dir = storage_root / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)
        channel_id = "ch_other"
        record: dict[str, Any] = {
            "id": channel_id,
            "kind": "test.other",
            "config": {"token_vault_ref": "v"},
            "inbound_policy": "open",
            "egress_policy": "enabled",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        (channels_dir / f"{channel_id}.json").write_text(json.dumps(record))

        rt = ChannelRuntime()
        rt.register(OtherDriver())
        app = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            channel_runtime=rt,
            secret_resolver=FixedSecretResolver(_INBOUND_SECRET),
        )
        client = TestClient(app, raise_server_exceptions=False)
        body = json.dumps({"sender_id": "ext-1", "content": "hi"}).encode()
        resp = client.post(
            f"/v1/channels/{channel_id}/inbound",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Full round-trip via system channel outbound
# ---------------------------------------------------------------------------


class TestWebhookRoundTrip:
    def test_outbound_route_delivers_and_returns_delivered_true(
        self, storage_root: Path
    ) -> None:
        captured: list[httpx.Request] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        client_http = httpx.Client(transport=transport)

        # We need an async http_client; use httpx.AsyncClient with the same transport.
        import asyncio

        async def _get_async_client() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(_handler))

        _async_client = asyncio.get_event_loop().run_until_complete(_get_async_client())

        channel_id = _make_channel_file(storage_root, channel_id="ch_rt")
        driver = WebhookChannelDriver(
            storage_root=storage_root,
            http_client=_async_client,
        )
        rt = _make_runtime(driver)
        app = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            channel_runtime=rt,
        )
        api_client = TestClient(app, raise_server_exceptions=False)
        resp = api_client.post(
            f"/v1/channels/{channel_id}/outbound",
            json={"session_id": "sess_x", "recipient": "ext-1", "content": "hi"},
        )
        assert resp.status_code == 200
        assert resp.json()["delivered"] is True
