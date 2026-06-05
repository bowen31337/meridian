"""
Tests for the Telegram channel driver (meridian.telegram).

Covers:
  - Driver capabilities: kind, can_send_text, can_thread, max_message_length,
    rate_limit_per_minute.
  - Outbound send (happy path): plain text, MarkdownV2, threaded reply.
  - Outbound send (failures): missing config, missing token ref, unresolvable
    token, missing telegram_chat_id, HTTP errors, network errors, Telegram
    API error response.
  - OTel spans: emitted on success and error, attributes, invocation event.
  - Audit log: written on failure with correct event and level.
  - start() long-poll mode: bot token resolved, poll task started with correct
    token and timeout (default and custom), idempotent re-start.
  - start() webhook mode: no poll task started.
  - stop(): poll task cancelled.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import httpx
from meridiand._telegram_channel_driver import (
    _DEFAULT_POLL_TIMEOUT,
    TELEGRAM_MARKDOWN_CONTENT_TYPE,
    TelegramChannelDriver,
)
from opentelemetry.trace import StatusCode
import pytest
from sdk_channel import (
    ChannelCapabilities,
    ChannelFailure,
    SendRequest,
    StartRequest,
    StopRequest,
)

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TELEGRAM_KIND = "meridian.telegram"
_BOT_TOKEN = "1234567890:ABCDefghIJKLmnopQRSTuvwxyz"
_BOT_TOKEN_REF = "vault/telegram_bot_token"
_CHAT_ID = "-1001234567890"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FixedSecretResolver:
    def __init__(self, secret: str) -> None:
        self._secret = secret

    def resolve(self, secret_ref: str) -> str | None:
        return self._secret


class NullSecretResolver:
    """Always returns None (secret unresolvable)."""

    def resolve(self, secret_ref: str) -> str | None:
        return None


def _make_channel_file(
    storage_root: Path,
    *,
    channel_id: str = "ch_tg_1",
    telegram_chat_id: str | None = _CHAT_ID,
    bot_token_ref: str | None = _BOT_TOKEN_REF,
    mode: str | None = None,
    poll_timeout: int | None = None,
    egress_policy: str = "enabled",
    inbound_policy: str = "open",
) -> str:
    channels_dir = storage_root / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"token_vault_ref": "vault/tok"}
    if bot_token_ref is not None:
        config["bot_token_ref"] = bot_token_ref
    if telegram_chat_id is not None:
        config["telegram_chat_id"] = telegram_chat_id
    if mode is not None:
        config["mode"] = mode
    if poll_timeout is not None:
        config["poll_timeout"] = poll_timeout
    record: dict[str, Any] = {
        "id": channel_id,
        "kind": _TELEGRAM_KIND,
        "config": config,
        "inbound_policy": inbound_policy,
        "egress_policy": egress_policy,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    (channels_dir / f"{channel_id}.json").write_text(json.dumps(record))
    return channel_id


def _make_send_request(
    channel_id: str = "ch_tg_1",
    *,
    content: str = "hello telegram",
    content_type: str = "text/plain",
    thread_id: str | None = None,
) -> SendRequest:
    return SendRequest(
        channel_id=channel_id,
        channel_kind=_TELEGRAM_KIND,
        session_id="sess_test",
        recipient="telegram-user-1",
        content=content,
        content_type=content_type,
        thread_id=thread_id,
    )


def _make_driver(
    storage_root: Path,
    *,
    secret: str | None = _BOT_TOKEN,
    audit_log=None,
    http_client: httpx.AsyncClient | None = None,
    long_poll_client=None,
) -> TelegramChannelDriver:
    resolver = FixedSecretResolver(secret) if secret is not None else NullSecretResolver()
    return TelegramChannelDriver(
        storage_root=storage_root,
        secret_resolver=resolver,
        audit_log=audit_log,
        http_client=http_client,
        long_poll_client=long_poll_client,
    )


def _telegram_ok_response(message_id: int = 999) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "ok": True,
            "result": {
                "message_id": message_id,
                "chat": {"id": int(_CHAT_ID)},
                "text": "hello telegram",
            },
        },
    )


# ---------------------------------------------------------------------------
# Driver: capabilities
# ---------------------------------------------------------------------------


class TestDriverCapabilities:
    def test_kind_is_meridian_telegram(self, storage_root: Path) -> None:
        assert _make_driver(storage_root).kind == "meridian.telegram"

    def test_capabilities_can_send_text(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert isinstance(caps, ChannelCapabilities)
        assert caps.can_send_text is True

    def test_capabilities_can_thread(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert caps.can_thread is True

    def test_capabilities_max_message_length(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert caps.max_message_length == 4096

    def test_capabilities_rate_limit_per_minute(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert caps.rate_limit_per_minute == 30


# ---------------------------------------------------------------------------
# Driver: outbound send — happy path
# ---------------------------------------------------------------------------


class TestDriverSendSuccess:
    @pytest.fixture(autouse=True)
    def _otel_clear(self) -> None:
        _otel_exporter.clear()

    async def test_posts_to_send_message_url(self, storage_root: Path) -> None:
        captured: list[httpx.Request] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            await driver.send(_make_send_request())

        assert len(captured) == 1
        assert "/sendMessage" in str(captured[0].url)

    async def test_url_contains_bot_token(self, storage_root: Path) -> None:
        captured: list[httpx.Request] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, secret=_BOT_TOKEN, http_client=client)
            await driver.send(_make_send_request())

        assert f"/bot{_BOT_TOKEN}/" in str(captured[0].url)

    async def test_result_delivered_true(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            result = await _make_driver(storage_root, http_client=client).send(_make_send_request())

        assert result.delivered is True

    async def test_result_message_id_from_telegram_response(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _telegram_ok_response(message_id=42)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            result = await _make_driver(storage_root, http_client=client).send(_make_send_request())

        assert result.message_id == "42"

    async def test_result_has_timestamp(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            result = await _make_driver(storage_root, http_client=client).send(_make_send_request())

        assert isinstance(result.timestamp, str) and len(result.timestamp) > 0

    async def test_plain_text_sends_text_field(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(content="hello world")
            )

        payload = json.loads(bodies[0])
        assert payload["text"] == "hello world"
        assert "parse_mode" not in payload

    async def test_markdown_content_type_sets_parse_mode(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(
                    content="*bold* text",
                    content_type=TELEGRAM_MARKDOWN_CONTENT_TYPE,
                )
            )

        payload = json.loads(bodies[0])
        assert payload["text"] == "*bold* text"
        assert payload["parse_mode"] == "MarkdownV2"

    async def test_payload_includes_chat_id(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(_make_send_request())

        payload = json.loads(bodies[0])
        assert payload["chat_id"] == _CHAT_ID

    async def test_thread_id_sets_reply_to_message_id(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(thread_id="777")
            )

        payload = json.loads(bodies[0])
        assert payload["reply_to_message_id"] == 777

    async def test_emits_otel_span(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(_make_send_request())

        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "telegram.channel.send" in span_names

    async def test_span_has_channel_id_attribute(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root, channel_id="ch_tg_attr")
            req = SendRequest(
                channel_id="ch_tg_attr",
                channel_kind=_TELEGRAM_KIND,
                session_id="sess_s",
                recipient="r",
                content="c",
            )
            await _make_driver(storage_root, http_client=client).send(req)

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("telegram.channel.send")
        assert span is not None
        assert span.attributes["channel.id"] == "ch_tg_attr"

    async def test_span_has_invocation_event(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _telegram_ok_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(_make_send_request())

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("telegram.channel.send")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names


# ---------------------------------------------------------------------------
# Driver: outbound send — failures
# ---------------------------------------------------------------------------


class TestDriverSendFailures:
    @pytest.fixture(autouse=True)
    def _otel_clear(self) -> None:
        _otel_exporter.clear()

    async def test_missing_config_raises_chan_config_not_found(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root)
        req = SendRequest(
            channel_id="ch_nonexistent",
            channel_kind=_TELEGRAM_KIND,
            session_id="s",
            recipient="r",
            content="c",
        )
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(req)
        assert exc_info.value.code == "CHAN_CONFIG_NOT_FOUND"

    async def test_missing_bot_token_ref_raises_failure(self, storage_root: Path) -> None:
        _make_channel_file(storage_root, bot_token_ref=None)
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(_make_send_request())
        assert exc_info.value.code == "CHAN_BOT_TOKEN_REF_MISSING"

    async def test_unresolvable_token_raises_failure(self, storage_root: Path) -> None:
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, secret=None)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(_make_send_request())
        assert exc_info.value.code == "CHAN_BOT_TOKEN_UNRESOLVABLE"

    async def test_missing_telegram_chat_id_raises_failure(self, storage_root: Path) -> None:
        _make_channel_file(storage_root, telegram_chat_id=None)
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(_make_send_request())
        assert exc_info.value.code == "CHAN_TELEGRAM_CHAT_ID_MISSING"

    async def test_http_error_raises_chan_send_failed(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"ok": False, "description": "Forbidden"})

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            with pytest.raises(ChannelFailure) as exc_info:
                await _make_driver(storage_root, http_client=client).send(_make_send_request())
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_network_error_raises_chan_send_failed(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            with pytest.raises(ChannelFailure) as exc_info:
                await _make_driver(storage_root, http_client=client).send(_make_send_request())
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_telegram_api_ok_false_raises_chan_send_failed(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"ok": False, "description": "Bad Request: chat not found"},
            )

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            with pytest.raises(ChannelFailure) as exc_info:
                await _make_driver(storage_root, http_client=client).send(_make_send_request())
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_http_error_writes_audit_log(self, storage_root: Path) -> None:
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
        assert entry.event == "telegram.channel.send.failed"

    async def test_http_error_span_marked_error(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            with pytest.raises(ChannelFailure):
                await _make_driver(storage_root, http_client=client).send(_make_send_request())

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("telegram.channel.send")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Driver: start() — long-poll and webhook modes
# ---------------------------------------------------------------------------


class TestDriverStart:
    async def test_start_long_poll_resolves_token_and_calls_poll(self, storage_root: Path) -> None:
        received: list[tuple[str, int]] = []

        class RecordingLongPollClient:
            async def poll(self, token: str, timeout: int) -> None:
                received.append((token, timeout))

            async def stop(self) -> None:
                pass

        lpc = RecordingLongPollClient()
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, secret=_BOT_TOKEN, long_poll_client=lpc)
        req = StartRequest(
            channel_id="ch_tg_1",
            channel_kind=_TELEGRAM_KIND,
            session_id="sess_lp",
        )
        await driver.start(req)
        await asyncio.sleep(0)

        assert len(received) == 1
        token_used, timeout_used = received[0]
        assert token_used == _BOT_TOKEN

    async def test_start_uses_default_poll_timeout(self, storage_root: Path) -> None:
        received: list[int] = []

        class RecordingLongPollClient:
            async def poll(self, token: str, timeout: int) -> None:
                received.append(timeout)

            async def stop(self) -> None:
                pass

        lpc = RecordingLongPollClient()
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, long_poll_client=lpc)
        await driver.start(
            StartRequest(channel_id="ch_tg_1", channel_kind=_TELEGRAM_KIND, session_id="s")
        )
        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0] == _DEFAULT_POLL_TIMEOUT

    async def test_start_uses_custom_poll_timeout_from_config(self, storage_root: Path) -> None:
        received: list[int] = []

        class RecordingLongPollClient:
            async def poll(self, token: str, timeout: int) -> None:
                received.append(timeout)

            async def stop(self) -> None:
                pass

        lpc = RecordingLongPollClient()
        _make_channel_file(storage_root, poll_timeout=60)
        driver = _make_driver(storage_root, long_poll_client=lpc)
        await driver.start(
            StartRequest(channel_id="ch_tg_1", channel_kind=_TELEGRAM_KIND, session_id="s")
        )
        await asyncio.sleep(0)

        assert received[0] == 60

    async def test_start_long_poll_idempotent_does_not_duplicate_task(
        self, storage_root: Path
    ) -> None:
        poll_count = 0

        class CountingLongPollClient:
            async def poll(self, token: str, timeout: int) -> None:
                nonlocal poll_count
                poll_count += 1
                await asyncio.sleep(10)

            async def stop(self) -> None:
                pass

        lpc = CountingLongPollClient()
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, long_poll_client=lpc)
        req = StartRequest(channel_id="ch_tg_1", channel_kind=_TELEGRAM_KIND, session_id="s1")
        await driver.start(req)
        await asyncio.sleep(0)
        req2 = StartRequest(channel_id="ch_tg_1", channel_kind=_TELEGRAM_KIND, session_id="s2")
        await driver.start(req2)
        await asyncio.sleep(0)

        assert poll_count == 1

        for task in driver._poll_tasks.values():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def test_start_webhook_mode_does_not_start_poll_task(self, storage_root: Path) -> None:
        poll_called = False

        class TrackingLongPollClient:
            async def poll(self, token: str, timeout: int) -> None:
                nonlocal poll_called
                poll_called = True

            async def stop(self) -> None:
                pass

        lpc = TrackingLongPollClient()
        _make_channel_file(storage_root, mode="webhook")
        driver = _make_driver(storage_root, long_poll_client=lpc)
        await driver.start(
            StartRequest(channel_id="ch_tg_1", channel_kind=_TELEGRAM_KIND, session_id="s")
        )
        await asyncio.sleep(0)

        assert not poll_called
        assert "ch_tg_1" not in driver._poll_tasks

    async def test_start_missing_bot_token_ref_raises_failure(self, storage_root: Path) -> None:
        _make_channel_file(storage_root, bot_token_ref=None)
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.start(
                StartRequest(channel_id="ch_tg_1", channel_kind=_TELEGRAM_KIND, session_id="s")
            )
        assert exc_info.value.code == "CHAN_BOT_TOKEN_REF_MISSING"

    async def test_start_missing_config_raises_failure(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.start(
                StartRequest(
                    channel_id="ch_nonexistent", channel_kind=_TELEGRAM_KIND, session_id="s"
                )
            )
        assert exc_info.value.code == "CHAN_CONFIG_NOT_FOUND"


# ---------------------------------------------------------------------------
# Driver: stop() — poll task lifecycle
# ---------------------------------------------------------------------------


class TestDriverStop:
    async def test_stop_cancels_poll_task(self, storage_root: Path) -> None:
        connected = asyncio.Event()
        cancelled = asyncio.Event()

        class LongRunningLongPollClient:
            async def poll(self, token: str, timeout: int) -> None:
                connected.set()
                try:
                    await asyncio.sleep(999)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            async def stop(self) -> None:
                pass

        lpc = LongRunningLongPollClient()
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, long_poll_client=lpc)

        await driver.start(
            StartRequest(channel_id="ch_tg_1", channel_kind=_TELEGRAM_KIND, session_id="s")
        )
        await connected.wait()

        await driver.stop(
            StopRequest(channel_id="ch_tg_1", channel_kind=_TELEGRAM_KIND, session_id="s")
        )

        assert cancelled.is_set()
        assert "ch_tg_1" not in driver._poll_tasks

    async def test_stop_noop_when_no_poll_task(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root)
        await driver.stop(
            StopRequest(channel_id="ch_tg_1", channel_kind=_TELEGRAM_KIND, session_id="s")
        )
