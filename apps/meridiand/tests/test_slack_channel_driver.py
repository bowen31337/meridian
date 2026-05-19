"""
Tests for the Slack channel driver (meridian.slack).

Covers:
  - Driver capabilities: kind, can_send_text, can_thread, max_message_length.
  - Outbound send (happy path): plain text, Block Kit blocks, threaded reply.
  - Outbound send (failures): missing config, missing token ref, unresolvable
    token, missing slack_channel_id, HTTP errors, Slack API ok=false, network
    errors, bad blocks JSON.
  - OTel spans: emitted on success and error, attributes, invocation event.
  - Audit log: written on failure with correct event and level.
  - start(): bot token resolved, Socket Mode task started with app token,
    idempotent re-start, no-op when slack_app_token_ref absent.
  - stop(): Socket Mode task cancelled.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from meridiand._slack_channel_driver import (
    SLACK_BLOCKS_CONTENT_TYPE,
    NoopSocketModeClient,
    NoopSecretResolver,
    SlackChannelDriver,
)
from opentelemetry.trace import StatusCode
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

_SLACK_KIND = "meridian.slack"
_BOT_TOKEN = "xoxb-test-bot-token"
_BOT_TOKEN_REF = "vault/slack_bot_token"
_APP_TOKEN = "xapp-test-app-token"
_APP_TOKEN_REF = "vault/slack_app_token"
_SLACK_CHANNEL_ID = "C0123456789"
_MESSAGE_TS = "1234567890.123456"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FixedSecretResolver:
    def __init__(self, secret: str) -> None:
        self._secret = secret

    def resolve(self, secret_ref: str) -> str | None:
        return self._secret


class MappedSecretResolver:
    """Returns a different secret per ref."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def resolve(self, secret_ref: str) -> str | None:
        return self._mapping.get(secret_ref)


class NullSecretResolver:
    """Always returns None (secret unresolvable)."""

    def resolve(self, secret_ref: str) -> str | None:
        return None


def _make_channel_file(
    storage_root: Path,
    *,
    channel_id: str = "ch_slack_1",
    slack_channel_id: str | None = _SLACK_CHANNEL_ID,
    bot_token_ref: str | None = _BOT_TOKEN_REF,
    slack_app_token_ref: str | None = None,
    egress_policy: str = "enabled",
    inbound_policy: str = "open",
) -> str:
    channels_dir = storage_root / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"token_vault_ref": "vault/tok"}
    if bot_token_ref is not None:
        config["bot_token_ref"] = bot_token_ref
    if slack_channel_id is not None:
        config["slack_channel_id"] = slack_channel_id
    if slack_app_token_ref is not None:
        config["slack_app_token_ref"] = slack_app_token_ref
    record: dict[str, Any] = {
        "id": channel_id,
        "kind": _SLACK_KIND,
        "config": config,
        "inbound_policy": inbound_policy,
        "egress_policy": egress_policy,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    (channels_dir / f"{channel_id}.json").write_text(json.dumps(record))
    return channel_id


def _make_send_request(
    channel_id: str = "ch_slack_1",
    *,
    content: str = "hello slack",
    content_type: str = "text/plain",
    thread_id: str | None = None,
    metadata: dict[str, str] | None = None,
) -> SendRequest:
    return SendRequest(
        channel_id=channel_id,
        channel_kind=_SLACK_KIND,
        session_id="sess_test",
        recipient="slack-user-1",
        content=content,
        content_type=content_type,
        thread_id=thread_id,
        metadata=metadata or {},
    )


def _make_driver(
    storage_root: Path,
    *,
    secret: str | None = _BOT_TOKEN,
    secret_mapping: dict[str, str] | None = None,
    audit_log=None,
    http_client: httpx.AsyncClient | None = None,
    socket_mode_client=None,
) -> SlackChannelDriver:
    if secret_mapping is not None:
        resolver = MappedSecretResolver(secret_mapping)
    elif secret is not None:
        resolver = FixedSecretResolver(secret)
    else:
        resolver = NullSecretResolver()
    return SlackChannelDriver(
        storage_root=storage_root,
        secret_resolver=resolver,
        audit_log=audit_log,
        http_client=http_client,
        socket_mode_client=socket_mode_client,
    )


def _slack_response(
    ts: str = _MESSAGE_TS,
    ok: bool = True,
    error: str | None = None,
) -> httpx.Response:
    body: dict[str, Any] = {"ok": ok, "ts": ts, "channel": _SLACK_CHANNEL_ID}
    if error is not None:
        body["error"] = error
    return httpx.Response(200, json=body)


# ---------------------------------------------------------------------------
# Driver: capabilities
# ---------------------------------------------------------------------------


class TestDriverCapabilities:
    def test_kind_is_meridian_slack(self, storage_root: Path) -> None:
        assert _make_driver(storage_root).kind == "meridian.slack"

    def test_capabilities_can_send_text(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert isinstance(caps, ChannelCapabilities)
        assert caps.can_send_text is True

    def test_capabilities_can_thread(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert caps.can_thread is True

    def test_capabilities_max_message_length(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert caps.max_message_length == 4000


# ---------------------------------------------------------------------------
# Driver: outbound send — happy path
# ---------------------------------------------------------------------------


class TestDriverSendSuccess:
    @pytest.fixture(autouse=True)
    def _otel_clear(self) -> None:
        _otel_exporter.clear()

    async def test_posts_to_slack_chat_post_message_url(self, storage_root: Path) -> None:
        captured: list[httpx.Request] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            await driver.send(_make_send_request())

        assert len(captured) == 1
        assert "chat.postMessage" in str(captured[0].url)

    async def test_authorization_header_uses_bearer_token(self, storage_root: Path) -> None:
        headers_seen: list[dict] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            headers_seen.append(dict(request.headers))
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, secret=_BOT_TOKEN, http_client=client)
            await driver.send(_make_send_request())

        assert headers_seen[0].get("authorization") == f"Bearer {_BOT_TOKEN}"

    async def test_result_delivered_true(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            result = await _make_driver(storage_root, http_client=client).send(
                _make_send_request()
            )

        assert result.delivered is True

    async def test_result_message_id_contains_ts(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _slack_response(ts="9999999999.000001")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            result = await _make_driver(storage_root, http_client=client).send(
                _make_send_request()
            )

        assert "9999999999.000001" in result.message_id

    async def test_result_has_timestamp(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            result = await _make_driver(storage_root, http_client=client).send(
                _make_send_request()
            )

        assert isinstance(result.timestamp, str) and len(result.timestamp) > 0

    async def test_plain_text_sends_text_field(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(content="hello world")
            )

        payload = json.loads(bodies[0])
        assert payload["text"] == "hello world"
        assert "blocks" not in payload

    async def test_plain_text_includes_channel_id(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(_make_send_request())

        payload = json.loads(bodies[0])
        assert payload["channel"] == _SLACK_CHANNEL_ID

    async def test_blocks_content_type_sends_blocks_field(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            block = {"type": "section", "text": {"type": "mrkdwn", "text": "*Hello*"}}
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(
                    content=json.dumps(block),
                    content_type=SLACK_BLOCKS_CONTENT_TYPE,
                )
            )

        payload = json.loads(bodies[0])
        assert "blocks" in payload
        assert payload["blocks"][0]["type"] == "section"
        assert "text" in payload  # fallback field required by Slack

    async def test_blocks_list_sends_all_blocks(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text": "B1"}},
                {"type": "divider"},
            ]
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(
                    content=json.dumps(blocks),
                    content_type=SLACK_BLOCKS_CONTENT_TYPE,
                )
            )

        payload = json.loads(bodies[0])
        assert len(payload["blocks"]) == 2
        assert payload["blocks"][1]["type"] == "divider"

    async def test_blocks_fallback_text_from_metadata(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            block = {"type": "section", "text": {"type": "mrkdwn", "text": "*Hi*"}}
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(
                    content=json.dumps(block),
                    content_type=SLACK_BLOCKS_CONTENT_TYPE,
                    metadata={"fallback_text": "Hi"},
                )
            )

        payload = json.loads(bodies[0])
        assert payload["text"] == "Hi"

    async def test_thread_id_sets_thread_ts(self, storage_root: Path) -> None:
        bodies: list[bytes] = []
        thread_ts = "1111111111.222222"

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(thread_id=thread_ts)
            )

        payload = json.loads(bodies[0])
        assert payload.get("thread_ts") == thread_ts

    async def test_no_thread_id_omits_thread_ts(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(_make_send_request())

        payload = json.loads(bodies[0])
        assert "thread_ts" not in payload

    async def test_emits_otel_span(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(_make_send_request())

        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "slack.channel.send" in span_names

    async def test_span_has_channel_id_attribute(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root, channel_id="ch_slack_attr")
            req = SendRequest(
                channel_id="ch_slack_attr",
                channel_kind=_SLACK_KIND,
                session_id="sess_s",
                recipient="r",
                content="c",
            )
            await _make_driver(storage_root, http_client=client).send(req)

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("slack.channel.send")
        assert span is not None
        assert span.attributes["channel.id"] == "ch_slack_attr"

    async def test_span_has_invocation_event(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _slack_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(_make_send_request())

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("slack.channel.send")
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

    async def test_missing_config_raises_chan_config_not_found(
        self, storage_root: Path
    ) -> None:
        driver = _make_driver(storage_root)
        req = SendRequest(
            channel_id="ch_nonexistent",
            channel_kind=_SLACK_KIND,
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

    async def test_missing_slack_channel_id_raises_failure(
        self, storage_root: Path
    ) -> None:
        _make_channel_file(storage_root, slack_channel_id=None)
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(_make_send_request())
        assert exc_info.value.code == "CHAN_SLACK_CHANNEL_MISSING"

    async def test_http_error_raises_chan_send_failed(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "internal_server_error"})

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            with pytest.raises(ChannelFailure) as exc_info:
                await _make_driver(storage_root, http_client=client).send(
                    _make_send_request()
                )
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_slack_api_ok_false_raises_chan_send_failed(
        self, storage_root: Path
    ) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _slack_response(ok=False, error="channel_not_found")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            with pytest.raises(ChannelFailure) as exc_info:
                await _make_driver(storage_root, http_client=client).send(
                    _make_send_request()
                )
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_network_error_raises_chan_send_failed(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            with pytest.raises(ChannelFailure) as exc_info:
                await _make_driver(storage_root, http_client=client).send(
                    _make_send_request()
                )
        assert exc_info.value.code == "CHAN_SEND_FAILED"

    async def test_invalid_blocks_json_raises_chan_blocks_parse_failed(
        self, storage_root: Path
    ) -> None:
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(
                _make_send_request(
                    content="not valid json {{{",
                    content_type=SLACK_BLOCKS_CONTENT_TYPE,
                )
            )
        assert exc_info.value.code == "CHAN_BLOCKS_PARSE_FAILED"

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
        assert entry.event == "slack.channel.send.failed"

    async def test_http_error_span_marked_error(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            with pytest.raises(ChannelFailure):
                await _make_driver(storage_root, http_client=client).send(
                    _make_send_request()
                )

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("slack.channel.send")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Driver: start() — Socket Mode lifecycle
# ---------------------------------------------------------------------------


class TestDriverStart:
    async def test_start_resolves_tokens_and_calls_socket_mode_connect(
        self, storage_root: Path
    ) -> None:
        received: list[tuple[str, str]] = []

        class RecordingSocketModeClient:
            async def connect(self, app_token: str, bot_token: str) -> None:
                received.append((app_token, bot_token))

            async def disconnect(self) -> None:
                pass

        smc = RecordingSocketModeClient()
        _make_channel_file(
            storage_root,
            slack_app_token_ref=_APP_TOKEN_REF,
        )
        driver = _make_driver(
            storage_root,
            secret_mapping={_BOT_TOKEN_REF: _BOT_TOKEN, _APP_TOKEN_REF: _APP_TOKEN},
            socket_mode_client=smc,
        )
        req = StartRequest(
            channel_id="ch_slack_1",
            channel_kind=_SLACK_KIND,
            session_id="sess_start",
        )
        await driver.start(req)
        await asyncio.sleep(0)

        assert len(received) == 1
        app_token_used, bot_token_used = received[0]
        assert app_token_used == _APP_TOKEN
        assert bot_token_used == _BOT_TOKEN

    async def test_start_without_app_token_ref_is_noop(self, storage_root: Path) -> None:
        connect_count = 0

        class CountingSocketModeClient:
            async def connect(self, app_token: str, bot_token: str) -> None:
                nonlocal connect_count
                connect_count += 1

            async def disconnect(self) -> None:
                pass

        smc = CountingSocketModeClient()
        _make_channel_file(storage_root)  # no slack_app_token_ref
        driver = _make_driver(storage_root, socket_mode_client=smc)
        await driver.start(
            StartRequest(channel_id="ch_slack_1", channel_kind=_SLACK_KIND, session_id="s")
        )
        await asyncio.sleep(0)

        assert connect_count == 0

    async def test_start_idempotent_does_not_duplicate_socket_task(
        self, storage_root: Path
    ) -> None:
        connect_count = 0

        class CountingSocketModeClient:
            async def connect(self, app_token: str, bot_token: str) -> None:
                nonlocal connect_count
                connect_count += 1
                await asyncio.sleep(10)

            async def disconnect(self) -> None:
                pass

        smc = CountingSocketModeClient()
        _make_channel_file(storage_root, slack_app_token_ref=_APP_TOKEN_REF)
        driver = _make_driver(
            storage_root,
            secret_mapping={_BOT_TOKEN_REF: _BOT_TOKEN, _APP_TOKEN_REF: _APP_TOKEN},
            socket_mode_client=smc,
        )
        req = StartRequest(channel_id="ch_slack_1", channel_kind=_SLACK_KIND, session_id="s1")
        await driver.start(req)
        await asyncio.sleep(0)

        req2 = StartRequest(channel_id="ch_slack_1", channel_kind=_SLACK_KIND, session_id="s2")
        await driver.start(req2)
        await asyncio.sleep(0)

        assert connect_count == 1

        for task in driver._socket_tasks.values():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_start_missing_bot_token_ref_raises_failure(
        self, storage_root: Path
    ) -> None:
        _make_channel_file(storage_root, bot_token_ref=None)
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.start(
                StartRequest(channel_id="ch_slack_1", channel_kind=_SLACK_KIND, session_id="s")
            )
        assert exc_info.value.code == "CHAN_BOT_TOKEN_REF_MISSING"

    async def test_start_missing_config_raises_failure(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.start(
                StartRequest(
                    channel_id="ch_nonexistent", channel_kind=_SLACK_KIND, session_id="s"
                )
            )
        assert exc_info.value.code == "CHAN_CONFIG_NOT_FOUND"

    async def test_start_unresolvable_app_token_raises_failure(
        self, storage_root: Path
    ) -> None:
        _make_channel_file(storage_root, slack_app_token_ref=_APP_TOKEN_REF)
        # bot token resolves, app token does not
        driver = _make_driver(
            storage_root,
            secret_mapping={_BOT_TOKEN_REF: _BOT_TOKEN},
        )
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.start(
                StartRequest(channel_id="ch_slack_1", channel_kind=_SLACK_KIND, session_id="s")
            )
        assert exc_info.value.code == "CHAN_APP_TOKEN_UNRESOLVABLE"


# ---------------------------------------------------------------------------
# Driver: stop() — Socket Mode task lifecycle
# ---------------------------------------------------------------------------


class TestDriverStop:
    async def test_stop_cancels_socket_task(self, storage_root: Path) -> None:
        connected = asyncio.Event()
        cancelled = asyncio.Event()

        class LongRunningSocketModeClient:
            async def connect(self, app_token: str, bot_token: str) -> None:
                connected.set()
                try:
                    await asyncio.sleep(999)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            async def disconnect(self) -> None:
                pass

        smc = LongRunningSocketModeClient()
        _make_channel_file(storage_root, slack_app_token_ref=_APP_TOKEN_REF)
        driver = _make_driver(
            storage_root,
            secret_mapping={_BOT_TOKEN_REF: _BOT_TOKEN, _APP_TOKEN_REF: _APP_TOKEN},
            socket_mode_client=smc,
        )

        await driver.start(
            StartRequest(channel_id="ch_slack_1", channel_kind=_SLACK_KIND, session_id="s")
        )
        await connected.wait()

        await driver.stop(
            StopRequest(channel_id="ch_slack_1", channel_kind=_SLACK_KIND, session_id="s")
        )

        assert cancelled.is_set()
        assert "ch_slack_1" not in driver._socket_tasks

    async def test_stop_noop_when_no_socket_task(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root)
        # Should not raise even if no task was started.
        await driver.stop(
            StopRequest(channel_id="ch_slack_1", channel_kind=_SLACK_KIND, session_id="s")
        )
