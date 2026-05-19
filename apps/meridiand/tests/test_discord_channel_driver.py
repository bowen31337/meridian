"""
Tests for the Discord channel driver (meridian.discord).

Covers:
  - Driver capabilities: kind, can_send_text, can_thread, max_message_length.
  - Outbound send (happy path): plain text, embed, threaded reply.
  - Outbound send (failures): missing config, missing token ref, unresolvable
    token, missing discord_channel_id, HTTP errors, network errors, bad embed JSON.
  - OTel spans: emitted on success and error, attributes, invocation event.
  - Audit log: written on failure with correct event and level.
  - start(): bot token resolved, slash commands registered, gateway task started
    with correct intents (default and custom), idempotent re-start.
  - stop(): gateway task cancelled.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from meridiand._discord_channel_driver import (
    DISCORD_EMBED_CONTENT_TYPE,
    _DEFAULT_INTENTS,
    DiscordChannelDriver,
    NoopGatewayClient,
    NoopSecretResolver,
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

_DISCORD_KIND = "meridian.discord"
_BOT_TOKEN = "Bot.test.token.abc123"
_BOT_TOKEN_REF = "vault/discord_bot_token"
_DISCORD_CHANNEL_ID = "111222333444555"
_APPLICATION_ID = "999888777666555"


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
    channel_id: str = "ch_disc_1",
    discord_channel_id: str | None = _DISCORD_CHANNEL_ID,
    bot_token_ref: str | None = _BOT_TOKEN_REF,
    application_id: str | None = None,
    slash_commands: list[dict[str, Any]] | None = None,
    intents: int | None = None,
    egress_policy: str = "enabled",
    inbound_policy: str = "open",
) -> str:
    channels_dir = storage_root / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"token_vault_ref": "vault/tok"}
    if bot_token_ref is not None:
        config["bot_token_ref"] = bot_token_ref
    if discord_channel_id is not None:
        config["discord_channel_id"] = discord_channel_id
    if application_id is not None:
        config["application_id"] = application_id
    if slash_commands is not None:
        config["slash_commands"] = slash_commands
    if intents is not None:
        config["intents"] = intents
    record: dict[str, Any] = {
        "id": channel_id,
        "kind": _DISCORD_KIND,
        "config": config,
        "inbound_policy": inbound_policy,
        "egress_policy": egress_policy,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    (channels_dir / f"{channel_id}.json").write_text(json.dumps(record))
    return channel_id


def _make_send_request(
    channel_id: str = "ch_disc_1",
    *,
    content: str = "hello discord",
    content_type: str = "text/plain",
    thread_id: str | None = None,
) -> SendRequest:
    return SendRequest(
        channel_id=channel_id,
        channel_kind=_DISCORD_KIND,
        session_id="sess_test",
        recipient="discord-user-1",
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
    gateway_client=None,
) -> DiscordChannelDriver:
    resolver = FixedSecretResolver(secret) if secret is not None else NullSecretResolver()
    return DiscordChannelDriver(
        storage_root=storage_root,
        secret_resolver=resolver,
        audit_log=audit_log,
        http_client=http_client,
        gateway_client=gateway_client,
    )


def _discord_response(message_id: str = "discord_msg_999") -> httpx.Response:
    return httpx.Response(200, json={"id": message_id, "content": "hello discord"})


# ---------------------------------------------------------------------------
# Driver: capabilities
# ---------------------------------------------------------------------------


class TestDriverCapabilities:
    def test_kind_is_meridian_discord(self, storage_root: Path) -> None:
        assert _make_driver(storage_root).kind == "meridian.discord"

    def test_capabilities_can_send_text(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert isinstance(caps, ChannelCapabilities)
        assert caps.can_send_text is True

    def test_capabilities_can_thread(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert caps.can_thread is True

    def test_capabilities_max_message_length(self, storage_root: Path) -> None:
        caps = _make_driver(storage_root).capabilities()
        assert caps.max_message_length == 2000


# ---------------------------------------------------------------------------
# Driver: outbound send — happy path
# ---------------------------------------------------------------------------


class TestDriverSendSuccess:
    @pytest.fixture(autouse=True)
    def _otel_clear(self) -> None:
        _otel_exporter.clear()

    async def test_posts_to_discord_messages_url(self, storage_root: Path) -> None:
        captured: list[httpx.Request] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, http_client=client)
            await driver.send(_make_send_request())

        assert len(captured) == 1
        assert f"/channels/{_DISCORD_CHANNEL_ID}/messages" in str(captured[0].url)

    async def test_authorization_header_uses_bot_token(self, storage_root: Path) -> None:
        headers_seen: list[dict] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            headers_seen.append(dict(request.headers))
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            driver = _make_driver(storage_root, secret=_BOT_TOKEN, http_client=client)
            await driver.send(_make_send_request())

        assert headers_seen[0].get("authorization") == f"Bot {_BOT_TOKEN}"

    async def test_result_delivered_true(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            result = await _make_driver(storage_root, http_client=client).send(
                _make_send_request()
            )

        assert result.delivered is True

    async def test_result_message_id_from_discord_response(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _discord_response(message_id="discord_12345")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            result = await _make_driver(storage_root, http_client=client).send(
                _make_send_request()
            )

        assert result.message_id == "discord_12345"

    async def test_result_has_timestamp(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            result = await _make_driver(storage_root, http_client=client).send(
                _make_send_request()
            )

        assert isinstance(result.timestamp, str) and len(result.timestamp) > 0

    async def test_plain_text_sends_content_field(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(content="hello world")
            )

        payload = json.loads(bodies[0])
        assert payload["content"] == "hello world"
        assert "embeds" not in payload

    async def test_embed_content_type_sends_embeds_field(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            embed = {"title": "Hello", "description": "World"}
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(
                    content=json.dumps(embed),
                    content_type=DISCORD_EMBED_CONTENT_TYPE,
                )
            )

        payload = json.loads(bodies[0])
        assert "embeds" in payload
        assert payload["embeds"][0]["title"] == "Hello"
        assert "content" not in payload

    async def test_embed_list_sends_all_embeds(self, storage_root: Path) -> None:
        bodies: list[bytes] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            bodies.append(request.content)
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            embeds = [{"title": "E1"}, {"title": "E2"}]
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(
                    content=json.dumps(embeds),
                    content_type=DISCORD_EMBED_CONTENT_TYPE,
                )
            )

        payload = json.loads(bodies[0])
        assert len(payload["embeds"]) == 2
        assert payload["embeds"][1]["title"] == "E2"

    async def test_thread_id_routes_to_thread_channel(self, storage_root: Path) -> None:
        captured: list[httpx.Request] = []
        thread_id = "777888999000111"

        async def _handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(
                _make_send_request(thread_id=thread_id)
            )

        assert f"/channels/{thread_id}/messages" in str(captured[0].url)
        assert f"/channels/{_DISCORD_CHANNEL_ID}/messages" not in str(captured[0].url)

    async def test_emits_otel_span(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(_make_send_request())

        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "discord.channel.send" in span_names

    async def test_span_has_channel_id_attribute(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root, channel_id="ch_disc_attr")
            req = SendRequest(
                channel_id="ch_disc_attr",
                channel_kind=_DISCORD_KIND,
                session_id="sess_s",
                recipient="r",
                content="c",
            )
            await _make_driver(storage_root, http_client=client).send(req)

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("discord.channel.send")
        assert span is not None
        assert span.attributes["channel.id"] == "ch_disc_attr"

    async def test_span_has_invocation_event(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return _discord_response()

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(storage_root)
            await _make_driver(storage_root, http_client=client).send(_make_send_request())

        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("discord.channel.send")
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
            channel_kind=_DISCORD_KIND,
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

    async def test_missing_discord_channel_id_raises_failure(
        self, storage_root: Path
    ) -> None:
        _make_channel_file(storage_root, discord_channel_id=None)
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(_make_send_request())
        assert exc_info.value.code == "CHAN_DISCORD_CHANNEL_MISSING"

    async def test_http_error_raises_chan_send_failed(self, storage_root: Path) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"message": "Missing Permissions"})

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

    async def test_invalid_embed_json_raises_chan_embed_parse_failed(
        self, storage_root: Path
    ) -> None:
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.send(
                _make_send_request(
                    content="not valid json {{{",
                    content_type=DISCORD_EMBED_CONTENT_TYPE,
                )
            )
        assert exc_info.value.code == "CHAN_EMBED_PARSE_FAILED"

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
        assert entry.event == "discord.channel.send.failed"

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
        span = spans.get("discord.channel.send")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Driver: start() — gateway intents and slash command registration
# ---------------------------------------------------------------------------


class TestDriverStart:
    async def test_start_resolves_bot_token_and_calls_gateway_connect(
        self, storage_root: Path
    ) -> None:
        received: list[tuple[str, int]] = []

        class RecordingGatewayClient:
            async def connect(self, token: str, intents: int) -> None:
                received.append((token, intents))

            async def disconnect(self) -> None:
                pass

        gw = RecordingGatewayClient()
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, secret=_BOT_TOKEN, gateway_client=gw)
        req = StartRequest(
            channel_id="ch_disc_1",
            channel_kind=_DISCORD_KIND,
            session_id="sess_gw",
        )
        await driver.start(req)
        # Allow the event loop to run the gateway task.
        await asyncio.sleep(0)

        assert len(received) == 1
        token_used, intents_used = received[0]
        assert token_used == _BOT_TOKEN

    async def test_start_uses_default_intents(self, storage_root: Path) -> None:
        received: list[int] = []

        class RecordingGatewayClient:
            async def connect(self, token: str, intents: int) -> None:
                received.append(intents)

            async def disconnect(self) -> None:
                pass

        gw = RecordingGatewayClient()
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, gateway_client=gw)
        await driver.start(
            StartRequest(channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s")
        )
        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0] == _DEFAULT_INTENTS

    async def test_start_uses_custom_intents_from_config(self, storage_root: Path) -> None:
        received: list[int] = []
        custom_intents = 512  # GUILD_MESSAGES only

        class RecordingGatewayClient:
            async def connect(self, token: str, intents: int) -> None:
                received.append(intents)

            async def disconnect(self) -> None:
                pass

        gw = RecordingGatewayClient()
        _make_channel_file(storage_root, intents=custom_intents)
        driver = _make_driver(storage_root, gateway_client=gw)
        await driver.start(
            StartRequest(channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s")
        )
        await asyncio.sleep(0)

        assert received[0] == custom_intents

    async def test_start_idempotent_does_not_duplicate_gateway_task(
        self, storage_root: Path
    ) -> None:
        connect_count = 0

        class CountingGatewayClient:
            async def connect(self, token: str, intents: int) -> None:
                nonlocal connect_count
                connect_count += 1
                # Simulate a long-running connection.
                await asyncio.sleep(10)

            async def disconnect(self) -> None:
                pass

        gw = CountingGatewayClient()
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, gateway_client=gw)
        req = StartRequest(channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s1")
        await driver.start(req)
        await asyncio.sleep(0)
        # Second start with same channel_id should not create another task.
        req2 = StartRequest(
            channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s2"
        )
        await driver.start(req2)
        await asyncio.sleep(0)

        assert connect_count == 1

        # Cleanup.
        for task in driver._gateway_tasks.values():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_start_registers_slash_commands(self, storage_root: Path) -> None:
        captured: list[httpx.Request] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(201, json={"id": "cmd_1"})

        slash_commands = [{"name": "ping", "description": "Ping the bot", "type": 1}]
        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(
                storage_root,
                application_id=_APPLICATION_ID,
                slash_commands=slash_commands,
            )
            driver = _make_driver(storage_root, http_client=client)
            await driver.start(
                StartRequest(channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s")
            )

        # One POST per slash command.
        command_posts = [
            r for r in captured if f"/applications/{_APPLICATION_ID}/commands" in str(r.url)
        ]
        assert len(command_posts) == 1
        body = json.loads(command_posts[0].content)
        assert body["name"] == "ping"

    async def test_start_slash_command_uses_bot_auth_header(
        self, storage_root: Path
    ) -> None:
        headers_seen: list[dict] = []

        async def _handler(request: httpx.Request) -> httpx.Response:
            headers_seen.append(dict(request.headers))
            return httpx.Response(201, json={"id": "cmd_1"})

        slash_commands = [{"name": "hello", "description": "Say hello", "type": 1}]
        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(
                storage_root,
                application_id=_APPLICATION_ID,
                slash_commands=slash_commands,
            )
            driver = _make_driver(storage_root, secret=_BOT_TOKEN, http_client=client)
            await driver.start(
                StartRequest(channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s")
            )

        assert headers_seen[0].get("authorization") == f"Bot {_BOT_TOKEN}"

    async def test_start_slash_command_failure_raises_channel_failure(
        self, storage_root: Path
    ) -> None:
        async def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "401: Unauthorized"})

        slash_commands = [{"name": "fail", "description": "Will fail", "type": 1}]
        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            _make_channel_file(
                storage_root,
                application_id=_APPLICATION_ID,
                slash_commands=slash_commands,
            )
            driver = _make_driver(storage_root, http_client=client)
            with pytest.raises(ChannelFailure) as exc_info:
                await driver.start(
                    StartRequest(
                        channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s"
                    )
                )
        assert exc_info.value.code == "CHAN_SLASH_CMD_FAILED"

    async def test_start_missing_bot_token_ref_raises_failure(
        self, storage_root: Path
    ) -> None:
        _make_channel_file(storage_root, bot_token_ref=None)
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.start(
                StartRequest(channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s")
            )
        assert exc_info.value.code == "CHAN_BOT_TOKEN_REF_MISSING"

    async def test_start_missing_config_raises_failure(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root)
        with pytest.raises(ChannelFailure) as exc_info:
            await driver.start(
                StartRequest(
                    channel_id="ch_nonexistent", channel_kind=_DISCORD_KIND, session_id="s"
                )
            )
        assert exc_info.value.code == "CHAN_CONFIG_NOT_FOUND"


# ---------------------------------------------------------------------------
# Driver: stop() — gateway task lifecycle
# ---------------------------------------------------------------------------


class TestDriverStop:
    async def test_stop_cancels_gateway_task(self, storage_root: Path) -> None:
        connected = asyncio.Event()
        cancelled = asyncio.Event()

        class LongRunningGatewayClient:
            async def connect(self, token: str, intents: int) -> None:
                connected.set()
                try:
                    await asyncio.sleep(999)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            async def disconnect(self) -> None:
                pass

        gw = LongRunningGatewayClient()
        _make_channel_file(storage_root)
        driver = _make_driver(storage_root, gateway_client=gw)

        await driver.start(
            StartRequest(channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s")
        )
        await connected.wait()

        await driver.stop(
            StopRequest(channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s")
        )

        assert cancelled.is_set()
        assert "ch_disc_1" not in driver._gateway_tasks

    async def test_stop_noop_when_no_gateway_task(self, storage_root: Path) -> None:
        driver = _make_driver(storage_root)
        # Should not raise even if no task was started.
        await driver.stop(
            StopRequest(channel_id="ch_disc_1", channel_kind=_DISCORD_KIND, session_id="s")
        )
