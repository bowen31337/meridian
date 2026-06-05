"""
Discord Channel Driver: meridian.discord

Outbound: POST to Discord REST API (channels/{id}/messages).  Supports plain
text (content_type: text/plain) and Discord embed payloads
(content_type: application/vnd.discord.embed+json).  Threaded replies are
routed to the Discord thread channel identified by SendRequest.thread_id.

Inbound: Discord Gateway WebSocket.  A per-channel background task is started
in start() and cancelled in stop().  Gateway intent flags are taken from the
channel config field "intents"; if absent, the driver subscribes to
GUILDS | GUILD_MESSAGES | MESSAGE_CONTENT | DIRECT_MESSAGES.  Slash commands
listed in the channel config "slash_commands" are registered with the Discord
applications API on each start() call.

Bot token is resolved from Vault at call time via SecretResolver using the
channel config field bot_token_ref.  The channel config must also supply
discord_channel_id for outbound delivery.

Every outbound send emits an OTel span "discord.channel.send" with
channel/session attributes and a structured invocation event.  On failure the
error is surfaced as ChannelFailure and an audit-log entry is written before
re-raising.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
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

_DISCORD_API_BASE = "https://discord.com/api/v10"

# Gateway intents bitmask components
_INTENT_GUILDS = 1 << 0
_INTENT_GUILD_MESSAGES = 1 << 9
_INTENT_DIRECT_MESSAGES = 1 << 12
_INTENT_MESSAGE_CONTENT = 1 << 15
_DEFAULT_INTENTS = (
    _INTENT_GUILDS | _INTENT_GUILD_MESSAGES | _INTENT_DIRECT_MESSAGES | _INTENT_MESSAGE_CONTENT
)

DISCORD_EMBED_CONTENT_TYPE = "application/vnd.discord.embed+json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@runtime_checkable
class SecretResolver(Protocol):
    """Resolves a secret_ref to the raw secret value, or None if unavailable."""

    def resolve(self, secret_ref: str) -> str | None: ...


class NoopSecretResolver:
    """Always returns None — no secrets resolved."""

    def resolve(self, secret_ref: str) -> str | None:
        return None


@runtime_checkable
class GatewayClient(Protocol):
    """
    Abstract Discord Gateway WebSocket client.

    The driver calls connect() in a background task started by start(); the
    implementation handles IDENTIFY, heartbeat, and event dispatch.
    Inject NoopGatewayClient (or a mock) for unit tests.
    """

    async def connect(self, token: str, intents: int) -> None:
        """Connect to the Discord Gateway, send IDENTIFY, and dispatch events."""
        ...

    async def disconnect(self) -> None:
        """Gracefully close the Gateway connection."""
        ...


class NoopGatewayClient:
    """No-op gateway client; performs no network I/O."""

    async def connect(self, token: str, intents: int) -> None:
        pass

    async def disconnect(self) -> None:
        pass


class DiscordChannelDriver(ChannelDriver):
    """
    Discord channel driver (kind: meridian.discord).

    Channel config fields (nested under "config" in the channel record):
        token_vault_ref      (required, system) Vault ref used by channel registration.
        bot_token_ref        (required) Secret ref for the Discord bot token.
        discord_channel_id   (required) Discord channel/DM ID for outbound messages.
        application_id       (optional) Discord application ID; needed for slash commands.
        intents              (optional) Gateway intents bitmask.
                             Default: GUILDS | GUILD_MESSAGES | MESSAGE_CONTENT |
                             DIRECT_MESSAGES (37377).
        slash_commands       (optional) List of slash command definition dicts to
                             register via POST applications/{id}/commands.
    """

    kind = "meridian.discord"

    def __init__(
        self,
        *,
        storage_root: Path,
        secret_resolver: SecretResolver | None = None,
        audit_log: AuditLog | None = None,
        http_client: httpx.AsyncClient | None = None,
        gateway_client: GatewayClient | None = None,
    ) -> None:
        self._storage_root = storage_root
        self._resolver = secret_resolver or NoopSecretResolver()
        self._audit_log = audit_log or NoopAuditLog()
        self._http_client = http_client
        self._gateway_client: GatewayClient = gateway_client or NoopGatewayClient()
        self._gateway_tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_driver_config(
        self, channel_id: str, channel_kind: str, session_id: str
    ) -> dict[str, Any]:
        channel_file = self._storage_root / "channels" / f"{channel_id}.json"
        if not channel_file.exists():
            raise ChannelFailure(
                code="CHAN_CONFIG_NOT_FOUND",
                message=f"Channel config not found for '{channel_id}'",
                channel_id=channel_id,
                channel_kind=channel_kind,
                session_id=session_id,
                timestamp=_now(),
            )
        return json.loads(channel_file.read_text()).get("config", {})

    def _resolve_bot_token(
        self,
        driver_config: dict[str, Any],
        channel_id: str,
        channel_kind: str,
        session_id: str,
    ) -> str:
        bot_token_ref: str | None = driver_config.get("bot_token_ref")
        if bot_token_ref is None:
            raise ChannelFailure(
                code="CHAN_BOT_TOKEN_REF_MISSING",
                message=f"Channel '{channel_id}' config missing bot_token_ref",
                channel_id=channel_id,
                channel_kind=channel_kind,
                session_id=session_id,
                timestamp=_now(),
            )
        token = self._resolver.resolve(bot_token_ref)
        if token is None:
            raise ChannelFailure(
                code="CHAN_BOT_TOKEN_UNRESOLVABLE",
                message=f"bot_token_ref '{bot_token_ref}' could not be resolved from Vault",
                channel_id=channel_id,
                channel_kind=channel_kind,
                session_id=session_id,
                timestamp=_now(),
            )
        return token

    async def _discord_post(
        self,
        url: str,
        payload: dict[str, Any],
        token: str,
    ) -> httpx.Response:
        headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
        }
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if self._http_client is not None:
            return await self._http_client.post(url, content=payload_bytes, headers=headers)
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.post(url, content=payload_bytes, headers=headers)

    def _build_message_payload(self, request: SendRequest) -> dict[str, Any]:
        """Build the Discord REST message payload from a SendRequest."""
        if request.content_type == DISCORD_EMBED_CONTENT_TYPE:
            try:
                embed_data: Any = json.loads(request.content)
            except json.JSONDecodeError as exc:
                raise ChannelFailure(
                    code="CHAN_EMBED_PARSE_FAILED",
                    message=(
                        f"Failed to parse embed JSON for channel '{request.channel_id}': {exc}"
                    ),
                    channel_id=request.channel_id,
                    channel_kind=request.channel_kind,
                    session_id=request.session_id,
                    timestamp=_now(),
                    cause=exc,
                ) from exc
            embeds = embed_data if isinstance(embed_data, list) else [embed_data]
            return {"embeds": embeds}
        return {"content": request.content}

    # ------------------------------------------------------------------
    # ChannelDriver interface
    # ------------------------------------------------------------------

    async def start(self, request: StartRequest) -> None:
        """
        Connect to the Discord Gateway and register slash commands.

        Resolves the bot token from Vault, registers any slash commands in the
        channel config, then starts a per-channel Gateway WebSocket task.  The
        task is idempotent: a second start() for the same channel_id is a no-op
        if the task is still running.
        """
        driver_config = self._load_driver_config(
            request.channel_id, request.channel_kind, request.session_id
        )
        token = self._resolve_bot_token(
            driver_config, request.channel_id, request.channel_kind, request.session_id
        )

        # Register slash commands when application_id and slash_commands are present.
        application_id: str | None = driver_config.get("application_id")
        slash_commands: list[dict[str, Any]] = driver_config.get("slash_commands", [])
        if application_id and slash_commands:
            commands_url = f"{_DISCORD_API_BASE}/applications/{application_id}/commands"
            for command in slash_commands:
                resp = await self._discord_post(commands_url, command, token)
                if not resp.is_success:
                    raise ChannelFailure(
                        code="CHAN_SLASH_CMD_FAILED",
                        message=(
                            f"Slash command '{command.get('name', '?')}' registration failed: "
                            f"HTTP {resp.status_code}"
                        ),
                        channel_id=request.channel_id,
                        channel_kind=request.channel_kind,
                        session_id=request.session_id,
                        timestamp=_now(),
                    )

        # Start gateway task per channel (idempotent).
        existing = self._gateway_tasks.get(request.channel_id)
        if existing is None or existing.done():
            intents: int = driver_config.get("intents", _DEFAULT_INTENTS)
            gw = self._gateway_client

            async def _run_gateway() -> None:
                await gw.connect(token, intents)

            task: asyncio.Task[None] = asyncio.ensure_future(_run_gateway())
            self._gateway_tasks[request.channel_id] = task

    async def send(self, request: SendRequest) -> SendResult:
        """
        Send a message to a Discord channel via the REST API.

        Routes to the thread channel when thread_id is set.  Renders embed
        payloads when content_type is application/vnd.discord.embed+json.
        Emits OTel span "discord.channel.send"; on failure marks the span
        ERROR, writes an audit-log entry, and raises ChannelFailure.
        """
        driver_config = self._load_driver_config(
            request.channel_id, request.channel_kind, request.session_id
        )

        # Thread routing: thread_id takes precedence over configured channel.
        discord_channel_id: str | None = (
            request.thread_id
            if request.thread_id is not None
            else driver_config.get("discord_channel_id")
        )
        if discord_channel_id is None:
            raise ChannelFailure(
                code="CHAN_DISCORD_CHANNEL_MISSING",
                message=f"Channel '{request.channel_id}' config missing discord_channel_id",
                channel_id=request.channel_id,
                channel_kind=request.channel_kind,
                session_id=request.session_id,
                timestamp=_now(),
            )

        token = self._resolve_bot_token(
            driver_config, request.channel_id, request.channel_kind, request.session_id
        )

        # May raise ChannelFailure for bad embed JSON — let it propagate.
        payload = self._build_message_payload(request)

        now = _now()
        url = f"{_DISCORD_API_BASE}/channels/{discord_channel_id}/messages"

        tracer = get_tracer()
        with tracer.start_as_current_span(
            "discord.channel.send",
            attributes={
                "channel.id": request.channel_id,
                "channel.kind": request.channel_kind,
                "session.id": request.session_id,
                "discord.channel_id": discord_channel_id,
                "discord.content_type": request.content_type,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="discord.channel.send.invocation",
                    code="discord_channel_send",
                    timestamp=now,
                ),
            )

            try:
                response = await self._discord_post(url, payload, token)

                if not response.is_success:
                    raise RuntimeError(
                        f"Discord API returned HTTP {response.status_code}: {response.text}"
                    )

                response_data: dict[str, Any] = response.json()
                message_id = str(response_data.get("id", f"discord_{uuid.uuid4().hex}"))

            except ChannelFailure:
                raise
            except Exception as exc:
                failure = ChannelFailure(
                    code="CHAN_SEND_FAILED",
                    message=(
                        f"Discord message delivery to channel '{discord_channel_id}' failed: {exc}"
                    ),
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
                        event="discord.channel.send.failed",
                        code=failure.code,
                        timestamp=failure.timestamp,
                        detail={
                            "channel_id": request.channel_id,
                            "session_id": request.session_id,
                            "discord_channel_id": discord_channel_id,
                            "message": failure.message,
                        },
                    )
                )
                raise failure from exc

        return SendResult(
            message_id=message_id,
            timestamp=now,
            delivered=True,
        )

    async def stop(self, request: StopRequest) -> None:
        """Cancel and await the Gateway background task for this channel."""
        task = self._gateway_tasks.pop(request.channel_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            can_send_text=True,
            can_thread=True,
            max_message_length=2000,
        )
