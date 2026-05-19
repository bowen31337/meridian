"""
Slack Channel Driver: meridian.slack

Outbound: POST to Slack Web API (chat.postMessage).  Supports plain text
(content_type: text/plain) and Block Kit block payloads
(content_type: application/vnd.slack.blocks+json).  Threaded replies pass
the SendRequest.thread_id value as thread_ts to the Slack API.

Inbound: Slack Socket Mode WebSocket client (optional).  A per-channel
background task is started in start() and cancelled in stop().  The
SocketModeClient protocol abstracts the WebSocket transport; inject
NoopSocketModeClient (or a mock) for unit tests.  The real implementation
is responsible for receiving events, dispatching slash command payloads,
and parsing @-mention messages.

Bot token is resolved from Vault at call time via SecretResolver using the
channel config field bot_token_ref.  Socket Mode additionally requires
slack_app_token_ref to hold the app-level token (xapp-…).  The channel
config must also supply slack_channel_id for outbound delivery.

Every outbound send emits an OTel span "slack.channel.send" with
channel/session attributes and a structured invocation event.  On failure
the error is surfaced as ChannelFailure and an audit-log entry is written
before re-raising.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from core_errors import (
    AuditLog,
    AuditLogEntry,
    NoopAuditLog,
    StructuredEvent,
    get_tracer,
    record_invocation_event,
)
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

_SLACK_API_BASE = "https://slack.com/api"

SLACK_BLOCKS_CONTENT_TYPE = "application/vnd.slack.blocks+json"


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
class SocketModeClient(Protocol):
    """
    Abstract Slack Socket Mode WebSocket client.

    The driver calls connect() in a background task started by start(); the
    implementation handles the WSS handshake, acknowledgement protocol, and
    event dispatch for messages, slash commands, and mention payloads.
    Inject NoopSocketModeClient (or a mock) for unit tests.
    """

    async def connect(self, app_token: str, bot_token: str) -> None:
        """Connect to the Slack Socket Mode endpoint and dispatch events."""
        ...

    async def disconnect(self) -> None:
        """Gracefully close the Socket Mode connection."""
        ...


class NoopSocketModeClient:
    """No-op Socket Mode client; performs no network I/O."""

    async def connect(self, app_token: str, bot_token: str) -> None:
        pass

    async def disconnect(self) -> None:
        pass


class SlackChannelDriver(ChannelDriver):
    """
    Slack channel driver (kind: meridian.slack).

    Channel config fields (nested under "config" in the channel record):
        token_vault_ref      (required, system) Vault ref used by channel registration.
        bot_token_ref        (required) Secret ref for the Slack bot token (xoxb-…).
        slack_channel_id     (required) Slack channel or DM ID for outbound messages.
        slack_app_token_ref  (optional) Secret ref for the Socket Mode app token (xapp-…).
                             When present, start() connects via Socket Mode.
    """

    kind = "meridian.slack"

    def __init__(
        self,
        *,
        storage_root: Path,
        secret_resolver: SecretResolver | None = None,
        audit_log: AuditLog | None = None,
        http_client: httpx.AsyncClient | None = None,
        socket_mode_client: SocketModeClient | None = None,
    ) -> None:
        self._storage_root = storage_root
        self._resolver = secret_resolver or NoopSecretResolver()
        self._audit_log = audit_log or NoopAuditLog()
        self._http_client = http_client
        self._socket_mode_client: SocketModeClient = (
            socket_mode_client or NoopSocketModeClient()
        )
        self._socket_tasks: dict[str, asyncio.Task[None]] = {}

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

    async def _slack_post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        token: str,
    ) -> httpx.Response:
        url = f"{_SLACK_API_BASE}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if self._http_client is not None:
            return await self._http_client.post(url, content=payload_bytes, headers=headers)
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.post(url, content=payload_bytes, headers=headers)

    def _build_message_payload(
        self,
        request: SendRequest,
        slack_channel_id: str,
    ) -> dict[str, Any]:
        """Build the Slack chat.postMessage payload from a SendRequest."""
        payload: dict[str, Any] = {"channel": slack_channel_id}

        if request.thread_id is not None:
            payload["thread_ts"] = request.thread_id

        if request.content_type == SLACK_BLOCKS_CONTENT_TYPE:
            try:
                blocks_data: Any = json.loads(request.content)
            except json.JSONDecodeError as exc:
                raise ChannelFailure(
                    code="CHAN_BLOCKS_PARSE_FAILED",
                    message=(
                        f"Failed to parse Block Kit JSON for channel '{request.channel_id}': {exc}"
                    ),
                    channel_id=request.channel_id,
                    channel_kind=request.channel_kind,
                    session_id=request.session_id,
                    timestamp=_now(),
                    cause=exc,
                )
            blocks = blocks_data if isinstance(blocks_data, list) else [blocks_data]
            payload["blocks"] = blocks
            # Slack requires a fallback text field alongside blocks.
            payload["text"] = request.metadata.get("fallback_text", "")
        else:
            payload["text"] = request.content

        return payload

    # ------------------------------------------------------------------
    # ChannelDriver interface
    # ------------------------------------------------------------------

    async def start(self, request: StartRequest) -> None:
        """
        Resolve bot token, optionally connect via Socket Mode.

        When the channel config contains slack_app_token_ref, a per-channel
        Socket Mode background task is started.  The task is idempotent: a
        second start() for the same channel_id is a no-op if the task is
        still running.  When slack_app_token_ref is absent, start() is a
        configuration-validation no-op.
        """
        driver_config = self._load_driver_config(
            request.channel_id, request.channel_kind, request.session_id
        )
        token = self._resolve_bot_token(
            driver_config, request.channel_id, request.channel_kind, request.session_id
        )

        app_token_ref: str | None = driver_config.get("slack_app_token_ref")
        if app_token_ref is None:
            return

        app_token = self._resolver.resolve(app_token_ref)
        if app_token is None:
            raise ChannelFailure(
                code="CHAN_APP_TOKEN_UNRESOLVABLE",
                message=(
                    f"slack_app_token_ref '{app_token_ref}' could not be resolved from Vault"
                ),
                channel_id=request.channel_id,
                channel_kind=request.channel_kind,
                session_id=request.session_id,
                timestamp=_now(),
            )

        existing = self._socket_tasks.get(request.channel_id)
        if existing is not None and not existing.done():
            return

        smc = self._socket_mode_client
        _app_token = app_token
        _bot_token = token

        async def _run_socket_mode() -> None:
            await smc.connect(_app_token, _bot_token)

        task: asyncio.Task[None] = asyncio.ensure_future(_run_socket_mode())
        self._socket_tasks[request.channel_id] = task

    async def send(self, request: SendRequest) -> SendResult:
        """
        Send a message to a Slack channel via the Web API.

        Routes threaded replies via thread_ts when thread_id is set.  Renders
        Block Kit payloads when content_type is application/vnd.slack.blocks+json.
        Emits OTel span "slack.channel.send"; on failure marks the span ERROR,
        writes an audit-log entry, and raises ChannelFailure.
        """
        driver_config = self._load_driver_config(
            request.channel_id, request.channel_kind, request.session_id
        )

        slack_channel_id: str | None = driver_config.get("slack_channel_id")
        if slack_channel_id is None:
            raise ChannelFailure(
                code="CHAN_SLACK_CHANNEL_MISSING",
                message=f"Channel '{request.channel_id}' config missing slack_channel_id",
                channel_id=request.channel_id,
                channel_kind=request.channel_kind,
                session_id=request.session_id,
                timestamp=_now(),
            )

        token = self._resolve_bot_token(
            driver_config, request.channel_id, request.channel_kind, request.session_id
        )

        # May raise ChannelFailure for bad blocks JSON — let it propagate.
        payload = self._build_message_payload(request, slack_channel_id)

        now = _now()

        tracer = get_tracer()
        with tracer.start_as_current_span(
            "slack.channel.send",
            attributes={
                "channel.id": request.channel_id,
                "channel.kind": request.channel_kind,
                "session.id": request.session_id,
                "slack.channel_id": slack_channel_id,
                "slack.content_type": request.content_type,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="slack.channel.send.invocation",
                    code="slack_channel_send",
                    timestamp=now,
                ),
            )

            try:
                response = await self._slack_post("chat.postMessage", payload, token)

                if not response.is_success:
                    raise RuntimeError(
                        f"Slack API returned HTTP {response.status_code}: {response.text}"
                    )

                response_data: dict[str, Any] = response.json()
                if not response_data.get("ok"):
                    error_code = response_data.get("error", "unknown_error")
                    raise RuntimeError(f"Slack API error: {error_code}")

                message_ts: str = response_data.get(
                    "ts", f"slack_{uuid.uuid4().hex}"
                )
                message_id = f"{slack_channel_id}:{message_ts}"

            except ChannelFailure:
                raise
            except Exception as exc:
                failure = ChannelFailure(
                    code="CHAN_SEND_FAILED",
                    message=(
                        f"Slack message delivery to channel '{slack_channel_id}' failed: {exc}"
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
                        event="slack.channel.send.failed",
                        code=failure.code,
                        timestamp=failure.timestamp,
                        detail={
                            "channel_id": request.channel_id,
                            "session_id": request.session_id,
                            "slack_channel_id": slack_channel_id,
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
        """Cancel and await the Socket Mode background task for this channel."""
        task = self._socket_tasks.pop(request.channel_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            can_send_text=True,
            can_thread=True,
            max_message_length=4000,
        )
