"""
Telegram Channel Driver: meridian.telegram

Outbound: POST to Telegram Bot API (sendMessage).  Supports plain text
(content_type: text/plain) and Telegram MarkdownV2 payloads
(content_type: application/vnd.telegram.markdownv2).  Threaded replies pass
the SendRequest.thread_id value as reply_to_message_id to the API.

Inbound (long-poll mode): a per-channel background task calls getUpdates
with a configurable timeout.  The LongPollClient protocol abstracts the
polling loop; inject NoopLongPollClient (or a mock) for unit tests.

Inbound (webhook mode): no background task is started; the external webhook
endpoint delivers updates and mode="webhook" suppresses polling.

Bot token is resolved from Vault at call time via SecretResolver using the
channel config field bot_token_ref.  The channel config must also supply
telegram_chat_id for outbound delivery.

Every outbound send emits an OTel span "telegram.channel.send" with
channel/session attributes and a structured invocation event.  On failure
the error is surfaced as ChannelFailure and an audit-log entry is written
before re-raising.
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

_TELEGRAM_API_BASE = "https://api.telegram.org"

TELEGRAM_MARKDOWN_CONTENT_TYPE = "application/vnd.telegram.markdownv2"

_DEFAULT_POLL_TIMEOUT = 30
_MODE_LONG_POLL = "long_poll"
_MODE_WEBHOOK = "webhook"


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
class LongPollClient(Protocol):
    """
    Abstract Telegram long-poll client.

    The driver calls poll() in a background task started by start(); the
    implementation calls getUpdates in a loop with the given timeout and
    dispatches each received Update.  Inject NoopLongPollClient (or a mock)
    for unit tests.
    """

    async def poll(self, token: str, timeout: int) -> None:
        """Run the getUpdates polling loop until cancelled."""
        ...

    async def stop(self) -> None:
        """Signal the polling loop to stop gracefully."""
        ...


class NoopLongPollClient:
    """No-op long-poll client; performs no network I/O."""

    async def poll(self, token: str, timeout: int) -> None:
        pass

    async def stop(self) -> None:
        pass


class TelegramChannelDriver(ChannelDriver):
    """
    Telegram channel driver (kind: meridian.telegram).

    Channel config fields (nested under "config" in the channel record):
        token_vault_ref     (required, system) Vault ref used by channel registration.
        bot_token_ref       (required) Secret ref for the Telegram bot token.
        telegram_chat_id    (required) Telegram chat or group ID for outbound messages.
        mode                (optional) "long_poll" (default) or "webhook".
                            When "long_poll", start() launches a background polling task.
                            When "webhook", start() is a validation-only no-op.
        poll_timeout        (optional) Long-poll timeout in seconds (default: 30).
    """

    kind = "meridian.telegram"

    def __init__(
        self,
        *,
        storage_root: Path,
        secret_resolver: SecretResolver | None = None,
        audit_log: AuditLog | None = None,
        http_client: httpx.AsyncClient | None = None,
        long_poll_client: LongPollClient | None = None,
    ) -> None:
        self._storage_root = storage_root
        self._resolver = secret_resolver or NoopSecretResolver()
        self._audit_log = audit_log or NoopAuditLog()
        self._http_client = http_client
        self._long_poll_client: LongPollClient = long_poll_client or NoopLongPollClient()
        self._poll_tasks: dict[str, asyncio.Task[None]] = {}

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

    async def _telegram_post(
        self,
        method: str,
        token: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        url = f"{_TELEGRAM_API_BASE}/bot{token}/{method}"
        headers = {"Content-Type": "application/json"}
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if self._http_client is not None:
            return await self._http_client.post(url, content=payload_bytes, headers=headers)
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.post(url, content=payload_bytes, headers=headers)

    def _build_send_message_payload(
        self,
        request: SendRequest,
        telegram_chat_id: str,
    ) -> dict[str, Any]:
        """Build the Telegram sendMessage payload from a SendRequest."""
        payload: dict[str, Any] = {"chat_id": telegram_chat_id}

        if request.content_type == TELEGRAM_MARKDOWN_CONTENT_TYPE:
            payload["text"] = request.content
            payload["parse_mode"] = "MarkdownV2"
        else:
            payload["text"] = request.content

        if request.thread_id is not None:
            try:
                payload["reply_to_message_id"] = int(request.thread_id)
            except ValueError:
                payload["reply_to_message_id"] = request.thread_id

        return payload

    # ------------------------------------------------------------------
    # ChannelDriver interface
    # ------------------------------------------------------------------

    async def start(self, request: StartRequest) -> None:
        """
        Resolve bot token and, in long-poll mode, start the polling task.

        When the channel config mode is "long_poll" (default), a per-channel
        background task is started that calls getUpdates in a loop.  The task
        is idempotent: a second start() for the same channel_id is a no-op if
        the task is still running.  When mode is "webhook", start() only
        validates the config.
        """
        driver_config = self._load_driver_config(
            request.channel_id, request.channel_kind, request.session_id
        )
        token = self._resolve_bot_token(
            driver_config, request.channel_id, request.channel_kind, request.session_id
        )

        mode: str = driver_config.get("mode", _MODE_LONG_POLL)
        if mode == _MODE_WEBHOOK:
            return

        existing = self._poll_tasks.get(request.channel_id)
        if existing is not None and not existing.done():
            return

        poll_timeout: int = int(driver_config.get("poll_timeout", _DEFAULT_POLL_TIMEOUT))
        lpc = self._long_poll_client
        _token = token

        async def _run_poll() -> None:
            await lpc.poll(_token, poll_timeout)

        task: asyncio.Task[None] = asyncio.ensure_future(_run_poll())
        self._poll_tasks[request.channel_id] = task

    async def send(self, request: SendRequest) -> SendResult:
        """
        Send a message to a Telegram chat via the Bot API.

        Renders MarkdownV2 when content_type is
        application/vnd.telegram.markdownv2.  Passes reply_to_message_id
        when thread_id is set.  Emits OTel span "telegram.channel.send"; on
        failure marks the span ERROR, writes an audit-log entry, and raises
        ChannelFailure.
        """
        driver_config = self._load_driver_config(
            request.channel_id, request.channel_kind, request.session_id
        )

        telegram_chat_id: str | None = driver_config.get("telegram_chat_id")
        if telegram_chat_id is None:
            raise ChannelFailure(
                code="CHAN_TELEGRAM_CHAT_ID_MISSING",
                message=f"Channel '{request.channel_id}' config missing telegram_chat_id",
                channel_id=request.channel_id,
                channel_kind=request.channel_kind,
                session_id=request.session_id,
                timestamp=_now(),
            )

        token = self._resolve_bot_token(
            driver_config, request.channel_id, request.channel_kind, request.session_id
        )

        payload = self._build_send_message_payload(request, telegram_chat_id)
        now = _now()

        tracer = get_tracer()
        with tracer.start_as_current_span(
            "telegram.channel.send",
            attributes={
                "channel.id": request.channel_id,
                "channel.kind": request.channel_kind,
                "session.id": request.session_id,
                "telegram.chat_id": telegram_chat_id,
                "telegram.content_type": request.content_type,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="telegram.channel.send.invocation",
                    code="telegram_channel_send",
                    timestamp=now,
                ),
            )

            try:
                response = await self._telegram_post("sendMessage", token, payload)

                if not response.is_success:
                    raise RuntimeError(
                        f"Telegram API returned HTTP {response.status_code}: {response.text}"
                    )

                response_data: dict[str, Any] = response.json()
                if not response_data.get("ok"):
                    description = response_data.get("description", "unknown error")
                    raise RuntimeError(f"Telegram API error: {description}")

                result_data: dict[str, Any] = response_data.get("result", {})
                message_id = str(result_data.get("message_id", f"tg_{uuid.uuid4().hex}"))

            except ChannelFailure:
                raise
            except Exception as exc:
                failure = ChannelFailure(
                    code="CHAN_SEND_FAILED",
                    message=(
                        f"Telegram message delivery to chat '{telegram_chat_id}' failed: {exc}"
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
                        event="telegram.channel.send.failed",
                        code=failure.code,
                        timestamp=failure.timestamp,
                        detail={
                            "channel_id": request.channel_id,
                            "session_id": request.session_id,
                            "telegram_chat_id": telegram_chat_id,
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
        """Cancel and await the long-poll background task for this channel."""
        task = self._poll_tasks.pop(request.channel_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            can_send_text=True,
            can_thread=True,
            max_message_length=4096,
            rate_limit_per_minute=30,
        )
