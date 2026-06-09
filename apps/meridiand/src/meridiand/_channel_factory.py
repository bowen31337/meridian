"""Builds the in-daemon ChannelRuntime.

Registers every v1 channel driver (cli / telegram / slack / discord / webhook)
and adapts the vault SecretRefResolver to the channel SecretResolver protocol.
Wired from __main__ so the Gateway (system-channel router) is actually mounted
in the running daemon: outbound / inbound / pairing dispatch to a registered
driver by kind.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

from core_errors import AuditLog, AuditLogEntry
from sdk_channel import ChannelRuntime, StartRequest, StopRequest

from ._cli_channel_driver import CliChannelDriver
from ._discord_channel_driver import DiscordChannelDriver
from ._secret_ref import SecretRefResolver
from ._slack_channel_driver import SlackChannelDriver
from ._telegram_channel_driver import TelegramChannelDriver, TelegramLongPollClient, UpdateSink
from ._webhook_channel_driver import SecretResolver as ChannelSecretResolver, WebhookChannelDriver


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _SecretRefChannelResolver:
    """
    Adapt SecretRefResolver to the channel SecretResolver protocol.

    SecretRefResolver.resolve returns str and raises when a ref can't be
    resolved; the channel drivers expect resolve(ref) -> str | None and treat
    None as "unresolvable" (surfaced as CHAN_BOT_TOKEN_UNRESOLVABLE rather than
    an uncaught exception).
    """

    def __init__(self, inner: SecretRefResolver) -> None:
        self._inner = inner

    def resolve(self, secret_ref: str) -> str | None:
        try:
            return self._inner.resolve(secret_ref)
        except Exception:
            return None


@dataclass
class ChannelRuntimeBundle:
    """The wired runtime plus the resolver the system-channel router uses for HMAC."""

    runtime: ChannelRuntime
    secret_resolver: ChannelSecretResolver | None


def build_channel_runtime(
    *,
    storage_root: Path,
    audit_log: AuditLog,
    secret_resolver: SecretRefResolver | None = None,
    inbound_sink: UpdateSink | None = None,
) -> ChannelRuntimeBundle:
    """
    Construct a ChannelRuntime with all v1 channel drivers registered.

    When ``inbound_sink`` is provided, the Telegram driver is wired with a
    per-channel TelegramLongPollClient factory so that, once a channel is
    started, its getUpdates loop dispatches decoded messages into the sink.
    """
    channel_resolver: _SecretRefChannelResolver | None = (
        _SecretRefChannelResolver(secret_resolver) if secret_resolver is not None else None
    )

    telegram_factory: Callable[[str], TelegramLongPollClient] | None = None
    if inbound_sink is not None:
        sink = inbound_sink

        def _build_telegram_client(channel_id: str) -> TelegramLongPollClient:
            return TelegramLongPollClient(channel_id=channel_id, sink=sink)

        telegram_factory = _build_telegram_client

    runtime = ChannelRuntime()
    runtime.register(CliChannelDriver(storage_root=storage_root, audit_log=audit_log))
    runtime.register(
        TelegramChannelDriver(
            storage_root=storage_root,
            audit_log=audit_log,
            secret_resolver=channel_resolver,
            long_poll_client_factory=telegram_factory,
        )
    )
    runtime.register(
        SlackChannelDriver(
            storage_root=storage_root, audit_log=audit_log, secret_resolver=channel_resolver
        )
    )
    runtime.register(
        DiscordChannelDriver(
            storage_root=storage_root, audit_log=audit_log, secret_resolver=channel_resolver
        )
    )
    runtime.register(
        WebhookChannelDriver(
            storage_root=storage_root, audit_log=audit_log, secret_resolver=channel_resolver
        )
    )
    return ChannelRuntimeBundle(runtime=runtime, secret_resolver=channel_resolver)


async def start_configured_channels(
    *,
    runtime: ChannelRuntime,
    storage_root: Path,
    audit_log: AuditLog,
) -> list[tuple[str, str]]:
    """
    Bind every persisted channel to its platform on daemon boot.

    Reads each ``channels/*.json`` record and calls ``runtime.start`` for it
    (long-poll binds, webhook validates). Best-effort: a failure to start one
    channel is audited and does not abort boot or other channels. Returns the
    list of (channel_id, channel_kind) successfully started, for shutdown.
    """
    started: list[tuple[str, str]] = []
    channels_dir = storage_root / "channels"
    if not channels_dir.exists():
        return started
    for channel_file in sorted(channels_dir.glob("*.json")):
        try:
            record = json.loads(channel_file.read_text())
            channel_id = record["id"]
            channel_kind = record["kind"]
            await runtime.start(
                StartRequest(channel_id=channel_id, channel_kind=channel_kind, session_id="")
            )
            started.append((channel_id, channel_kind))
        except Exception as exc:  # noqa: BLE001 - one bad channel must not abort boot
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="channel.autostart.failed",
                    code="channel_autostart_failed",
                    timestamp=_now(),
                    detail={"file": channel_file.name, "message": str(exc)},
                )
            )
    return started


async def stop_channels(
    *,
    runtime: ChannelRuntime,
    started: list[tuple[str, str]],
    audit_log: AuditLog,
) -> None:
    """Unbind channels started by start_configured_channels on daemon shutdown."""
    for channel_id, channel_kind in started:
        try:
            await runtime.stop(
                StopRequest(channel_id=channel_id, channel_kind=channel_kind, session_id="")
            )
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="channel.autostop.failed",
                    code="channel_autostop_failed",
                    timestamp=_now(),
                    detail={"channel_id": channel_id, "message": str(exc)},
                )
            )
