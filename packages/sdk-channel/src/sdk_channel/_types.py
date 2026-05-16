from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ChannelCapabilities:
    """Feature and resource limits declared by a channel driver."""

    can_send_text: bool = True
    can_send_files: bool = False
    can_receive_reactions: bool = False
    can_thread: bool = False
    max_message_length: int | None = None
    rate_limit_per_minute: int | None = None


@dataclass(frozen=True)
class StartRequest:
    """Request to connect a channel instance and begin accepting messages."""

    channel_id: str
    channel_kind: str
    session_id: str


@dataclass(frozen=True)
class SendRequest:
    """Request to send a message over an active channel connection."""

    channel_id: str
    channel_kind: str
    session_id: str
    recipient: str
    content: str
    content_type: str = "text/plain"
    thread_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StopRequest:
    """Request to disconnect a channel instance and release resources."""

    channel_id: str
    channel_kind: str
    session_id: str


@dataclass(frozen=True)
class SendResult:
    """Successful outcome of a send operation."""

    message_id: str
    timestamp: str
    delivered: bool


class ChannelFailure(Exception):
    """
    Structured failure raised by the runtime on any channel operation error.
    Written to the audit log and recorded on the OTel span before being raised.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        channel_id: str,
        channel_kind: str,
        session_id: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.channel_id = channel_id
        self.channel_kind = channel_kind
        self.session_id = session_id
        self.timestamp = timestamp
        self.cause = cause


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every channel failure."""

    level: Literal["info", "warn", "error"]
    event: str
    channel_id: str
    channel_kind: str
    session_id: str
    timestamp: str
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class StructuredEvent:
    """Structured event attached to every OTel span, one per operation invocation."""

    name: str
    channel_id: str
    channel_kind: str
    session_id: str
    timestamp: str
    operation: str
