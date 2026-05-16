from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


class EventLogFailure(Exception):
    """
    Structured failure raised by EventLogRuntime on any event-log operation error.
    Recorded on the OTel span and written to the audit log before being raised.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        session_id: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.session_id = session_id
        self.timestamp = timestamp
        self.cause = cause


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every event-log failure."""

    level: Literal["info", "warn", "error"]
    event: str
    session_id: str
    timestamp: str
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class SessionEvent:
    """One event line written to the NDJSON event log."""

    seq: int
    ts: str
    type: str
    data: dict[str, Any]
    thread_id: str | None = None


@dataclass(frozen=True)
class StructuredEvent:
    """Structured event attached to every OTel span, one per operation invocation."""

    name: str
    session_id: str
    timestamp: str
    operation: str
