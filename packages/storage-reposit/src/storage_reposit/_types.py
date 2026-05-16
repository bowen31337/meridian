from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

EventSeq = int
"""Monotonically increasing sequence number identifying a position in the event log."""


class IndexerFailure(Exception):
    """
    Structured failure raised by IndexerRuntime on any indexer operation error.
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
    """Append-only record written to the audit log on every indexer failure."""

    level: Literal["info", "warn", "error"]
    event: str
    session_id: str
    timestamp: str
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class StructuredEvent:
    """Structured event attached to every OTel span, one per operation invocation."""

    name: str
    session_id: str
    timestamp: str
    operation: str
