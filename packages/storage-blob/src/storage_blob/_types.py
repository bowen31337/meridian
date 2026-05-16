from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


class BlobFailure(Exception):
    """
    Structured failure raised by BlobRuntime on any blob-store operation error.
    Recorded on the OTel span and written to the audit log before being raised.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        key: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.key = key
        self.timestamp = timestamp
        self.cause = cause


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every blob-store failure."""

    level: Literal["info", "warn", "error"]
    event: str
    key: str
    timestamp: str
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class StructuredEvent:
    """Structured event attached to every OTel span, one per operation invocation."""

    name: str
    key: str
    timestamp: str
    operation: str
