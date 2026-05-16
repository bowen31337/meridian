from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


class UlidFailure(Exception):
    """
    Structured failure raised by UlidRuntime on any ULID generation error.
    Recorded on the OTel span and written to the audit log before being raised.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        prefix: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.prefix = prefix
        self.timestamp = timestamp
        self.cause = cause


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every ULID generation failure."""

    level: Literal["info", "warn", "error"]
    event: str
    prefix: str
    timestamp: str
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class StructuredEvent:
    """Structured event attached to every OTel span, one per operation invocation."""

    name: str
    prefix: str
    timestamp: str
    operation: str
