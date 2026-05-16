from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every authorize() invocation."""

    level: Literal["info", "warn", "error"]
    event: str
    agent_id: str
    session_id: str
    timestamp: str
    detail: dict[str, Any] | None = None


class AuditLog(ABC):
    """Write interface injected by the caller. All authorize decisions are recorded here."""

    @abstractmethod
    def write(self, entry: AuditLogEntry) -> None: ...


class NoopAuditLog(AuditLog):
    """Fallback used when the caller has not provided an AuditLog implementation."""

    def write(self, entry: AuditLogEntry) -> None:
        pass
