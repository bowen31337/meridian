from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel


class AuditLogEntry(BaseModel):
    """An append-only entry written on every provider failure."""

    level: Literal["info", "warning", "error"]
    event: str
    provider_name: str
    provider_kind: str
    model: str | None = None
    session_id: str | None = None
    timestamp: str
    detail: dict[str, Any] = {}


class AuditLog(Protocol):
    """Write interface injected into the ModelRouter by the host application.

    All provider failures are recorded here synchronously.  Implementations
    may write to a database, a file, or any other sink.
    """

    def write(self, entry: AuditLogEntry) -> None: ...


class NoopAuditLog:
    """Fallback used when the host has not provided an AuditLog."""

    def write(self, entry: AuditLogEntry) -> None:
        pass
