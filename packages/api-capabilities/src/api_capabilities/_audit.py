from __future__ import annotations

from abc import ABC, abstractmethod

from ._types import AuditLogEntry


class AuditLog(ABC):
    """Write interface injected by the host application. All capabilities API failures
    are recorded here."""

    @abstractmethod
    def write(self, entry: AuditLogEntry) -> None: ...


class NoopAuditLog(AuditLog):
    """Fallback used when the host has not provided an AuditLog implementation."""

    def write(self, entry: AuditLogEntry) -> None:
        pass
