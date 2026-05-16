from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


class MeridianError(Exception):
    """Base class for all Meridian runtime errors; carries a machine-readable code."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.timestamp = timestamp
        self.cause = cause

    def http_status(self) -> int:
        return 500

    def to_envelope(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "timestamp": self.timestamp,
            }
        }


class CapabilityDeniedError(MeridianError):
    """Raised when the caller lacks a required capability."""

    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="capability_denied", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 403


class SchemaInvalidError(MeridianError):
    """Raised when a request or payload fails schema validation."""

    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="schema_invalid", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 422


class VaultUnauthorizedError(MeridianError):
    """Raised when a vault secret access is denied."""

    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="vault_unauthorized", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 403


class BudgetExceededError(MeridianError):
    """Raised when a cost or token budget is exhausted."""

    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="budget_exceeded", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 429


class DivergenceError(MeridianError):
    """Raised when concurrent state has diverged irreconcilably."""

    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="divergence", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 409


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every handled MeridianError."""

    level: Literal["info", "warn", "error"]
    event: str
    code: str
    timestamp: str
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class StructuredEvent:
    """Structured event attached to every OTel span on error handler invocation."""

    name: str
    code: str
    timestamp: str
