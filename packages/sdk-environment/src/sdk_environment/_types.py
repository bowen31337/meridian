from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class NetworkPolicy:
    """Egress/ingress policy for an environment kind."""

    egress_allowed: bool = False
    allowed_hosts: tuple[str, ...] = ()
    blocked_hosts: tuple[str, ...] = ()
    max_bandwidth_mbps: int | None = None


@dataclass(frozen=True)
class CapabilityEnvelope:
    """Resource limits and permission set for an environment kind."""

    cpu_millicores: int = 1000
    memory_mb: int = 512
    disk_mb: int = 1024
    timeout_seconds: int = 30
    can_write_filesystem: bool = True
    can_exec_subprocesses: bool = True
    network: NetworkPolicy = field(default_factory=NetworkPolicy)


@dataclass(frozen=True)
class ProvisionRequest:
    """Request to allocate a new environment instance."""

    environment_id: str
    environment_kind: str
    session_id: str


@dataclass(frozen=True)
class ExecuteRequest:
    """Request to run a command inside an active environment."""

    environment_id: str
    environment_kind: str
    session_id: str
    command: tuple[str, ...]
    stdin: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class ReclaimRequest:
    """Request to destroy and release an environment instance."""

    environment_id: str
    environment_kind: str
    session_id: str


@dataclass(frozen=True)
class ExecuteResult:
    """Successful outcome of an execute operation."""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float


class EnvironmentFailure(Exception):
    """
    Structured failure raised by the runtime on any operation error.
    Written to the audit log and recorded on the OTel span before being raised.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        environment_id: str,
        environment_kind: str,
        session_id: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.environment_id = environment_id
        self.environment_kind = environment_kind
        self.session_id = session_id
        self.timestamp = timestamp
        self.cause = cause


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every environment failure."""

    level: Literal["info", "warn", "error"]
    event: str
    environment_id: str
    environment_kind: str
    session_id: str
    timestamp: str
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class StructuredEvent:
    """Structured event attached to every OTel span, one per operation invocation."""

    name: str
    environment_id: str
    environment_kind: str
    session_id: str
    timestamp: str
    operation: str
