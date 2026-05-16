from __future__ import annotations

from datetime import UTC, datetime

from opentelemetry.trace import Status, StatusCode

from ._audit import AuditLog, NoopAuditLog
from ._fs_enforcer import FilesystemEnforcer
from ._telemetry import get_tracer
from ._types import AuditLogEntry, FilesystemViolation


def _now() -> str:
    return datetime.now(UTC).isoformat()


class FilesystemGate:
    """
    Enforces FilesystemPolicy at the point of every fs.* capability invocation.

    Container and SSH backends mount only $WORKSPACE, so operations are
    physically constrained to the workspace root even before this gate runs.
    The gate provides the software enforcement layer: policy checking, OTel
    spans, and audit-log writes.

    Per invocation of ``check()``:
      1. Opens an OTel span ``"fs.access"`` with fs.path, fs.operation,
         environment.id, session.id, and agent.id attributes.
      2. Checks the operation via FilesystemEnforcer.is_allowed().
      3a. Denied  — writes audit entry at level="error", marks span ERROR,
          raises FilesystemViolation (surfaces the error message to the caller).
      3b. Allowed — writes audit entry at level="info", closes span normally.
    """

    def __init__(
        self,
        enforcer: FilesystemEnforcer,
        *,
        environment_id: str = "",
        session_id: str = "",
        agent_id: str = "",
        audit_log: AuditLog | None = None,
    ) -> None:
        self._enforcer = enforcer
        self._environment_id = environment_id
        self._session_id = session_id
        self._agent_id = agent_id
        self._audit_log = audit_log or NoopAuditLog()

    def check(self, operation: str, path: str) -> None:
        """Enforce policy, emit fs.access event; raise FilesystemViolation if denied."""
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "fs.access",
            attributes={
                "fs.path": path,
                "fs.operation": operation,
                "environment.id": self._environment_id,
                "session.id": self._session_id,
                "agent.id": self._agent_id,
            },
        ) as span:
            allowed = self._enforcer.is_allowed(operation, path)

            self._audit_log.write(
                AuditLogEntry(
                    level="info" if allowed else "error",
                    event="fs.access",
                    environment_id=self._environment_id,
                    environment_kind="",
                    session_id=self._session_id,
                    timestamp=now,
                    detail={
                        "path": path,
                        "operation": operation,
                        "allowed": allowed,
                        "agent_id": self._agent_id,
                    },
                )
            )

            if not allowed:
                span.set_status(
                    Status(StatusCode.ERROR, f"Filesystem {operation} on '{path}' denied by policy")
                )
                raise FilesystemViolation(
                    operation=operation,
                    path=path,
                    agent_id=self._agent_id,
                    environment_id=self._environment_id,
                    session_id=self._session_id,
                    timestamp=now,
                )
