from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from opentelemetry.trace import Status, StatusCode

from ._audit import AuditLog, NoopAuditLog
from ._enforcer import NetworkEnforcer
from ._telemetry import get_tracer
from ._types import AuditLogEntry, NetworkViolation


def _now() -> str:
    return datetime.now(UTC).isoformat()


class OutboundProxyTransport:
    """
    HTTPX-compatible BaseTransport that enforces the active NetworkPolicy on
    every outbound request.

    Per request:
      1. Opens an OTel span "net.outbound" with host/agent/environment attrs.
      2. Checks the host via NetworkEnforcer.is_allowed().
      3a. Denied  — writes audit entry at level="error", marks span ERROR,
          raises NetworkViolation (surfaces the error message to the caller).
      3b. Allowed — writes audit entry at level="info", forwards to inner
          transport, closes span on return.

    Usage with httpx::

        enforcer = NetworkEnforcer(env_policy, agent_policy)
        transport = OutboundProxyTransport(enforcer, inner=httpx.HTTPTransport())
        client = httpx.Client(transport=transport)
    """

    def __init__(
        self,
        enforcer: NetworkEnforcer,
        *,
        environment_id: str = "",
        session_id: str = "",
        agent_id: str = "",
        audit_log: AuditLog | None = None,
        inner: Any | None = None,
    ) -> None:
        self._enforcer = enforcer
        self._environment_id = environment_id
        self._session_id = session_id
        self._agent_id = agent_id
        self._audit_log = audit_log or NoopAuditLog()
        self._inner = inner

    def handle_request(self, request: Any) -> Any:
        """Enforce policy, emit net.outbound event, forward or raise."""
        host: str = request.url.host
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "net.outbound",
            attributes={
                "net.host": host,
                "environment.id": self._environment_id,
                "session.id": self._session_id,
                "agent.id": self._agent_id,
            },
        ) as span:
            allowed = self._enforcer.is_allowed(host)

            self._audit_log.write(
                AuditLogEntry(
                    level="info" if allowed else "error",
                    event="net.outbound",
                    environment_id=self._environment_id,
                    environment_kind="",
                    session_id=self._session_id,
                    timestamp=now,
                    detail={
                        "host": host,
                        "allowed": allowed,
                        "agent_id": self._agent_id,
                    },
                )
            )

            if not allowed:
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        f"Outbound connection to '{host}' denied by network policy",
                    )
                )
                raise NetworkViolation(
                    host=host,
                    agent_id=self._agent_id,
                    environment_id=self._environment_id,
                    session_id=self._session_id,
                    timestamp=now,
                )

            if self._inner is not None:
                return self._inner.handle_request(request)

            raise RuntimeError("OutboundProxyTransport: no inner transport configured")
