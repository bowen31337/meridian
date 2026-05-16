from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ._audit import AuditLog, NoopAuditLog
from ._contract import EnvironmentDriver
from ._telemetry import get_tracer, record_environment_failure, record_invocation_event
from ._types import (
    AuditLogEntry,
    CapabilityEnvelope,
    EnvironmentFailure,
    ExecuteRequest,
    ExecuteResult,
    NetworkPolicy,
    ProvisionRequest,
    ReclaimRequest,
    StructuredEvent,
)


@dataclass
class RuntimeOptions:
    """Options supplied by the host application for each runtime call."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[EnvironmentFailure], None] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EnvironmentRuntime:
    """
    Central dispatcher for environment operations.

    Register drivers once at startup; the runtime dispatches each
    provision / execute / reclaim call to the matching driver by kind,
    wrapping every invocation with an OTel span, a structured event, and
    audit-log writes on failure.
    """

    def __init__(self) -> None:
        self._drivers: dict[str, EnvironmentDriver] = {}

    def register(self, driver: EnvironmentDriver) -> None:
        """Register a driver by its kind. Raises ValueError on duplicate."""
        kind = driver.kind
        if kind in self._drivers:
            raise ValueError(f'Environment kind "{kind}" is already registered')
        self._drivers[kind] = driver

    def get(self, kind: str) -> EnvironmentDriver | None:
        """Return the registered driver for a kind, or None."""
        return self._drivers.get(kind)

    def network_policy(self, kind: str, options: RuntimeOptions | None = None) -> NetworkPolicy:
        """
        Return the network policy for a registered kind.
        Raises EnvironmentFailure(ENV_KIND_NOT_REGISTERED) if unknown.
        """
        opts = options or RuntimeOptions()
        driver = self._drivers.get(kind)
        if driver is None:
            raise EnvironmentFailure(
                code="ENV_KIND_NOT_REGISTERED",
                message=f'No driver registered for kind "{kind}"',
                environment_id="",
                environment_kind=kind,
                session_id="",
                timestamp=_now(),
            )
        return driver.network_policy()

    def capability_envelope(self, kind: str, options: RuntimeOptions | None = None) -> CapabilityEnvelope:
        """
        Return the capability envelope for a registered kind.
        Raises EnvironmentFailure(ENV_KIND_NOT_REGISTERED) if unknown.
        """
        driver = self._drivers.get(kind)
        if driver is None:
            raise EnvironmentFailure(
                code="ENV_KIND_NOT_REGISTERED",
                message=f'No driver registered for kind "{kind}"',
                environment_id="",
                environment_kind=kind,
                session_id="",
                timestamp=_now(),
            )
        return driver.capability_envelope()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fail(
        self,
        span: object,
        failure: EnvironmentFailure,
        options: RuntimeOptions,
        audit_event: str,
    ) -> None:
        record_environment_failure(span, failure)  # type: ignore[arg-type]
        options.audit_log.write(
            AuditLogEntry(
                level="error",
                event=audit_event,
                environment_id=failure.environment_id,
                environment_kind=failure.environment_kind,
                session_id=failure.session_id,
                timestamp=failure.timestamp,
                detail={"code": failure.code, "message": failure.message},
            )
        )
        if options.on_error is not None:
            options.on_error(failure)

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    async def provision(self, request: ProvisionRequest, options: RuntimeOptions | None = None) -> None:
        """
        Provision a new environment instance.

        Per-invocation:
          1. Opens OTel span "environment.provision" with environment/session attributes.
          2. Attaches an "environment.invocation" structured event to the span.
          3. Validates that the requested kind is registered.
             On failure: records span error, writes audit log, calls on_error, raises EnvironmentFailure.
          4. Dispatches to the driver. Driver exceptions are wrapped in EnvironmentFailure
             (code ENV_PROVISION_FAILED) and handled identically to step 3.
        """
        opts = options or RuntimeOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "environment.provision",
            attributes={
                "environment.id": request.environment_id,
                "environment.kind": request.environment_kind,
                "session.id": request.session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="environment.invocation",
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    operation="provision",
                ),
            )

            driver = self._drivers.get(request.environment_kind)
            if driver is None:
                failure = EnvironmentFailure(
                    code="ENV_KIND_NOT_REGISTERED",
                    message=f'No driver registered for kind "{request.environment_kind}"',
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                    timestamp=now,
                )
                self._fail(span, failure, opts, "environment.provision.failed")
                raise failure

            try:
                await driver.provision(request)
            except EnvironmentFailure:
                raise
            except Exception as exc:
                failure = EnvironmentFailure(
                    code="ENV_PROVISION_FAILED",
                    message=str(exc),
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "environment.provision.failed")
                raise failure from exc

    async def execute(self, request: ExecuteRequest, options: RuntimeOptions | None = None) -> ExecuteResult:
        """
        Execute a command in an active environment instance.

        Per-invocation:
          1. Opens OTel span "environment.execute" with environment/session attributes.
          2. Attaches an "environment.invocation" structured event to the span.
          3. Validates that the requested kind is registered.
          4. Dispatches to the driver; wraps exceptions as ENV_EXECUTE_FAILED.
        Returns ExecuteResult on success.
        """
        opts = options or RuntimeOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "environment.execute",
            attributes={
                "environment.id": request.environment_id,
                "environment.kind": request.environment_kind,
                "session.id": request.session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="environment.invocation",
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    operation="execute",
                ),
            )

            driver = self._drivers.get(request.environment_kind)
            if driver is None:
                failure = EnvironmentFailure(
                    code="ENV_KIND_NOT_REGISTERED",
                    message=f'No driver registered for kind "{request.environment_kind}"',
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                    timestamp=now,
                )
                self._fail(span, failure, opts, "environment.execute.failed")
                raise failure

            try:
                return await driver.execute(request)
            except EnvironmentFailure:
                raise
            except Exception as exc:
                failure = EnvironmentFailure(
                    code="ENV_EXECUTE_FAILED",
                    message=str(exc),
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "environment.execute.failed")
                raise failure from exc

    async def reclaim(self, request: ReclaimRequest, options: RuntimeOptions | None = None) -> None:
        """
        Reclaim (destroy) an environment instance.

        Per-invocation:
          1. Opens OTel span "environment.reclaim" with environment/session attributes.
          2. Attaches an "environment.invocation" structured event to the span.
          3. Validates that the requested kind is registered.
          4. Dispatches to the driver; wraps exceptions as ENV_RECLAIM_FAILED.
        """
        opts = options or RuntimeOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "environment.reclaim",
            attributes={
                "environment.id": request.environment_id,
                "environment.kind": request.environment_kind,
                "session.id": request.session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="environment.invocation",
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    operation="reclaim",
                ),
            )

            driver = self._drivers.get(request.environment_kind)
            if driver is None:
                failure = EnvironmentFailure(
                    code="ENV_KIND_NOT_REGISTERED",
                    message=f'No driver registered for kind "{request.environment_kind}"',
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                    timestamp=now,
                )
                self._fail(span, failure, opts, "environment.reclaim.failed")
                raise failure

            try:
                await driver.reclaim(request)
            except EnvironmentFailure:
                raise
            except Exception as exc:
                failure = EnvironmentFailure(
                    code="ENV_RECLAIM_FAILED",
                    message=str(exc),
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "environment.reclaim.failed")
                raise failure from exc


default_runtime = EnvironmentRuntime()
