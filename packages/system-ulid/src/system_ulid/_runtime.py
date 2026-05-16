from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ._audit import AuditLog, NoopAuditLog
from ._generator import generate_ulid
from ._prefixes import IdPrefix
from ._telemetry import get_tracer, record_invocation_event, record_ulid_failure
from ._types import AuditLogEntry, StructuredEvent, UlidFailure


@dataclass
class UlidOptions:
    """Options supplied by the host application for each runtime call."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[UlidFailure], None] | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class UlidRuntime:
    """
    Thin wrapper around a ULID generator that adds OTel spans, structured
    invocation events, and audit-log writes on failure.

    Instantiate once (optionally providing a custom generator), then call
    generate() so every operation is traced and any failure is recorded in
    the audit log before being raised.
    """

    def __init__(self, generator: Callable[[], str] | None = None) -> None:
        self._generator = generator or generate_ulid

    def _fail(
        self,
        span: object,
        failure: UlidFailure,
        options: UlidOptions,
        audit_event: str,
    ) -> None:
        record_ulid_failure(span, failure)  # type: ignore[arg-type]
        options.audit_log.write(
            AuditLogEntry(
                level="error",
                event=audit_event,
                prefix=failure.prefix,
                timestamp=failure.timestamp,
                detail={"code": failure.code, "message": failure.message},
            )
        )
        if options.on_error is not None:
            options.on_error(failure)

    def generate(self, prefix: IdPrefix, *, options: UlidOptions | None = None) -> str:
        """
        Generate a monotonic, URL-safe ULID with the given typed prefix.

        Per-invocation:
          1. Opens OTel span "ulid.generate" with ulid.prefix attribute.
          2. Attaches a "ulid.invocation" structured event.
          3. Calls the underlying generator; wraps unexpected exceptions as
             ULID_GENERATE_FAILED.

        Returns a string of the form "<prefix>_<26-char ULID>".
        """
        opts = options or UlidOptions()
        now = _now()
        tracer = get_tracer()
        prefix_value = prefix.value

        with tracer.start_as_current_span(
            "ulid.generate",
            attributes={"ulid.prefix": prefix_value},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="ulid.invocation",
                    prefix=prefix_value,
                    timestamp=now,
                    operation="generate",
                ),
            )

            try:
                ulid = self._generator()
                return f"{prefix_value}_{ulid}"
            except UlidFailure as failure:
                self._fail(span, failure, opts, "ulid.generate.failed")
                raise
            except Exception as exc:
                failure = UlidFailure(
                    code="ULID_GENERATE_FAILED",
                    message=str(exc),
                    prefix=prefix_value,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "ulid.generate.failed")
                raise failure from exc
