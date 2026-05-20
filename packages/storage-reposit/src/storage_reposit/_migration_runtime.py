from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ._audit import AuditLog, NoopAuditLog
from ._store import SQLiteProjectionStore
from ._telemetry import get_tracer, record_invocation_event, record_migration_failure
from ._types import AuditLogEntry, IndexerFailure, StructuredEvent


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class MigrationOptions:
    """Options supplied by the host application for each migration invocation."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[IndexerFailure], None] | None = None


class MigrationRuntime:
    """
    Thin wrapper around SQLiteProjectionStore.migrate() that adds OTel spans,
    structured invocation events, and audit-log writes on failure.

    Instantiate once with a SQLiteProjectionStore, then call migrate() through
    the runtime so every invocation is traced and any failure is recorded in the
    audit log before being raised.
    """

    def __init__(self, store: SQLiteProjectionStore) -> None:
        self._store = store

    def _fail(
        self,
        span: object,
        failure: IndexerFailure,
        options: MigrationOptions,
        audit_event: str,
    ) -> None:
        record_migration_failure(span, failure)  # type: ignore[arg-type]
        options.audit_log.write(
            AuditLogEntry(
                level="error",
                event=audit_event,
                session_id="",
                timestamp=failure.timestamp,
                detail={"code": failure.code, "message": failure.message},
            )
        )
        if options.on_error is not None:
            options.on_error(failure)

    def migrate(
        self,
        *,
        options: MigrationOptions | None = None,
    ) -> int:
        """
        Apply pending schema migrations with OTel tracing and audit logging.

        Returns the number of migrations applied.

        Per-invocation:
          1. Opens OTel span "migration.migrate".
          2. Attaches a "migration.invocation" structured event.
          3. Dispatches to the store; re-audits IndexerFailure, wraps unexpected
             exceptions as MIGRATION_FAILED and writes to the audit log.
        """
        opts = options or MigrationOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span("migration.migrate") as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="migration.invocation",
                    session_id="",
                    timestamp=now,
                    operation="migrate",
                ),
            )

            try:
                return self._store.migrate()
            except IndexerFailure as failure:
                self._fail(span, failure, opts, "migration.migrate.failed")
                raise
            except Exception as exc:
                failure = IndexerFailure(
                    code="MIGRATION_FAILED",
                    message=str(exc),
                    session_id="",
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "migration.migrate.failed")
                raise failure from exc
