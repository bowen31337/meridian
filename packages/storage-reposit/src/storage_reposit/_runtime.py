from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ._audit import AuditLog, NoopAuditLog
from ._indexer import BackgroundIndexer
from ._telemetry import get_tracer, record_indexer_failure, record_invocation_event
from ._types import AuditLogEntry, IndexerFailure, StructuredEvent


@dataclass
class IndexerOptions:
    """Options supplied by the host application for each runtime call."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[IndexerFailure], None] | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class IndexerRuntime:
    """
    Thin wrapper around BackgroundIndexer that adds OTel spans, structured
    invocation events, and audit-log writes on failure.

    Instantiate once with a BackgroundIndexer, then call index_session through
    the runtime so every operation is traced and any failure is recorded in the
    audit log before being raised.
    """

    def __init__(self, indexer: BackgroundIndexer) -> None:
        self._indexer = indexer

    def _fail(
        self,
        span: object,
        failure: IndexerFailure,
        options: IndexerOptions,
        audit_event: str,
    ) -> None:
        record_indexer_failure(span, failure)  # type: ignore[arg-type]
        options.audit_log.write(
            AuditLogEntry(
                level="error",
                event=audit_event,
                session_id=failure.session_id,
                timestamp=failure.timestamp,
                detail={"code": failure.code, "message": failure.message},
            )
        )
        if options.on_error is not None:
            options.on_error(failure)

    async def index_session(
        self,
        session_id: str,
        *,
        options: IndexerOptions | None = None,
    ) -> int:
        """
        Index new events for session_id and return the count applied.

        Per-invocation:
          1. Opens OTel span "indexer.index_session" with indexer.session_id attribute.
          2. Attaches an "indexer.invocation" structured event.
          3. Dispatches to the indexer; re-audits IndexerFailure (e.g. bad NDJSON),
             wraps unexpected exceptions as INDEXER_INDEX_SESSION_FAILED.
        """
        opts = options or IndexerOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "indexer.index_session",
            attributes={"indexer.session_id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="indexer.invocation",
                    session_id=session_id,
                    timestamp=now,
                    operation="index_session",
                ),
            )

            try:
                return await self._indexer.index_session(session_id)
            except IndexerFailure as failure:
                self._fail(span, failure, opts, "indexer.index_session.failed")
                raise
            except Exception as exc:
                failure = IndexerFailure(
                    code="INDEXER_INDEX_SESSION_FAILED",
                    message=str(exc),
                    session_id=session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "indexer.index_session.failed")
                raise failure from exc
