from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ._audit import AuditLog, NoopAuditLog
from ._contract import BlobStore
from ._telemetry import get_tracer, record_blob_failure, record_invocation_event
from ._types import AuditLogEntry, BlobFailure, StructuredEvent


@dataclass
class BlobOptions:
    """Options supplied by the host application for each runtime call."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[BlobFailure], None] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BlobRuntime:
    """
    Thin wrapper around a BlobStore that adds OTel spans, structured invocation
    events, and audit-log writes on failure.

    Instantiate once with a concrete BlobStore backend (e.g. LocalBlobStore),
    then call put / get / delete through the runtime so every operation is
    traced and any failure is recorded in the audit log before being raised.
    """

    def __init__(self, store: BlobStore) -> None:
        self._store = store

    def _fail(
        self,
        span: object,
        failure: BlobFailure,
        options: BlobOptions,
        audit_event: str,
    ) -> None:
        record_blob_failure(span, failure)  # type: ignore[arg-type]
        options.audit_log.write(
            AuditLogEntry(
                level="error",
                event=audit_event,
                key=failure.key,
                timestamp=failure.timestamp,
                detail={"code": failure.code, "message": failure.message},
            )
        )
        if options.on_error is not None:
            options.on_error(failure)

    async def put(self, key: str, data: bytes, options: BlobOptions | None = None) -> None:
        """
        Write data under key.

        Per-invocation:
          1. Opens OTel span "blob.put" with blob.key attribute.
          2. Attaches a "blob.invocation" structured event.
          3. Dispatches to the store; wraps unexpected exceptions as BLOB_PUT_FAILED.
        """
        opts = options or BlobOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "blob.put",
            attributes={"blob.key": key},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(name="blob.invocation", key=key, timestamp=now, operation="put"),
            )

            try:
                await self._store.put(key, data)
            except BlobFailure as failure:
                self._fail(span, failure, opts, "blob.put.failed")
                raise
            except Exception as exc:
                failure = BlobFailure(
                    code="BLOB_PUT_FAILED",
                    message=str(exc),
                    key=key,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "blob.put.failed")
                raise failure from exc

    async def get(self, key: str, options: BlobOptions | None = None) -> bytes:
        """
        Return the bytes stored under key.

        Per-invocation:
          1. Opens OTel span "blob.get" with blob.key attribute.
          2. Attaches a "blob.invocation" structured event.
          3. Dispatches to the store; BlobFailure(BLOB_KEY_NOT_FOUND) is audited
             and re-raised; unexpected exceptions are wrapped as BLOB_GET_FAILED.
        """
        opts = options or BlobOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "blob.get",
            attributes={"blob.key": key},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(name="blob.invocation", key=key, timestamp=now, operation="get"),
            )

            try:
                return await self._store.get(key)
            except BlobFailure as failure:
                self._fail(span, failure, opts, "blob.get.failed")
                raise
            except Exception as exc:
                failure = BlobFailure(
                    code="BLOB_GET_FAILED",
                    message=str(exc),
                    key=key,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "blob.get.failed")
                raise failure from exc

    async def delete(self, key: str, options: BlobOptions | None = None) -> None:
        """
        Remove the blob stored under key (no-op if the key does not exist).

        Per-invocation:
          1. Opens OTel span "blob.delete" with blob.key attribute.
          2. Attaches a "blob.invocation" structured event.
          3. Dispatches to the store; wraps unexpected exceptions as BLOB_DELETE_FAILED.
        """
        opts = options or BlobOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "blob.delete",
            attributes={"blob.key": key},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(name="blob.invocation", key=key, timestamp=now, operation="delete"),
            )

            try:
                await self._store.delete(key)
            except BlobFailure as failure:
                self._fail(span, failure, opts, "blob.delete.failed")
                raise
            except Exception as exc:
                failure = BlobFailure(
                    code="BLOB_DELETE_FAILED",
                    message=str(exc),
                    key=key,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "blob.delete.failed")
                raise failure from exc
