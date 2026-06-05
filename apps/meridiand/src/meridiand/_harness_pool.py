from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def _now() -> str:
    return datetime.now(UTC).isoformat()


_STOP_PHASES: frozenset[str] = frozenset({"idle", "paused", "terminated"})


@runtime_checkable
class _PhaseReader(Protocol):
    def current_phase(self, session_id: str) -> str: ...


class HarnessPoolError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="harness_pool_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 422


@dataclass
class _WorkerSlot:
    slot: int
    queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = field(default=None, repr=False)


class HarnessPool:
    """
    In-process pool of N harness worker tasks.

    Assignments are routed by session_id hash: slot = hash(session_id) % num_workers.
    Each slot has a persistent asyncio task draining a queue of session_ids; sessions
    mapped to the same slot run sequentially with consistent per-slot affinity.

    On start(), all worker tasks are started and every session in an active phase found
    under storage_root/sessions/ is auto-resumed via wake() to recover from SIGKILL-induced
    restarts.

    Emits OTel span "harness.pool.assign" (assign) and "harness.pool.wake" (wake) with
    session.id and harness.pool.worker_slot attributes plus a structured invocation event
    on every call.  On failure surfaces HarnessPoolError to the caller and writes the
    failure to the audit log.
    """

    def __init__(
        self,
        *,
        num_workers: int,
        run_session: Callable[[str], Awaitable[tuple[int, int, str]]],
        phase_reader: _PhaseReader,
        audit_log: AuditLog,
        storage_root: Path,
    ) -> None:
        self._num_workers = num_workers
        self._run_session = run_session
        self._phase_reader = phase_reader
        self._audit_log = audit_log
        self._storage_root = storage_root
        self._slots: list[_WorkerSlot] = [_WorkerSlot(slot=i) for i in range(num_workers)]

    def _slot_for(self, session_id: str) -> int:
        return hash(session_id) % self._num_workers

    def _load_session_traceparent(self, session_id: str) -> str:
        manifest_path = self._storage_root / "sessions" / session_id / "manifest.json"
        if not manifest_path.exists():
            return ""
        try:
            return json.loads(manifest_path.read_text()).get("traceparent", "")
        except Exception:
            return ""

    async def _worker_loop(self, slot: _WorkerSlot) -> None:
        while True:
            session_id = await slot.queue.get()
            try:
                _traceparent = self._load_session_traceparent(session_id)
                _ctx = (
                    TraceContextTextMapPropagator().extract({"traceparent": _traceparent})
                    if _traceparent
                    else None
                )
                with get_tracer().start_as_current_span(
                    "session.run_span",
                    context=_ctx,
                    attributes={"session.id": session_id},
                ):
                    await self._run_session(session_id)
            except Exception:
                pass  # run_session is responsible for its own error handling
            finally:
                slot.queue.task_done()

    async def assign(self, session_id: str) -> None:
        """Assign a new session to its hashed worker slot and enqueue it for processing."""
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "harness.pool.assign",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="harness.pool.assign.invocation",
                    code="harness_pool_assign",
                    timestamp=now,
                ),
            )
            try:
                worker_slot = self._slot_for(session_id)
                self._slots[worker_slot].queue.put_nowait(session_id)
                span.set_attribute("harness.pool.worker_slot", worker_slot)
            except Exception as exc:
                err = HarnessPoolError(
                    message=f"Pool assign failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="harness.pool.assign.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

    async def wake(self, session_id: str) -> None:
        """Resume a session in its hashed worker slot."""
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "harness.pool.wake",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="harness.pool.wake.invocation",
                    code="harness_pool_wake",
                    timestamp=now,
                ),
            )
            try:
                worker_slot = self._slot_for(session_id)
                self._slots[worker_slot].queue.put_nowait(session_id)
                span.set_attribute("harness.pool.worker_slot", worker_slot)
            except Exception as exc:
                err = HarnessPoolError(
                    message=f"Pool wake failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="harness.pool.wake.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

    async def start(self) -> None:
        """Start all worker tasks and auto-resume active sessions from storage."""
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span("harness.pool.start") as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="harness.pool.start.invocation",
                    code="harness_pool_start",
                    timestamp=now,
                ),
            )
            try:
                for slot in self._slots:
                    if slot.task is None or slot.task.done():
                        slot.task = asyncio.create_task(
                            self._worker_loop(slot),
                            name=f"harness-pool-worker-{slot.slot}",
                        )

                sessions_dir = self._storage_root / "sessions"
                if sessions_dir.exists():
                    for manifest_path in sorted(sessions_dir.glob("*/manifest.json")):
                        session_id = manifest_path.parent.name
                        try:
                            phase = self._phase_reader.current_phase(session_id)
                        except Exception:
                            continue
                        if phase not in _STOP_PHASES:
                            # HarnessPoolError already logged; continue with remaining sessions.
                            with contextlib.suppress(HarnessPoolError):
                                await self.wake(session_id)
            except Exception as exc:
                if isinstance(exc, HarnessPoolError):
                    raise
                err = HarnessPoolError(
                    message=f"Pool start failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="harness.pool.start.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message},
                    )
                )
                raise err from exc

    async def stop(self) -> None:
        """Cancel all running worker tasks."""
        for slot in self._slots:
            if slot.task is not None and not slot.task.done():
                slot.task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await slot.task
                slot.task = None
