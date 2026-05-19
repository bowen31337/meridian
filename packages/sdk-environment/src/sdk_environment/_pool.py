from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ._audit import NoopAuditLog
from ._runtime import EnvironmentRuntime, RuntimeOptions
from ._telemetry import get_tracer, record_pool_event
from ._types import (
    EnvironmentFailure,
    ExecuteRequest,
    ExecuteResult,
    PoolEvent,
    PoolOptions,
    ProvisionRequest,
    ReclaimRequest,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class _WorkerEntry:
    environment_id: str
    environment_kind: str
    session_id: str
    last_used_at: float = field(default_factory=time.monotonic)
    provisioned: bool = False
    _provision_error: EnvironmentFailure | None = field(default=None, repr=False)
    _provision_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


class WorkerPool:
    """
    Lifecycle manager for environment workers.

    Pool-backed kinds (on_demand=False):
      - Provision on first use (lazy); keep the worker alive across calls (warm pool).
      - Background reaper reclaims workers idle longer than idle_ttl_seconds.

    On-demand kinds (on_demand=True — container, serverless, etc.):
      - Provision → execute → reclaim inline at every tool-call.
      - No state is retained between calls.

    All provision/execute/reclaim calls pass through EnvironmentRuntime, which
    wraps each with an OTel span, a structured invocation event, and audit-log
    writes on failure.  The pool itself emits additional pool-lifecycle spans:
      - "environment.pool.provision_first_use" (pool path, first call)
      - "environment.pool.on_demand"            (on-demand path, every call)
      - "environment.pool.idle_reclaim"          (reaper, drain, and pool-size eviction)

    Pool-size enforcement (PoolOptions.max_workers):
      When set, the pool keeps at most max_workers warm workers.  When a new
      environment_id is admitted and the pool is full, the least-recently-used
      provisioned worker is evicted synchronously (reason="pool_size_eviction")
      to make room.  Reusing an existing entry never triggers eviction.
    """

    def __init__(
        self,
        runtime: EnvironmentRuntime,
        options: PoolOptions | None = None,
    ) -> None:
        self._runtime = runtime
        self._options = options or PoolOptions()
        self._workers: dict[str, _WorkerEntry] = {}
        self._dict_lock = asyncio.Lock()
        self._reaper_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        request: ExecuteRequest,
        options: RuntimeOptions | None = None,
    ) -> ExecuteResult:
        """
        Execute a command in the appropriate worker.

        On-demand backends: provision → execute → reclaim inline, no pool state kept.
        Pool-backed backends: provision on first use, then execute; update last_used_at.
        """
        opts = options or RuntimeOptions()
        driver = self._runtime.get(request.environment_kind)
        if driver is not None and driver.on_demand:
            return await self._execute_on_demand(request, opts)
        return await self._execute_pooled(request, opts)

    async def start(self) -> None:
        """Start the background idle-TTL reaper."""
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(
                self._reaper_loop(), name="env-pool-reaper"
            )

    async def stop(self) -> None:
        """Cancel the reaper and reclaim all remaining pool workers."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None
        await self._reclaim_all(reason="pool_drain")

    # ------------------------------------------------------------------
    # On-demand path (container / serverless)
    # ------------------------------------------------------------------

    async def _execute_on_demand(
        self, request: ExecuteRequest, opts: RuntimeOptions
    ) -> ExecuteResult:
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "environment.pool.on_demand",
            attributes={
                "environment.id": request.environment_id,
                "environment.kind": request.environment_kind,
                "session.id": request.session_id,
            },
        ) as span:
            record_pool_event(
                span,
                PoolEvent(
                    name="environment.pool.event",
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    operation="on_demand_provision",
                ),
            )
            provision_req = ProvisionRequest(
                environment_id=request.environment_id,
                environment_kind=request.environment_kind,
                session_id=request.session_id,
            )
            await self._runtime.provision(provision_req, opts)
            try:
                result = await self._runtime.execute(request, opts)
            finally:
                reclaim_req = ReclaimRequest(
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                )
                await self._runtime.reclaim(reclaim_req, opts)
        return result

    # ------------------------------------------------------------------
    # Pool path (provision-on-first-use / warm pool)
    # ------------------------------------------------------------------

    async def _execute_pooled(
        self, request: ExecuteRequest, opts: RuntimeOptions
    ) -> ExecuteResult:
        entry = await self._get_or_provision(request, opts)
        result = await self._runtime.execute(request, opts)
        entry.last_used_at = time.monotonic()
        return result

    def _lru_eviction_candidate(self) -> _WorkerEntry | None:
        """Return the LRU provisioned entry to evict, or None if not needed.

        Must be called with _dict_lock held.
        """
        max_w = self._options.max_workers
        if max_w is None:
            return None
        provisioned = [e for e in self._workers.values() if e.provisioned]
        if not provisioned or len(provisioned) < max_w:
            return None
        return min(provisioned, key=lambda e: e.last_used_at)

    async def _get_or_provision(
        self, request: ExecuteRequest, opts: RuntimeOptions
    ) -> _WorkerEntry:
        evict_entry: _WorkerEntry | None = None
        async with self._dict_lock:
            entry = self._workers.get(request.environment_id)
            if entry is None:
                evict_entry = self._lru_eviction_candidate()
                if evict_entry is not None:
                    self._workers.pop(evict_entry.environment_id)
                entry = _WorkerEntry(
                    environment_id=request.environment_id,
                    environment_kind=request.environment_kind,
                    session_id=request.session_id,
                )
                self._workers[request.environment_id] = entry

        if evict_entry is not None:
            await self._reclaim_entry(evict_entry, reason="pool_size_eviction")

        if not entry.provisioned:
            async with entry._provision_lock:
                if entry._provision_error is not None:
                    raise entry._provision_error
                if not entry.provisioned:
                    await self._provision_entry(entry, opts)

        return entry

    async def _provision_entry(self, entry: _WorkerEntry, opts: RuntimeOptions) -> None:
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "environment.pool.provision_first_use",
            attributes={
                "environment.id": entry.environment_id,
                "environment.kind": entry.environment_kind,
                "session.id": entry.session_id,
            },
        ) as span:
            record_pool_event(
                span,
                PoolEvent(
                    name="environment.pool.event",
                    environment_id=entry.environment_id,
                    environment_kind=entry.environment_kind,
                    session_id=entry.session_id,
                    timestamp=now,
                    operation="provision_first_use",
                ),
            )
            provision_req = ProvisionRequest(
                environment_id=entry.environment_id,
                environment_kind=entry.environment_kind,
                session_id=entry.session_id,
            )
            try:
                await self._runtime.provision(provision_req, opts)
            except EnvironmentFailure as exc:
                entry._provision_error = exc
                async with self._dict_lock:
                    self._workers.pop(entry.environment_id, None)
                raise
        entry.provisioned = True
        entry.last_used_at = time.monotonic()

    # ------------------------------------------------------------------
    # TTL reaper
    # ------------------------------------------------------------------

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(self._options.reap_interval_seconds)
            await self._reap_idle()

    async def _reap_idle(self) -> None:
        cutoff = time.monotonic() - self._options.idle_ttl_seconds
        to_reclaim: list[_WorkerEntry] = []

        async with self._dict_lock:
            expired = [
                eid
                for eid, e in self._workers.items()
                if e.provisioned and e.last_used_at < cutoff
            ]
            for eid in expired:
                to_reclaim.append(self._workers.pop(eid))

        for entry in to_reclaim:
            await self._reclaim_entry(entry, reason="idle_ttl_exceeded")

    async def _reclaim_all(self, *, reason: str) -> None:
        async with self._dict_lock:
            remaining = [e for e in self._workers.values() if e.provisioned]
            self._workers.clear()
        for entry in remaining:
            await self._reclaim_entry(entry, reason=reason)

    async def _reclaim_entry(self, entry: _WorkerEntry, *, reason: str) -> None:
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "environment.pool.idle_reclaim",
            attributes={
                "environment.id": entry.environment_id,
                "environment.kind": entry.environment_kind,
                "session.id": entry.session_id,
            },
        ) as span:
            record_pool_event(
                span,
                PoolEvent(
                    name="environment.pool.event",
                    environment_id=entry.environment_id,
                    environment_kind=entry.environment_kind,
                    session_id=entry.session_id,
                    timestamp=now,
                    operation="idle_reclaim",
                    reason=reason,
                ),
            )
            reclaim_req = ReclaimRequest(
                environment_id=entry.environment_id,
                environment_kind=entry.environment_kind,
                session_id=entry.session_id,
            )
            try:
                await self._runtime.reclaim(reclaim_req)
            except EnvironmentFailure:
                pass  # runtime already marked the span and wrote the audit entry
