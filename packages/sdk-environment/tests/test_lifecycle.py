"""
WorkerPool lifecycle conformance suite.

Covers:
  - Pool path: provision on first use (lazy); warm pool reuse on subsequent calls.
  - Pool path: OTel span "environment.pool.provision_first_use" emitted with
    "environment.pool.event" (operation=provision_first_use) on first call only.
  - Pool path: last_used_at updated after each execute call.
  - On-demand path: provision → execute → reclaim inline per call; no state kept.
  - On-demand path: OTel span "environment.pool.on_demand" emitted with
    "environment.pool.event" (operation=on_demand_provision) on every call.
  - TTL reaper: workers idle beyond idle_ttl_seconds are reclaimed by _reap_idle().
  - TTL reaper: recently-used workers are not reclaimed.
  - TTL reaper: OTel span "environment.pool.idle_reclaim" emitted with
    "environment.pool.event" (operation=idle_reclaim, reason=idle_ttl_exceeded).
  - Pool drain (stop): remaining workers reclaimed with reason=pool_drain.
  - Provision failure: EnvironmentFailure surfaced to caller; audit entry written;
    entry removed from pool so next call can re-attempt.
  - Unknown kind: EnvironmentFailure(ENV_KIND_NOT_REGISTERED) propagated.
  - Background reaper lifecycle: start() / stop() round-trip.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from sdk_environment import (
    AuditLogEntry,
    CapabilityEnvelope,
    EnvironmentDriver,
    EnvironmentFailure,
    EnvironmentRuntime,
    ExecuteRequest,
    ExecuteResult,
    FilesystemPolicy,
    NetworkPolicy,
    PoolOptions,
    ProvisionRequest,
    ReclaimRequest,
    RuntimeOptions,
    WorkerPool,
)
from sdk_environment._audit import AuditLog
from sdk_environment._pool import _WorkerEntry

from .conftest import CapturingAuditLog

# ---------------------------------------------------------------------------
# Span-capturing tracer (captures every span, not just the last one)
# ---------------------------------------------------------------------------


class _MockSpan:
    def __init__(self, name: str, attributes: dict[str, Any]) -> None:
        self.name = name
        self.attributes = dict(attributes)
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.ended = False

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exc: BaseException, **_: Any) -> None:
        pass

    def __enter__(self) -> _MockSpan:
        return self

    def __exit__(self, *_: Any) -> bool:
        self.ended = True
        return False

    def pool_events(self) -> list[dict[str, Any]]:
        return [attrs for name, attrs in self.events if name == "environment.pool.event"]

    def invocation_events(self) -> list[dict[str, Any]]:
        return [attrs for name, attrs in self.events if name == "environment.invocation"]


class SpanCapturingTracer:
    """Records every span opened during a test."""

    def __init__(self) -> None:
        self.spans: list[_MockSpan] = []

    def start_as_current_span(
        self, name: str, *, attributes: dict[str, Any] | None = None, **_: Any
    ) -> _MockSpan:
        span = _MockSpan(name, attributes or {})
        self.spans.append(span)
        return span

    def by_name(self, name: str) -> list[_MockSpan]:
        return [s for s in self.spans if s.name == name]

    def first(self, name: str) -> _MockSpan:
        return self.by_name(name)[0]


@pytest.fixture()
def span_tracer(monkeypatch: pytest.MonkeyPatch) -> SpanCapturingTracer:
    tracer = SpanCapturingTracer()
    monkeypatch.setattr("sdk_environment._runtime.get_tracer", lambda: tracer)
    monkeypatch.setattr("sdk_environment._pool.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def audit_log() -> CapturingAuditLog:
    return CapturingAuditLog()


# ---------------------------------------------------------------------------
# Driver stubs
# ---------------------------------------------------------------------------


class PoolDriver(EnvironmentDriver):
    """Long-lived pool-backed driver (on_demand=False, the default)."""

    kind = "test.pool"

    def __init__(self, *, execute_raises: Exception | None = None) -> None:
        self.provisions: list[ProvisionRequest] = []
        self.executions: list[ExecuteRequest] = []
        self.reclaims: list[ReclaimRequest] = []
        self._execute_raises = execute_raises
        self._provision_raises: Exception | None = None

    async def provision(self, request: ProvisionRequest) -> None:
        if self._provision_raises:
            raise self._provision_raises
        self.provisions.append(request)

    async def execute(self, request: ExecuteRequest) -> ExecuteResult:
        if self._execute_raises:
            raise self._execute_raises
        self.executions.append(request)
        return ExecuteResult(stdout="pool-ok", stderr="", exit_code=0, duration_ms=1.0)

    async def reclaim(self, request: ReclaimRequest) -> None:
        self.reclaims.append(request)

    def network_policy(self) -> NetworkPolicy:
        return NetworkPolicy()

    def filesystem_policy(self) -> FilesystemPolicy:
        return FilesystemPolicy(read_globs=("**",))

    def capability_envelope(self) -> CapabilityEnvelope:
        return CapabilityEnvelope()


class OnDemandDriver(EnvironmentDriver):
    """Container / serverless driver that provisions per call (on_demand=True)."""

    kind = "test.on_demand"

    @property
    def on_demand(self) -> bool:
        return True

    def __init__(self) -> None:
        self.provisions: list[ProvisionRequest] = []
        self.executions: list[ExecuteRequest] = []
        self.reclaims: list[ReclaimRequest] = []

    async def provision(self, request: ProvisionRequest) -> None:
        self.provisions.append(request)

    async def execute(self, request: ExecuteRequest) -> ExecuteResult:
        self.executions.append(request)
        return ExecuteResult(stdout="on-demand-ok", stderr="", exit_code=0, duration_ms=2.0)

    async def reclaim(self, request: ReclaimRequest) -> None:
        self.reclaims.append(request)

    def network_policy(self) -> NetworkPolicy:
        return NetworkPolicy()

    def filesystem_policy(self) -> FilesystemPolicy:
        return FilesystemPolicy()

    def capability_envelope(self) -> CapabilityEnvelope:
        return CapabilityEnvelope()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(
    *drivers: EnvironmentDriver,
    options: PoolOptions | None = None,
) -> tuple[EnvironmentRuntime, WorkerPool]:
    rt = EnvironmentRuntime()
    for d in drivers:
        rt.register(d)
    pool = WorkerPool(rt, options)
    return rt, pool


def _exec_req(kind: str, env_id: str = "env1") -> ExecuteRequest:
    return ExecuteRequest(
        environment_id=env_id,
        environment_kind=kind,
        session_id="sess1",
        command=("echo", "hi"),
    )


def _make_opts(audit: CapturingAuditLog) -> RuntimeOptions:
    return RuntimeOptions(audit_log=audit)


# ---------------------------------------------------------------------------
# Pool path — provision on first use
# ---------------------------------------------------------------------------


class TestProvisionOnFirstUse:
    async def test_first_execute_triggers_provision(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert len(driver.provisions) == 1

    async def test_second_execute_reuses_worker(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert len(driver.provisions) == 1
        assert len(driver.executions) == 2

    async def test_provision_span_emitted_on_first_call_only(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert len(span_tracer.by_name("environment.pool.provision_first_use")) == 1

    async def test_provision_first_use_span_attributes(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind, "env-attr"), _make_opts(audit_log))
        span = span_tracer.first("environment.pool.provision_first_use")
        assert span.attributes["environment.id"] == "env-attr"
        assert span.attributes["environment.kind"] == driver.kind
        assert span.attributes["session.id"] == "sess1"

    async def test_provision_first_use_pool_event_attached(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        span = span_tracer.first("environment.pool.provision_first_use")
        events = span.pool_events()
        assert len(events) == 1
        assert events[0]["operation"] == "provision_first_use"

    async def test_provision_span_ended(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        span = span_tracer.first("environment.pool.provision_first_use")
        assert span.ended

    async def test_different_env_ids_each_provisioned_once(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind, "env-a"), _make_opts(audit_log))
        await pool.execute(_exec_req(driver.kind, "env-b"), _make_opts(audit_log))
        assert len(driver.provisions) == 2

    async def test_no_audit_entries_on_success(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert audit_log.entries == []

    async def test_execute_result_returned(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        result = await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert isinstance(result, ExecuteResult)
        assert result.stdout == "pool-ok"


# ---------------------------------------------------------------------------
# Pool path — warm pool last_used_at tracking
# ---------------------------------------------------------------------------


class TestWarmPool:
    async def test_last_used_at_updated_after_execute(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        entry = pool._workers["env1"]
        first_use = entry.last_used_at

        await asyncio.sleep(0)  # yield to allow monotonic clock to advance
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert entry.last_used_at >= first_use

    async def test_worker_stays_in_pool_between_calls(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert "env1" in pool._workers
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert "env1" in pool._workers


# ---------------------------------------------------------------------------
# On-demand path (container / serverless)
# ---------------------------------------------------------------------------


class TestOnDemandPath:
    async def test_each_execute_provisions(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = OnDemandDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert len(driver.provisions) == 2

    async def test_each_execute_reclaims(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = OnDemandDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert len(driver.reclaims) == 2

    async def test_no_workers_kept_in_pool(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = OnDemandDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert pool._workers == {}

    async def test_on_demand_span_emitted_per_call(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = OnDemandDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert len(span_tracer.by_name("environment.pool.on_demand")) == 2

    async def test_on_demand_span_attributes(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = OnDemandDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind, "env-od"), _make_opts(audit_log))
        span = span_tracer.first("environment.pool.on_demand")
        assert span.attributes["environment.id"] == "env-od"
        assert span.attributes["environment.kind"] == driver.kind

    async def test_on_demand_pool_event_operation(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = OnDemandDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        span = span_tracer.first("environment.pool.on_demand")
        events = span.pool_events()
        assert len(events) == 1
        assert events[0]["operation"] == "on_demand_provision"

    async def test_on_demand_span_ended(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = OnDemandDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        span = span_tracer.first("environment.pool.on_demand")
        assert span.ended

    async def test_on_demand_execute_result_returned(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = OnDemandDriver()
        _, pool = _make_pool(driver)
        result = await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert result.stdout == "on-demand-ok"

    async def test_reclaim_called_even_when_execute_raises(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        class FailingOnDemandDriver(OnDemandDriver):
            async def execute(self, request: ExecuteRequest) -> ExecuteResult:
                raise RuntimeError("execute boom")

        driver = FailingOnDemandDriver()
        _, pool = _make_pool(driver)
        with pytest.raises(EnvironmentFailure):
            await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert len(driver.reclaims) == 1


# ---------------------------------------------------------------------------
# TTL reaper
# ---------------------------------------------------------------------------


class TestTTLReaper:
    async def test_idle_worker_reclaimed_after_ttl(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver, options=PoolOptions(idle_ttl_seconds=60))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        entry = pool._workers["env1"]
        # Backdate last_used_at to simulate idle timeout
        entry.last_used_at = time.monotonic() - 61

        await pool._reap_idle()
        assert len(driver.reclaims) == 1

    async def test_recently_used_worker_not_reclaimed(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver, options=PoolOptions(idle_ttl_seconds=60))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        # last_used_at is fresh — well within TTL
        await pool._reap_idle()
        assert len(driver.reclaims) == 0
        assert "env1" in pool._workers

    async def test_idle_entry_removed_from_pool(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver, options=PoolOptions(idle_ttl_seconds=60))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        pool._workers["env1"].last_used_at = time.monotonic() - 61

        await pool._reap_idle()
        assert "env1" not in pool._workers

    async def test_idle_reclaim_span_emitted(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver, options=PoolOptions(idle_ttl_seconds=60))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        pool._workers["env1"].last_used_at = time.monotonic() - 61

        await pool._reap_idle()
        assert len(span_tracer.by_name("environment.pool.idle_reclaim")) == 1

    async def test_idle_reclaim_span_attributes(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver, options=PoolOptions(idle_ttl_seconds=60))
        await pool.execute(_exec_req(driver.kind, "env-ttl"), _make_opts(audit_log))
        pool._workers["env-ttl"].last_used_at = time.monotonic() - 61

        await pool._reap_idle()
        span = span_tracer.first("environment.pool.idle_reclaim")
        assert span.attributes["environment.id"] == "env-ttl"
        assert span.attributes["environment.kind"] == driver.kind

    async def test_idle_reclaim_pool_event_operation_and_reason(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver, options=PoolOptions(idle_ttl_seconds=60))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        pool._workers["env1"].last_used_at = time.monotonic() - 61

        await pool._reap_idle()
        span = span_tracer.first("environment.pool.idle_reclaim")
        events = span.pool_events()
        assert len(events) == 1
        assert events[0]["operation"] == "idle_reclaim"
        assert events[0]["reason"] == "idle_ttl_exceeded"

    async def test_idle_reclaim_span_ended(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver, options=PoolOptions(idle_ttl_seconds=60))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        pool._workers["env1"].last_used_at = time.monotonic() - 61

        await pool._reap_idle()
        span = span_tracer.first("environment.pool.idle_reclaim")
        assert span.ended

    async def test_only_expired_workers_reclaimed(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver, options=PoolOptions(idle_ttl_seconds=60))
        await pool.execute(_exec_req(driver.kind, "env-old"), _make_opts(audit_log))
        await pool.execute(_exec_req(driver.kind, "env-new"), _make_opts(audit_log))
        pool._workers["env-old"].last_used_at = time.monotonic() - 61
        # env-new remains fresh

        await pool._reap_idle()
        assert len(driver.reclaims) == 1
        assert driver.reclaims[0].environment_id == "env-old"
        assert "env-new" in pool._workers


# ---------------------------------------------------------------------------
# Pool drain (stop)
# ---------------------------------------------------------------------------


class TestPoolDrain:
    async def test_stop_reclaims_all_workers(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind, "env-a"), _make_opts(audit_log))
        await pool.execute(_exec_req(driver.kind, "env-b"), _make_opts(audit_log))

        await pool.stop()
        assert len(driver.reclaims) == 2

    async def test_stop_clears_pool(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))

        await pool.stop()
        assert pool._workers == {}

    async def test_stop_drain_emits_idle_reclaim_span_with_drain_reason(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver)
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))

        await pool.stop()
        span = span_tracer.first("environment.pool.idle_reclaim")
        events = span.pool_events()
        assert events[0]["reason"] == "pool_drain"

    async def test_stop_is_safe_with_empty_pool(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        _, pool = _make_pool(PoolDriver())
        await pool.stop()  # must not raise

    async def test_start_stop_round_trip(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        _, pool = _make_pool(PoolDriver())
        await pool.start()
        assert pool._reaper_task is not None
        await pool.stop()
        assert pool._reaper_task is None


# ---------------------------------------------------------------------------
# Provision failure
# ---------------------------------------------------------------------------


class TestProvisionFailure:
    async def test_provision_failure_raised_to_caller(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        driver._provision_raises = RuntimeError("disk full")
        _, pool = _make_pool(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert exc_info.value.code == "ENV_PROVISION_FAILED"

    async def test_provision_failure_message_surfaces_to_caller(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        driver._provision_raises = RuntimeError("disk full")
        _, pool = _make_pool(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert "disk full" in exc_info.value.message

    async def test_provision_failure_writes_audit_entry(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        driver._provision_raises = RuntimeError("disk full")
        _, pool = _make_pool(driver)
        with pytest.raises(EnvironmentFailure):
            await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].level == "error"
        assert audit_log.entries[0].event == "environment.provision.failed"

    async def test_failed_entry_removed_from_pool(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        driver._provision_raises = RuntimeError("disk full")
        _, pool = _make_pool(driver)
        with pytest.raises(EnvironmentFailure):
            await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert "env1" not in pool._workers

    async def test_retry_after_provision_failure(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        driver._provision_raises = RuntimeError("transient")
        _, pool = _make_pool(driver)
        with pytest.raises(EnvironmentFailure):
            await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        # Heal the driver; next call should re-attempt provision
        driver._provision_raises = None
        result = await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))
        assert result.stdout == "pool-ok"


# ---------------------------------------------------------------------------
# Unknown kind
# ---------------------------------------------------------------------------


class TestUnknownKind:
    async def test_unknown_kind_raises_environment_failure(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        _, pool = _make_pool(PoolDriver())
        with pytest.raises(EnvironmentFailure) as exc_info:
            await pool.execute(
                ExecuteRequest(
                    environment_id="env1",
                    environment_kind="acme.unknown",
                    session_id="sess1",
                    command=("echo",),
                ),
                _make_opts(audit_log),
            )
        assert exc_info.value.code == "ENV_KIND_NOT_REGISTERED"

    async def test_unknown_kind_writes_audit_entry(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        _, pool = _make_pool(PoolDriver())
        with pytest.raises(EnvironmentFailure):
            await pool.execute(
                ExecuteRequest(
                    environment_id="env1",
                    environment_kind="acme.unknown",
                    session_id="sess1",
                    command=("echo",),
                ),
                _make_opts(audit_log),
            )
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].event == "environment.provision.failed"


# ---------------------------------------------------------------------------
# Configurable TTL
# ---------------------------------------------------------------------------


class TestConfigurableTTL:
    async def test_custom_ttl_respected(
        self, span_tracer: SpanCapturingTracer, audit_log: CapturingAuditLog
    ) -> None:
        driver = PoolDriver()
        _, pool = _make_pool(driver, options=PoolOptions(idle_ttl_seconds=120))
        await pool.execute(_exec_req(driver.kind), _make_opts(audit_log))

        # Idle for 100s — within the 120s TTL, should survive
        pool._workers["env1"].last_used_at = time.monotonic() - 100
        await pool._reap_idle()
        assert "env1" in pool._workers

        # Idle for 130s — beyond the 120s TTL, should be reaped
        pool._workers["env1"].last_used_at = time.monotonic() - 130
        await pool._reap_idle()
        assert "env1" not in pool._workers
