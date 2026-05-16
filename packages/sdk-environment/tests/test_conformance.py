"""
Environment conformance suite.

Every implementation of EnvironmentDriver must satisfy these tests when
exercised through EnvironmentRuntime. The suite covers:

  - Successful provision / execute / reclaim: span emitted, invocation event
    attached, no audit entries, correct results returned.
  - Unknown kind (ENV_KIND_NOT_REGISTERED): EnvironmentFailure raised, audit
    entry written at level "error", span status set to ERROR.
  - Driver exceptions (ENV_PROVISION_FAILED / ENV_EXECUTE_FAILED /
    ENV_RECLAIM_FAILED): wrapped in EnvironmentFailure with cause, audit entry
    written, span marked ERROR, on_error callback called.
  - network_policy / capability_envelope retrieval.
  - Duplicate registration guard.
  - on_error callback invocation.
  - Span lifecycle: span ended on both success and failure paths.
"""
from __future__ import annotations

import pytest

from sdk_environment import (
    AuditLogEntry,
    CapabilityEnvelope,
    EnvironmentDriver,
    EnvironmentFailure,
    EnvironmentRuntime,
    ExecuteRequest,
    ExecuteResult,
    NetworkPolicy,
    ProvisionRequest,
    ReclaimRequest,
    RuntimeOptions,
)
from opentelemetry.trace import StatusCode

from .conftest import CapturingAuditLog, MockSpan


# ---------------------------------------------------------------------------
# Stub driver
# ---------------------------------------------------------------------------

class StubDriver(EnvironmentDriver):
    kind = "test.stub"

    def __init__(
        self,
        *,
        provision_raises: Exception | None = None,
        execute_raises: Exception | None = None,
        reclaim_raises: Exception | None = None,
    ) -> None:
        self._provision_raises = provision_raises
        self._execute_raises = execute_raises
        self._reclaim_raises = reclaim_raises
        self.provisions: list[ProvisionRequest] = []
        self.executions: list[ExecuteRequest] = []
        self.reclaims: list[ReclaimRequest] = []

    async def provision(self, request: ProvisionRequest) -> None:
        if self._provision_raises:
            raise self._provision_raises
        self.provisions.append(request)

    async def execute(self, request: ExecuteRequest) -> ExecuteResult:
        if self._execute_raises:
            raise self._execute_raises
        self.executions.append(request)
        return ExecuteResult(stdout="ok", stderr="", exit_code=0, duration_ms=1.0)

    async def reclaim(self, request: ReclaimRequest) -> None:
        if self._reclaim_raises:
            raise self._reclaim_raises
        self.reclaims.append(request)

    def network_policy(self) -> NetworkPolicy:
        return NetworkPolicy(egress_allowed=True, allowed_hosts=("example.com",))

    def capability_envelope(self) -> CapabilityEnvelope:
        return CapabilityEnvelope(cpu_millicores=500, memory_mb=256)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_provision(kind: str = "test.stub") -> ProvisionRequest:
    return ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="sess1")


def make_execute(kind: str = "test.stub") -> ExecuteRequest:
    return ExecuteRequest(
        environment_id="env1",
        environment_kind=kind,
        session_id="sess1",
        command=("echo", "hello"),
    )


def make_reclaim(kind: str = "test.stub") -> ReclaimRequest:
    return ReclaimRequest(environment_id="env1", environment_kind=kind, session_id="sess1")


def make_options(audit: CapturingAuditLog, errors: list[EnvironmentFailure] | None = None) -> RuntimeOptions:
    return RuntimeOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


def registered_runtime() -> EnvironmentRuntime:
    rt = EnvironmentRuntime()
    rt.register(StubDriver())
    return rt


# ---------------------------------------------------------------------------
# provision — success
# ---------------------------------------------------------------------------

class TestProvisionSuccess:
    async def test_dispatches_to_driver(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        await rt.provision(make_provision(), make_options(audit_log))
        assert len(driver.provisions) == 1

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().provision(make_provision(), make_options(audit_log))
        assert mock_span.name == "environment.provision"

    async def test_span_attributes(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().provision(make_provision(), make_options(audit_log))
        assert mock_span.attributes["environment.id"] == "env1"
        assert mock_span.attributes["environment.kind"] == "test.stub"
        assert mock_span.attributes["session.id"] == "sess1"

    async def test_invocation_event_attached(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().provision(make_provision(), make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "environment.invocation" in event_names

    async def test_invocation_event_operation(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().provision(make_provision(), make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "environment.invocation")
        assert inv[1]["operation"] == "provision"

    async def test_no_audit_entries_on_success(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().provision(make_provision(), make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().provision(make_provision(), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# provision — ENV_KIND_NOT_REGISTERED
# ---------------------------------------------------------------------------

class TestProvisionUnknownKind:
    async def test_raises_environment_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log))
        assert exc_info.value.code == "ENV_KIND_NOT_REGISTERED"

    async def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "environment.provision.failed"

    async def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log))
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_error_event_on_span(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "environment.error" in event_names

    async def test_on_error_callback(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        errors: list[EnvironmentFailure] = []
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "ENV_KIND_NOT_REGISTERED"

    async def test_span_ended_on_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# provision — driver raises
# ---------------------------------------------------------------------------

class TestProvisionDriverRaises:
    async def test_wraps_as_provision_failed(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(provision_raises=RuntimeError("disk full"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.provision(make_provision(), make_options(audit_log))
        assert exc_info.value.code == "ENV_PROVISION_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = RuntimeError("disk full")
        driver = StubDriver(provision_raises=orig)
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.provision(make_provision(), make_options(audit_log))
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(provision_raises=RuntimeError("boom"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision(), make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].level == "error"

    async def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(provision_raises=RuntimeError("boom"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision(), make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_exception_recorded_on_span(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = RuntimeError("boom")
        driver = StubDriver(provision_raises=orig)
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision(), make_options(audit_log))
        assert orig in mock_span.recorded_exceptions


# ---------------------------------------------------------------------------
# execute — success
# ---------------------------------------------------------------------------

class TestExecuteSuccess:
    async def test_returns_result(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        result = await registered_runtime().execute(make_execute(), make_options(audit_log))
        assert result.stdout == "ok"
        assert result.exit_code == 0

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().execute(make_execute(), make_options(audit_log))
        assert mock_span.name == "environment.execute"

    async def test_invocation_event_operation(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().execute(make_execute(), make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "environment.invocation")
        assert inv[1]["operation"] == "execute"

    async def test_no_audit_entries_on_success(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().execute(make_execute(), make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().execute(make_execute(), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# execute — ENV_KIND_NOT_REGISTERED
# ---------------------------------------------------------------------------

class TestExecuteUnknownKind:
    async def test_raises_environment_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute("acme.unknown"), make_options(audit_log))
        assert exc_info.value.code == "ENV_KIND_NOT_REGISTERED"

    async def test_audit_event_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute("acme.unknown"), make_options(audit_log))
        assert audit_log.entries[0].event == "environment.execute.failed"

    async def test_span_marked_error(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute("acme.unknown"), make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# execute — driver raises
# ---------------------------------------------------------------------------

class TestExecuteDriverRaises:
    async def test_wraps_as_execute_failed(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(execute_raises=RuntimeError("timeout"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute(), make_options(audit_log))
        assert exc_info.value.code == "ENV_EXECUTE_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = RuntimeError("timeout")
        driver = StubDriver(execute_raises=orig)
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute(), make_options(audit_log))
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(execute_raises=RuntimeError("boom"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute(), make_options(audit_log))
        assert len(audit_log.entries) == 1


# ---------------------------------------------------------------------------
# reclaim — success
# ---------------------------------------------------------------------------

class TestReclaimSuccess:
    async def test_dispatches_to_driver(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        await rt.reclaim(make_reclaim(), make_options(audit_log))
        assert len(driver.reclaims) == 1

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().reclaim(make_reclaim(), make_options(audit_log))
        assert mock_span.name == "environment.reclaim"

    async def test_invocation_event_operation(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().reclaim(make_reclaim(), make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "environment.invocation")
        assert inv[1]["operation"] == "reclaim"

    async def test_no_audit_entries_on_success(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().reclaim(make_reclaim(), make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().reclaim(make_reclaim(), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# reclaim — ENV_KIND_NOT_REGISTERED
# ---------------------------------------------------------------------------

class TestReclaimUnknownKind:
    async def test_raises_environment_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.reclaim(make_reclaim("acme.unknown"), make_options(audit_log))
        assert exc_info.value.code == "ENV_KIND_NOT_REGISTERED"

    async def test_audit_event_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.reclaim(make_reclaim("acme.unknown"), make_options(audit_log))
        assert audit_log.entries[0].event == "environment.reclaim.failed"


# ---------------------------------------------------------------------------
# reclaim — driver raises
# ---------------------------------------------------------------------------

class TestReclaimDriverRaises:
    async def test_wraps_as_reclaim_failed(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(reclaim_raises=RuntimeError("still running"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.reclaim(make_reclaim(), make_options(audit_log))
        assert exc_info.value.code == "ENV_RECLAIM_FAILED"

    async def test_audit_entry_written(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = StubDriver(reclaim_raises=RuntimeError("boom"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.reclaim(make_reclaim(), make_options(audit_log))
        assert len(audit_log.entries) == 1


# ---------------------------------------------------------------------------
# network_policy / capability_envelope
# ---------------------------------------------------------------------------

class TestDriverProperties:
    def test_network_policy_registered(self) -> None:
        rt = registered_runtime()
        policy = rt.network_policy("test.stub")
        assert isinstance(policy, NetworkPolicy)
        assert policy.egress_allowed is True
        assert "example.com" in policy.allowed_hosts

    def test_network_policy_unknown_kind(self) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure) as exc_info:
            rt.network_policy("acme.unknown")
        assert exc_info.value.code == "ENV_KIND_NOT_REGISTERED"

    def test_capability_envelope_registered(self) -> None:
        rt = registered_runtime()
        caps = rt.capability_envelope("test.stub")
        assert isinstance(caps, CapabilityEnvelope)
        assert caps.cpu_millicores == 500
        assert caps.memory_mb == 256

    def test_capability_envelope_unknown_kind(self) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure) as exc_info:
            rt.capability_envelope("acme.unknown")
        assert exc_info.value.code == "ENV_KIND_NOT_REGISTERED"


# ---------------------------------------------------------------------------
# Registry guard
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_duplicate_registration_raises(self) -> None:
        rt = EnvironmentRuntime()
        rt.register(StubDriver())
        with pytest.raises(ValueError, match="already registered"):
            rt.register(StubDriver())

    def test_get_returns_driver(self) -> None:
        rt = EnvironmentRuntime()
        driver = StubDriver()
        rt.register(driver)
        assert rt.get("test.stub") is driver

    def test_get_returns_none_for_unknown(self) -> None:
        rt = EnvironmentRuntime()
        assert rt.get("acme.unknown") is None
