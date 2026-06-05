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
  - Schema-valid round-trip: all fields on reference request types survive
    intact to the driver; ExecuteResult fields are the correct Python types.
  - Capability enforcement: CapabilityEnvelope fields are well-typed; driver
    violations (memory cap, subprocess denial) wrap as ENV_EXECUTE_FAILED and
    write an audit entry.
  - Timeout behavior: timeout_seconds (including None) is forwarded verbatim;
    TimeoutError from the driver wraps as ENV_EXECUTE_FAILED, surfaces the
    message to the caller, and writes the audit entry.
  - Scratch directory isolation: different environment_ids receive independent
    scratch paths; the same environment_id reuses its path across calls; reclaim
    releases the scratch entry.
  - Env-var scoping: env dict from ExecuteRequest is forwarded verbatim; two
    concurrent executions have independent env dicts with no bleed.
  - Network policy honored: NetworkEnforcer correctly allows/denies by policy;
    a NetworkViolation raised by a driver wraps as ENV_EXECUTE_FAILED and writes
    the audit entry.
  - Backend conformance (parameterized): every registered EnvironmentDriver
    implementation must pass the core lifecycle, OTel, and audit contracts.
"""

from __future__ import annotations

from datetime import UTC, datetime

from opentelemetry.trace import StatusCode
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
    NetworkEnforcer,
    NetworkPolicy,
    NetworkViolation,
    ProvisionRequest,
    ReclaimRequest,
    RuntimeOptions,
)

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

    def filesystem_policy(self) -> FilesystemPolicy:
        return FilesystemPolicy(read_globs=("**",), write_globs=("**",))

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


def make_options(
    audit: CapturingAuditLog, errors: list[EnvironmentFailure] | None = None
) -> RuntimeOptions:
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
    async def test_dispatches_to_driver(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
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

    async def test_invocation_event_attached(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await registered_runtime().provision(make_provision(), make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "environment.invocation" in event_names

    async def test_invocation_event_operation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await registered_runtime().provision(make_provision(), make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "environment.invocation")
        assert inv[1]["operation"] == "provision"

    async def test_no_audit_entries_on_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await registered_runtime().provision(make_provision(), make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().provision(make_provision(), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# provision — ENV_KIND_NOT_REGISTERED
# ---------------------------------------------------------------------------


class TestProvisionUnknownKind:
    async def test_raises_environment_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log))
        assert exc_info.value.code == "ENV_KIND_NOT_REGISTERED"

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "environment.provision.failed"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log))
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_error_event_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "environment.error" in event_names

    async def test_on_error_callback(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[EnvironmentFailure] = []
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "ENV_KIND_NOT_REGISTERED"

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision("acme.unknown"), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# provision — driver raises
# ---------------------------------------------------------------------------


class TestProvisionDriverRaises:
    async def test_wraps_as_provision_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
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

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(provision_raises=RuntimeError("boom"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision(), make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].level == "error"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(provision_raises=RuntimeError("boom"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.provision(make_provision(), make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_exception_recorded_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
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

    async def test_invocation_event_operation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await registered_runtime().execute(make_execute(), make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "environment.invocation")
        assert inv[1]["operation"] == "execute"

    async def test_no_audit_entries_on_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await registered_runtime().execute(make_execute(), make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().execute(make_execute(), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# execute — ENV_KIND_NOT_REGISTERED
# ---------------------------------------------------------------------------


class TestExecuteUnknownKind:
    async def test_raises_environment_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute("acme.unknown"), make_options(audit_log))
        assert exc_info.value.code == "ENV_KIND_NOT_REGISTERED"

    async def test_audit_event_name(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute("acme.unknown"), make_options(audit_log))
        assert audit_log.entries[0].event == "environment.execute.failed"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute("acme.unknown"), make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# execute — driver raises
# ---------------------------------------------------------------------------


class TestExecuteDriverRaises:
    async def test_wraps_as_execute_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
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

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
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
    async def test_dispatches_to_driver(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        await rt.reclaim(make_reclaim(), make_options(audit_log))
        assert len(driver.reclaims) == 1

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().reclaim(make_reclaim(), make_options(audit_log))
        assert mock_span.name == "environment.reclaim"

    async def test_invocation_event_operation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await registered_runtime().reclaim(make_reclaim(), make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "environment.invocation")
        assert inv[1]["operation"] == "reclaim"

    async def test_no_audit_entries_on_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await registered_runtime().reclaim(make_reclaim(), make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_runtime().reclaim(make_reclaim(), make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# reclaim — ENV_KIND_NOT_REGISTERED
# ---------------------------------------------------------------------------


class TestReclaimUnknownKind:
    async def test_raises_environment_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.reclaim(make_reclaim("acme.unknown"), make_options(audit_log))
        assert exc_info.value.code == "ENV_KIND_NOT_REGISTERED"

    async def test_audit_event_name(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = EnvironmentRuntime()
        with pytest.raises(EnvironmentFailure):
            await rt.reclaim(make_reclaim("acme.unknown"), make_options(audit_log))
        assert audit_log.entries[0].event == "environment.reclaim.failed"


# ---------------------------------------------------------------------------
# reclaim — driver raises
# ---------------------------------------------------------------------------


class TestReclaimDriverRaises:
    async def test_wraps_as_reclaim_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(reclaim_raises=RuntimeError("still running"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.reclaim(make_reclaim(), make_options(audit_log))
        assert exc_info.value.code == "ENV_RECLAIM_FAILED"

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
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


# ---------------------------------------------------------------------------
# Second backend: ScratchTrackingDriver
# ---------------------------------------------------------------------------


class ScratchTrackingDriver(EnvironmentDriver):
    """Backend that assigns an isolated scratch path per environment_id.

    execute() echoes the scratch path as stdout so tests can assert isolation.
    reclaim() removes the scratch entry to simulate cleanup.
    """

    kind = "test.scratch"

    def __init__(self) -> None:
        self._scratch: dict[str, str] = {}
        self.reclaimed: set[str] = set()
        self._execute_raises: Exception | None = None

    async def provision(self, request: ProvisionRequest) -> None:
        self._scratch[request.environment_id] = f"/scratch/{request.environment_id}"

    async def execute(self, request: ExecuteRequest) -> ExecuteResult:
        if self._execute_raises:
            raise self._execute_raises
        path = self._scratch.get(request.environment_id, "")
        return ExecuteResult(stdout=path, stderr="", exit_code=0, duration_ms=1.0)

    async def reclaim(self, request: ReclaimRequest) -> None:
        self.reclaimed.add(request.environment_id)
        self._scratch.pop(request.environment_id, None)

    def network_policy(self) -> NetworkPolicy:
        return NetworkPolicy(egress_allowed=False, allowed_hosts=("internal.example",))

    def filesystem_policy(self) -> FilesystemPolicy:
        return FilesystemPolicy(read_globs=("**",), write_globs=("**",), delete_globs=())

    def capability_envelope(self) -> CapabilityEnvelope:
        return CapabilityEnvelope(cpu_millicores=2000, memory_mb=1024, timeout_seconds=60)


# ---------------------------------------------------------------------------
# Schema-valid round-trip
# ---------------------------------------------------------------------------


class TestSchemaRoundTrip:
    """All fields on reference request types survive intact to the driver;
    ExecuteResult fields are the correct Python types on the way back."""

    async def test_provision_all_fields_reach_driver(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req = ProvisionRequest(
            environment_id="env-schema-1",
            environment_kind="test.stub",
            session_id="sess-schema-1",
        )
        await rt.provision(req, make_options(audit_log))
        received = driver.provisions[0]
        assert received.environment_id == "env-schema-1"
        assert received.environment_kind == "test.stub"
        assert received.session_id == "sess-schema-1"

    async def test_execute_all_fields_reach_driver(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req = ExecuteRequest(
            environment_id="env-schema-2",
            environment_kind="test.stub",
            session_id="sess-schema-2",
            command=("python", "-c", "print('hi')"),
            stdin="stdin-data",
            env={"VAR_A": "alpha", "VAR_B": "beta"},
            timeout_seconds=99,
        )
        await rt.execute(req, make_options(audit_log))
        received = driver.executions[0]
        assert received.environment_id == "env-schema-2"
        assert received.environment_kind == "test.stub"
        assert received.session_id == "sess-schema-2"
        assert received.command == ("python", "-c", "print('hi')")
        assert received.stdin == "stdin-data"
        assert received.env == {"VAR_A": "alpha", "VAR_B": "beta"}
        assert received.timeout_seconds == 99

    async def test_reclaim_all_fields_reach_driver(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req = ReclaimRequest(
            environment_id="env-schema-3",
            environment_kind="test.stub",
            session_id="sess-schema-3",
        )
        await rt.reclaim(req, make_options(audit_log))
        received = driver.reclaims[0]
        assert received.environment_id == "env-schema-3"
        assert received.environment_kind == "test.stub"
        assert received.session_id == "sess-schema-3"

    async def test_execute_result_stdout_is_str(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        result = await registered_runtime().execute(make_execute(), make_options(audit_log))
        assert isinstance(result.stdout, str)

    async def test_execute_result_stderr_is_str(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        result = await registered_runtime().execute(make_execute(), make_options(audit_log))
        assert isinstance(result.stderr, str)

    async def test_execute_result_exit_code_is_int(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        result = await registered_runtime().execute(make_execute(), make_options(audit_log))
        assert isinstance(result.exit_code, int)

    async def test_execute_result_duration_ms_is_float(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        result = await registered_runtime().execute(make_execute(), make_options(audit_log))
        assert isinstance(result.duration_ms, float)

    async def test_request_identity_preserved_to_driver(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req = make_execute()
        await rt.execute(req, make_options(audit_log))
        assert driver.executions[0] is req


# ---------------------------------------------------------------------------
# Capability enforcement
# ---------------------------------------------------------------------------


class TestCapabilityEnforcement:
    """CapabilityEnvelope fields are well-typed; driver violations wrap correctly."""

    def test_capability_envelope_cpu_millicores_is_int(self) -> None:
        caps = registered_runtime().capability_envelope("test.stub")
        assert isinstance(caps.cpu_millicores, int)

    def test_capability_envelope_memory_mb_is_int(self) -> None:
        caps = registered_runtime().capability_envelope("test.stub")
        assert isinstance(caps.memory_mb, int)

    def test_capability_envelope_disk_mb_is_int(self) -> None:
        caps = registered_runtime().capability_envelope("test.stub")
        assert isinstance(caps.disk_mb, int)

    def test_capability_envelope_timeout_seconds_is_positive_int(self) -> None:
        caps = registered_runtime().capability_envelope("test.stub")
        assert isinstance(caps.timeout_seconds, int)
        assert caps.timeout_seconds > 0

    def test_capability_envelope_can_write_filesystem_is_bool(self) -> None:
        caps = registered_runtime().capability_envelope("test.stub")
        assert isinstance(caps.can_write_filesystem, bool)

    def test_capability_envelope_can_exec_subprocesses_is_bool(self) -> None:
        caps = registered_runtime().capability_envelope("test.stub")
        assert isinstance(caps.can_exec_subprocesses, bool)

    def test_capability_envelope_network_is_network_policy(self) -> None:
        caps = registered_runtime().capability_envelope("test.stub")
        assert isinstance(caps.network, NetworkPolicy)

    def test_capability_envelope_filesystem_is_filesystem_policy(self) -> None:
        caps = registered_runtime().capability_envelope("test.stub")
        assert isinstance(caps.filesystem, FilesystemPolicy)

    async def test_memory_cap_exceeded_wraps_as_execute_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(execute_raises=MemoryError("memory cap exceeded"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute(), make_options(audit_log))
        assert exc_info.value.code == "ENV_EXECUTE_FAILED"

    async def test_cap_violation_message_surfaces_to_caller(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(execute_raises=MemoryError("memory cap exceeded"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute(), make_options(audit_log))
        assert "memory cap exceeded" in exc_info.value.message

    async def test_cap_violation_writes_audit_entry(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(execute_raises=PermissionError("subprocess execution denied"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute(), make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].event == "environment.execute.failed"

    async def test_cap_violation_marks_span_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(execute_raises=MemoryError("oom"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute(), make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Timeout behavior
# ---------------------------------------------------------------------------


class TestTimeoutBehavior:
    """timeout_seconds is forwarded verbatim; TimeoutError surfaces correctly."""

    async def test_timeout_seconds_reaches_driver(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req = ExecuteRequest(
            environment_id="env1",
            environment_kind="test.stub",
            session_id="sess1",
            command=("sleep", "10"),
            timeout_seconds=5,
        )
        await rt.execute(req, make_options(audit_log))
        assert driver.executions[0].timeout_seconds == 5

    async def test_none_timeout_reaches_driver(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req = ExecuteRequest(
            environment_id="env1",
            environment_kind="test.stub",
            session_id="sess1",
            command=("echo", "hi"),
            timeout_seconds=None,
        )
        await rt.execute(req, make_options(audit_log))
        assert driver.executions[0].timeout_seconds is None

    async def test_timeout_error_wraps_as_execute_timeout(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(execute_raises=TimeoutError("command timed out"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute(), make_options(audit_log))
        assert exc_info.value.code == "ENV_EXECUTE_TIMEOUT"

    async def test_timeout_message_surfaces_to_caller(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(execute_raises=TimeoutError("timed out after 5s"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute(), make_options(audit_log))
        assert "timed out after 5s" in exc_info.value.message

    async def test_timeout_error_writes_audit_entry(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(execute_raises=TimeoutError("command timed out"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute(), make_options(audit_log))
        assert len(audit_log.entries) == 1

    async def test_timeout_error_marks_span_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver(execute_raises=TimeoutError("command timed out"))
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute(), make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_timeout_cause_preserved(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = TimeoutError("deadline exceeded")
        driver = StubDriver(execute_raises=orig)
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute(), make_options(audit_log))
        assert exc_info.value.cause is orig


# ---------------------------------------------------------------------------
# Scratch directory isolation
# ---------------------------------------------------------------------------


class TestScratchDirectoryIsolation:
    """Different environment_ids receive independent scratch paths; reclaim frees them."""

    async def test_two_envs_get_different_scratch_paths(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = ScratchTrackingDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        await rt.provision(
            ProvisionRequest("env-a", "test.scratch", "sess1"), make_options(audit_log)
        )
        await rt.provision(
            ProvisionRequest("env-b", "test.scratch", "sess1"), make_options(audit_log)
        )
        result_a = await rt.execute(
            ExecuteRequest("env-a", "test.scratch", "sess1", ("pwd",)),
            make_options(audit_log),
        )
        result_b = await rt.execute(
            ExecuteRequest("env-b", "test.scratch", "sess1", ("pwd",)),
            make_options(audit_log),
        )
        assert result_a.stdout != result_b.stdout

    async def test_same_env_id_reuses_scratch_path(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = ScratchTrackingDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        await rt.provision(
            ProvisionRequest("env-a", "test.scratch", "sess1"), make_options(audit_log)
        )
        result1 = await rt.execute(
            ExecuteRequest("env-a", "test.scratch", "sess1", ("pwd",)),
            make_options(audit_log),
        )
        result2 = await rt.execute(
            ExecuteRequest("env-a", "test.scratch", "sess1", ("pwd",)),
            make_options(audit_log),
        )
        assert result1.stdout == result2.stdout

    async def test_reclaim_releases_scratch_entry(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = ScratchTrackingDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        await rt.provision(
            ProvisionRequest("env-a", "test.scratch", "sess1"), make_options(audit_log)
        )
        await rt.reclaim(ReclaimRequest("env-a", "test.scratch", "sess1"), make_options(audit_log))
        assert "env-a" in driver.reclaimed
        assert "env-a" not in driver._scratch

    async def test_scratch_paths_are_non_empty_after_provision(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = ScratchTrackingDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        await rt.provision(
            ProvisionRequest("env-x", "test.scratch", "sess1"), make_options(audit_log)
        )
        result = await rt.execute(
            ExecuteRequest("env-x", "test.scratch", "sess1", ("pwd",)),
            make_options(audit_log),
        )
        assert result.stdout != ""


# ---------------------------------------------------------------------------
# Env-var scoping
# ---------------------------------------------------------------------------


class TestEnvVarScoping:
    """Env vars from ExecuteRequest are forwarded verbatim; no bleed between calls."""

    async def test_env_dict_reaches_driver(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req = ExecuteRequest(
            environment_id="env1",
            environment_kind="test.stub",
            session_id="sess1",
            command=("env",),
            env={"SECRET": "abc123", "PORT": "8080"},
        )
        await rt.execute(req, make_options(audit_log))
        assert driver.executions[0].env == {"SECRET": "abc123", "PORT": "8080"}

    async def test_two_executions_have_independent_env_dicts(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req1 = ExecuteRequest("env1", "test.stub", "sess1", ("env",), env={"ONLY_IN_1": "yes"})
        req2 = ExecuteRequest("env1", "test.stub", "sess1", ("env",), env={"ONLY_IN_2": "yes"})
        await rt.execute(req1, make_options(audit_log))
        await rt.execute(req2, make_options(audit_log))
        assert "ONLY_IN_2" not in driver.executions[0].env
        assert "ONLY_IN_1" not in driver.executions[1].env

    async def test_empty_env_dict_is_valid(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req = ExecuteRequest("env1", "test.stub", "sess1", ("env",), env={})
        await rt.execute(req, make_options(audit_log))
        assert driver.executions[0].env == {}

    async def test_default_env_is_empty_dict(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req = ExecuteRequest("env1", "test.stub", "sess1", ("env",))
        await rt.execute(req, make_options(audit_log))
        assert driver.executions[0].env == {}

    async def test_env_dict_values_are_strings(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = StubDriver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        req = ExecuteRequest("env1", "test.stub", "sess1", ("env",), env={"K1": "v1", "K2": "v2"})
        await rt.execute(req, make_options(audit_log))
        for k, v in driver.executions[0].env.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# Network policy honored
# ---------------------------------------------------------------------------


class TestNetworkPolicyHonored:
    """NetworkEnforcer enforces the policy advertised by the driver."""

    def test_allowed_host_passes_enforcer(self) -> None:
        policy = NetworkPolicy(egress_allowed=True, allowed_hosts=("api.example.com",))
        enforcer = NetworkEnforcer(policy)
        assert enforcer.is_allowed("api.example.com") is True

    def test_unlisted_host_blocked_when_egress_denied(self) -> None:
        policy = NetworkPolicy(egress_allowed=False, allowed_hosts=("safe.internal",))
        enforcer = NetworkEnforcer(policy)
        assert enforcer.is_allowed("external.com") is False

    def test_listed_host_allowed_when_egress_denied(self) -> None:
        policy = NetworkPolicy(egress_allowed=False, allowed_hosts=("safe.internal",))
        enforcer = NetworkEnforcer(policy)
        assert enforcer.is_allowed("safe.internal") is True

    def test_blocked_host_denied_regardless_of_egress_flag(self) -> None:
        policy = NetworkPolicy(egress_allowed=True, blocked_hosts=("evil.example.com",))
        enforcer = NetworkEnforcer(policy)
        assert enforcer.is_allowed("evil.example.com") is False

    def test_assert_allowed_raises_network_violation(self) -> None:
        policy = NetworkPolicy(egress_allowed=False)
        enforcer = NetworkEnforcer(policy)
        with pytest.raises(NetworkViolation) as exc_info:
            enforcer.assert_allowed("evil.com", environment_id="env1", session_id="sess1")
        assert exc_info.value.host == "evil.com"
        assert exc_info.value.environment_id == "env1"
        assert exc_info.value.session_id == "sess1"

    def test_driver_network_policy_usable_with_enforcer(self) -> None:
        rt = registered_runtime()
        policy = rt.network_policy("test.stub")
        enforcer = NetworkEnforcer(policy)
        assert enforcer.is_allowed("example.com") is True

    async def test_network_violation_raised_by_driver_wraps_as_execute_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        violation = NetworkViolation(
            host="evil.com",
            agent_id="",
            environment_id="env1",
            session_id="sess1",
            timestamp=datetime.now(UTC).isoformat(),
        )
        driver = StubDriver(execute_raises=violation)
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(make_execute(), make_options(audit_log))
        assert exc_info.value.code == "ENV_EXECUTE_FAILED"

    async def test_network_violation_writes_audit_entry(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        violation = NetworkViolation(
            host="evil.com",
            agent_id="",
            environment_id="env1",
            session_id="sess1",
            timestamp=datetime.now(UTC).isoformat(),
        )
        driver = StubDriver(execute_raises=violation)
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute(), make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].event == "environment.execute.failed"

    async def test_network_violation_marks_span_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        violation = NetworkViolation(
            host="evil.com",
            agent_id="",
            environment_id="env1",
            session_id="sess1",
            timestamp=datetime.now(UTC).isoformat(),
        )
        driver = StubDriver(execute_raises=violation)
        rt = EnvironmentRuntime()
        rt.register(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.execute(make_execute(), make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Backend conformance (parameterized)
# ---------------------------------------------------------------------------


def _make_rt(driver: EnvironmentDriver) -> EnvironmentRuntime:
    rt = EnvironmentRuntime()
    rt.register(driver)
    return rt


def _provision_req(kind: str) -> ProvisionRequest:
    return ProvisionRequest(
        environment_id="env-conform", environment_kind=kind, session_id="sess-conform"
    )


def _execute_req(kind: str) -> ExecuteRequest:
    return ExecuteRequest(
        environment_id="env-conform",
        environment_kind=kind,
        session_id="sess-conform",
        command=("echo", "conform"),
    )


def _reclaim_req(kind: str) -> ReclaimRequest:
    return ReclaimRequest(
        environment_id="env-conform", environment_kind=kind, session_id="sess-conform"
    )


@pytest.mark.parametrize(
    "driver_cls",
    [StubDriver, ScratchTrackingDriver],
    ids=["stub", "scratch"],
)
class TestBackendConformance:
    """Core lifecycle, OTel, and audit contracts that every EnvironmentDriver must satisfy."""

    async def test_provision_execute_reclaim_lifecycle(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = driver_cls()
        rt = _make_rt(driver)
        await rt.provision(_provision_req(driver.kind), make_options(audit_log))
        result = await rt.execute(_execute_req(driver.kind), make_options(audit_log))
        assert isinstance(result, ExecuteResult)
        await rt.reclaim(_reclaim_req(driver.kind), make_options(audit_log))

    async def test_provision_emits_span(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = _make_rt(driver_cls())
        await rt.provision(_provision_req(driver_cls.kind), make_options(audit_log))
        assert mock_span.name == "environment.provision"

    async def test_execute_emits_span(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = driver_cls()
        rt = _make_rt(driver)
        await rt.provision(_provision_req(driver.kind), make_options(audit_log))
        await rt.execute(_execute_req(driver.kind), make_options(audit_log))
        assert mock_span.name == "environment.execute"

    async def test_reclaim_emits_span(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = driver_cls()
        rt = _make_rt(driver)
        await rt.provision(_provision_req(driver.kind), make_options(audit_log))
        await rt.reclaim(_reclaim_req(driver.kind), make_options(audit_log))
        assert mock_span.name == "environment.reclaim"

    async def test_provision_attaches_invocation_event(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = _make_rt(driver_cls())
        await rt.provision(_provision_req(driver_cls.kind), make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "environment.invocation" in event_names

    async def test_execute_attaches_invocation_event(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = driver_cls()
        rt = _make_rt(driver)
        await rt.provision(_provision_req(driver.kind), make_options(audit_log))
        await rt.execute(_execute_req(driver.kind), make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "environment.invocation" in event_names

    async def test_reclaim_attaches_invocation_event(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = driver_cls()
        rt = _make_rt(driver)
        await rt.provision(_provision_req(driver.kind), make_options(audit_log))
        await rt.reclaim(_reclaim_req(driver.kind), make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "environment.invocation" in event_names

    async def test_no_audit_entries_on_success(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = driver_cls()
        rt = _make_rt(driver)
        await rt.provision(_provision_req(driver.kind), make_options(audit_log))
        await rt.execute(_execute_req(driver.kind), make_options(audit_log))
        await rt.reclaim(_reclaim_req(driver.kind), make_options(audit_log))
        assert audit_log.entries == []

    async def test_driver_exception_wraps_with_failure_code(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[EnvironmentFailure] = []
        driver = driver_cls()
        driver._execute_raises = RuntimeError("backend error")  # type: ignore[attr-defined]
        rt = _make_rt(driver)
        with pytest.raises(EnvironmentFailure) as exc_info:
            await rt.execute(_execute_req(driver.kind), make_options(audit_log, errors))
        assert exc_info.value.code == "ENV_EXECUTE_FAILED"
        assert len(audit_log.entries) == 1
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_on_error_callback_called_on_failure(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[EnvironmentFailure] = []
        driver = driver_cls()
        driver._execute_raises = RuntimeError("injected failure")  # type: ignore[attr-defined]
        rt = _make_rt(driver)
        with pytest.raises(EnvironmentFailure):
            await rt.execute(_execute_req(driver.kind), make_options(audit_log, errors))
        assert len(errors) == 1

    async def test_capability_envelope_has_required_fields(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = _make_rt(driver_cls())
        caps = rt.capability_envelope(driver_cls.kind)
        assert isinstance(caps.cpu_millicores, int)
        assert isinstance(caps.memory_mb, int)
        assert isinstance(caps.timeout_seconds, int)
        assert isinstance(caps.can_write_filesystem, bool)
        assert isinstance(caps.network, NetworkPolicy)
        assert isinstance(caps.filesystem, FilesystemPolicy)

    async def test_network_policy_has_required_fields(
        self, driver_cls: type, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = _make_rt(driver_cls())
        policy = rt.network_policy(driver_cls.kind)
        assert isinstance(policy.egress_allowed, bool)
        assert isinstance(policy.allowed_hosts, tuple)
        assert isinstance(policy.blocked_hosts, tuple)
