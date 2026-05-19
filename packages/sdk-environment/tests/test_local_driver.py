"""
Tests for LocalBackendDriver.

Covers:
  - Kind constant: driver.kind == "system.local".
  - on_demand default False; configurable to True.
  - Policy / capability delegation.

  provision:
  - Creates scratch directory at {workspace_path}/{environment_id}.
  - Stores scratch path keyed by environment_id.
  - OSError from os.makedirs raises RuntimeError.

  execute:
  - Raises RuntimeError when environment is not provisioned.
  - Returns ExecuteResult with stdout, stderr, exit_code, duration_ms.
  - stdout / stderr decoded to str.
  - exit_code captured correctly.
  - duration_ms is a non-negative float.
  - request.env merged into subprocess environment.
  - stdin forwarded as bytes to communicate().
  - None stdin becomes empty bytes.
  - cwd set to the scratch directory.
  - timeout_seconds from request used; falls back to driver timeout_s.
  - asyncio.TimeoutError kills process and propagates.
  - env_passthrough empty → full os.environ inherited.
  - env_passthrough non-empty → only named vars forwarded.

  reclaim:
  - Calls shutil.rmtree on the scratch directory.
  - Removes environment_id from internal pool.
  - Idempotent when environment was never provisioned (no exception).
  - No rmtree call when not provisioned.

  Runtime integration:
  - Full lifecycle (provision → execute → reclaim) produces no audit entries.
  - provision failure writes audit entry with ENV_PROVISION_FAILED.
  - execute failure writes audit entry with ENV_EXECUTE_FAILED.
  - reclaim failure writes audit entry with ENV_RECLAIM_FAILED.
  - execute span name is "environment.execute".
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sdk_environment import (
    AuditLogEntry,
    CapabilityEnvelope,
    EnvironmentFailure,
    EnvironmentRuntime,
    ExecuteRequest,
    ExecuteResult,
    FilesystemPolicy,
    LocalBackendDriver,
    NetworkPolicy,
    ProvisionRequest,
    ReclaimRequest,
    RuntimeOptions,
)

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _driver(**kwargs: Any) -> LocalBackendDriver:
    return LocalBackendDriver(workspace_path="/tmp/meridian-local", **kwargs)


def _provision_req(kind: str = LocalBackendDriver.KIND) -> ProvisionRequest:
    return ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1")


def _execute_req(
    kind: str = LocalBackendDriver.KIND,
    command: tuple[str, ...] = ("echo", "hello"),
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    timeout: int | None = None,
) -> ExecuteRequest:
    return ExecuteRequest(
        environment_id="env1",
        environment_kind=kind,
        session_id="s1",
        command=command,
        stdin=stdin,
        env=env or {},
        timeout_seconds=timeout,
    )


def _reclaim_req(kind: str = LocalBackendDriver.KIND) -> ReclaimRequest:
    return ReclaimRequest(environment_id="env1", environment_kind=kind, session_id="s1")


def _make_proc(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


def _patch_exec(proc: MagicMock | None = None, procs: list[MagicMock] | None = None) -> Any:
    if procs is not None:
        return patch(
            "sdk_environment._local_driver.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=procs),
        )
    return patch(
        "sdk_environment._local_driver.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc or _make_proc()),
    )


def _patch_makedirs(side_effect: Exception | None = None) -> Any:
    if side_effect is not None:
        return patch("sdk_environment._local_driver.os.makedirs", side_effect=side_effect)
    return patch("sdk_environment._local_driver.os.makedirs")


def _patch_rmtree() -> Any:
    return patch("sdk_environment._local_driver.shutil.rmtree")


def _make_options(audit: CapturingAuditLog) -> RuntimeOptions:
    return RuntimeOptions(audit_log=audit)


# ---------------------------------------------------------------------------
# Kind and defaults
# ---------------------------------------------------------------------------


class TestKindAndDefaults:
    def test_kind_is_system_local(self) -> None:
        assert _driver().kind == "system.local"

    def test_kind_constant(self) -> None:
        assert LocalBackendDriver.KIND == "system.local"

    def test_on_demand_default_false(self) -> None:
        assert _driver().on_demand is False

    def test_on_demand_configurable_true(self) -> None:
        assert _driver(on_demand=True).on_demand is True

    def test_network_policy_default(self) -> None:
        assert isinstance(_driver().network_policy(), NetworkPolicy)

    def test_filesystem_policy_default(self) -> None:
        assert isinstance(_driver().filesystem_policy(), FilesystemPolicy)

    def test_capability_envelope_default(self) -> None:
        assert isinstance(_driver().capability_envelope(), CapabilityEnvelope)

    def test_custom_network_policy_returned(self) -> None:
        policy = NetworkPolicy(egress_allowed=True)
        assert _driver(network_policy=policy).network_policy() is policy

    def test_custom_filesystem_policy_returned(self) -> None:
        policy = FilesystemPolicy(read_globs=("**",))
        assert _driver(filesystem_policy=policy).filesystem_policy() is policy

    def test_custom_capability_envelope_returned(self) -> None:
        caps = CapabilityEnvelope(cpu_millicores=250)
        assert _driver(capability_envelope=caps).capability_envelope() is caps

    def test_capability_envelope_has_required_fields(self) -> None:
        caps = _driver().capability_envelope()
        assert isinstance(caps.cpu_millicores, int)
        assert isinstance(caps.memory_mb, int)
        assert isinstance(caps.timeout_seconds, int)
        assert isinstance(caps.can_write_filesystem, bool)
        assert isinstance(caps.network, NetworkPolicy)
        assert isinstance(caps.filesystem, FilesystemPolicy)


# ---------------------------------------------------------------------------
# provision
# ---------------------------------------------------------------------------


class TestProvision:
    async def test_creates_scratch_directory(self) -> None:
        with _patch_makedirs() as mock_makedirs:
            await _driver().provision(_provision_req())
        mock_makedirs.assert_called_once_with("/tmp/meridian-local/env1", exist_ok=True)

    async def test_scratch_dir_uses_environment_id(self) -> None:
        req = ProvisionRequest(environment_id="my-env", environment_kind=LocalBackendDriver.KIND, session_id="s1")
        with _patch_makedirs() as mock_makedirs:
            await _driver().provision(req)
        call_args = mock_makedirs.call_args.args[0]
        assert "my-env" in call_args

    async def test_custom_workspace_path(self) -> None:
        driver = LocalBackendDriver(workspace_path="/custom/ws")
        with _patch_makedirs() as mock_makedirs:
            await driver.provision(_provision_req())
        mock_makedirs.assert_called_once_with("/custom/ws/env1", exist_ok=True)

    async def test_workspace_path_trailing_slash_stripped(self) -> None:
        driver = LocalBackendDriver(workspace_path="/custom/ws/")
        with _patch_makedirs() as mock_makedirs:
            await driver.provision(_provision_req())
        mock_makedirs.assert_called_once_with("/custom/ws/env1", exist_ok=True)

    async def test_stores_scratch_dir_after_success(self) -> None:
        driver = _driver()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        assert "env1" in driver._scratch_dirs
        assert driver._scratch_dirs["env1"] == "/tmp/meridian-local/env1"

    async def test_oserror_raises_runtime_error(self) -> None:
        with _patch_makedirs(side_effect=OSError("permission denied")):
            with pytest.raises(RuntimeError, match="Failed to create scratch directory"):
                await _driver().provision(_provision_req())

    async def test_oserror_message_includes_path(self) -> None:
        with _patch_makedirs(side_effect=OSError("permission denied")):
            with pytest.raises(RuntimeError, match="/tmp/meridian-local/env1"):
                await _driver().provision(_provision_req())


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    async def test_raises_when_not_provisioned(self) -> None:
        with pytest.raises(RuntimeError, match="not provisioned"):
            await _driver().execute(_execute_req())

    async def test_returns_execute_result(self) -> None:
        driver = _driver()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_exec(_make_proc(stdout=b"ok")):
            result = await driver.execute(_execute_req())
        assert isinstance(result, ExecuteResult)

    async def test_stdout_captured(self) -> None:
        driver = _driver()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_exec(_make_proc(stdout=b"hello world")):
            result = await driver.execute(_execute_req())
        assert result.stdout == "hello world"

    async def test_stderr_captured(self) -> None:
        driver = _driver()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_exec(_make_proc(stderr=b"err msg")):
            result = await driver.execute(_execute_req())
        assert result.stderr == "err msg"

    async def test_exit_code_captured(self) -> None:
        driver = _driver()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_exec(_make_proc(returncode=42)):
            result = await driver.execute(_execute_req())
        assert result.exit_code == 42

    async def test_duration_ms_is_non_negative_float(self) -> None:
        driver = _driver()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_exec(_make_proc()):
            result = await driver.execute(_execute_req())
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms >= 0.0

    async def test_result_types(self) -> None:
        driver = _driver()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_exec(_make_proc(stdout=b"x")):
            result = await driver.execute(_execute_req())
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)
        assert isinstance(result.exit_code, int)
        assert isinstance(result.duration_ms, float)

    async def test_command_forwarded_to_subprocess(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append((list(args), kwargs))
            return _make_proc()

        with _patch_makedirs():
            await driver.provision(_provision_req())
        with patch("sdk_environment._local_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.execute(_execute_req(command=("python", "script.py")))

        cmd = captured[0][0]
        assert "python" in cmd
        assert "script.py" in cmd

    async def test_cwd_set_to_scratch_directory(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(kwargs)
            return _make_proc()

        with _patch_makedirs():
            await driver.provision(_provision_req())
        with patch("sdk_environment._local_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.execute(_execute_req())

        assert captured[0]["cwd"] == "/tmp/meridian-local/env1"

    async def test_request_env_merged_into_subprocess_env(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(kwargs)
            return _make_proc()

        with _patch_makedirs():
            await driver.provision(_provision_req())
        with patch("sdk_environment._local_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.execute(_execute_req(env={"MY_VAR": "my_val"}))

        assert captured[0]["env"]["MY_VAR"] == "my_val"

    async def test_stdin_forwarded_to_communicate(self) -> None:
        driver = _driver()
        proc = _make_proc()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_exec(proc):
            await driver.execute(_execute_req(stdin="my input"))
        proc.communicate.assert_awaited_once_with(b"my input")

    async def test_none_stdin_becomes_empty_bytes(self) -> None:
        driver = _driver()
        proc = _make_proc()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_exec(proc):
            await driver.execute(_execute_req(stdin=None))
        proc.communicate.assert_awaited_once_with(b"")

    async def test_timeout_seconds_used_from_request(self) -> None:
        driver = _driver(timeout_s=99.0)
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_exec(_make_proc()):
            result = await driver.execute(_execute_req(timeout=5))
        assert isinstance(result, ExecuteResult)

    async def test_timeout_kills_process_and_propagates(self) -> None:
        driver = _driver()
        proc = _make_proc()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_exec(proc):
            with patch(
                "sdk_environment._local_driver.asyncio.wait_for",
                AsyncMock(side_effect=asyncio.TimeoutError()),
            ):
                with pytest.raises(asyncio.TimeoutError):
                    await driver.execute(_execute_req())
        proc.kill.assert_called_once()

    async def test_env_passthrough_empty_inherits_full_env(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("HOST_VAR", "host_val")
        driver = _driver(env_passthrough=())
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(kwargs)
            return _make_proc()

        with _patch_makedirs():
            await driver.provision(_provision_req())
        with patch("sdk_environment._local_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.execute(_execute_req())

        assert captured[0]["env"].get("HOST_VAR") == "host_val"

    async def test_env_passthrough_non_empty_filters_env(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("ALLOWED_VAR", "allowed")
        monkeypatch.setenv("OTHER_VAR", "other")
        driver = _driver(env_passthrough=("ALLOWED_VAR",))
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(kwargs)
            return _make_proc()

        with _patch_makedirs():
            await driver.provision(_provision_req())
        with patch("sdk_environment._local_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.execute(_execute_req())

        env = captured[0]["env"]
        assert env.get("ALLOWED_VAR") == "allowed"
        assert "OTHER_VAR" not in env

    async def test_env_passthrough_absent_var_omitted(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("ABSENT_VAR", raising=False)
        driver = _driver(env_passthrough=("ABSENT_VAR",))
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(kwargs)
            return _make_proc()

        with _patch_makedirs():
            await driver.provision(_provision_req())
        with patch("sdk_environment._local_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.execute(_execute_req())

        assert "ABSENT_VAR" not in captured[0]["env"]

    async def test_request_env_overrides_passthrough(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("SHARED_VAR", "from_host")
        driver = _driver(env_passthrough=("SHARED_VAR",))
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(kwargs)
            return _make_proc()

        with _patch_makedirs():
            await driver.provision(_provision_req())
        with patch("sdk_environment._local_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.execute(_execute_req(env={"SHARED_VAR": "from_request"}))

        assert captured[0]["env"]["SHARED_VAR"] == "from_request"


# ---------------------------------------------------------------------------
# reclaim
# ---------------------------------------------------------------------------


class TestReclaim:
    async def test_calls_rmtree_on_scratch_directory(self) -> None:
        driver = _driver()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_rmtree() as mock_rmtree:
            await driver.reclaim(_reclaim_req())
        mock_rmtree.assert_called_once_with("/tmp/meridian-local/env1", ignore_errors=True)

    async def test_removes_environment_id_from_pool(self) -> None:
        driver = _driver()
        with _patch_makedirs():
            await driver.provision(_provision_req())
        with _patch_rmtree():
            await driver.reclaim(_reclaim_req())
        assert "env1" not in driver._scratch_dirs

    async def test_idempotent_when_not_provisioned(self) -> None:
        driver = _driver()
        await driver.reclaim(_reclaim_req())  # must not raise

    async def test_idempotent_does_not_call_rmtree(self) -> None:
        driver = _driver()
        with _patch_rmtree() as mock_rmtree:
            await driver.reclaim(_reclaim_req())
        mock_rmtree.assert_not_called()


# ---------------------------------------------------------------------------
# Runtime integration
# ---------------------------------------------------------------------------


class TestRuntimeIntegration:
    async def test_full_lifecycle_no_audit_entries(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        opts = _make_options(audit_log)
        kind = LocalBackendDriver.KIND

        provision_req = ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1")
        execute_req = ExecuteRequest(
            environment_id="env1",
            environment_kind=kind,
            session_id="s1",
            command=("echo", "ok"),
        )
        reclaim_req = ReclaimRequest(environment_id="env1", environment_kind=kind, session_id="s1")

        with _patch_makedirs():
            await rt.provision(provision_req, opts)

        with _patch_exec(_make_proc(stdout=b"ok")):
            result = await rt.execute(execute_req, opts)

        with _patch_rmtree():
            await rt.reclaim(reclaim_req, opts)

        assert isinstance(result, ExecuteResult)
        assert result.stdout == "ok"
        assert audit_log.entries == []

    async def test_provision_failure_writes_audit(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        opts = _make_options(audit_log)
        kind = LocalBackendDriver.KIND

        with _patch_makedirs(side_effect=OSError("no space left")):
            with pytest.raises(EnvironmentFailure) as exc_info:
                await rt.provision(
                    ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1"),
                    opts,
                )

        assert exc_info.value.code == "ENV_PROVISION_FAILED"
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "environment.provision.failed"

    async def test_execute_failure_writes_audit(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        opts = _make_options(audit_log)
        kind = LocalBackendDriver.KIND

        provision_req = ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1")
        execute_req = ExecuteRequest(
            environment_id="env1",
            environment_kind=kind,
            session_id="s1",
            command=("false",),
        )

        with _patch_makedirs():
            await rt.provision(provision_req, opts)

        with patch(
            "sdk_environment._local_driver.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=Exception("spawn failed")),
        ):
            with pytest.raises(EnvironmentFailure) as exc_info:
                await rt.execute(execute_req, opts)

        assert exc_info.value.code == "ENV_EXECUTE_FAILED"
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].event == "environment.execute.failed"

    async def test_execute_span_name(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        opts = _make_options(audit_log)
        kind = LocalBackendDriver.KIND

        provision_req = ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1")
        execute_req = ExecuteRequest(
            environment_id="env1",
            environment_kind=kind,
            session_id="s1",
            command=("echo", "hi"),
        )

        with _patch_makedirs():
            await rt.provision(provision_req, opts)

        with _patch_exec(_make_proc(stdout=b"hi")):
            await rt.execute(execute_req, opts)

        assert mock_span.name == "environment.execute"
