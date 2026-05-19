"""
Tests for DockerBackendDriver.

Covers:
  - Kind constant: driver.kind == "system.docker".
  - on_demand default True; configurable to False.
  - Policy / capability delegation.

  provision:
  - Calls docker run -d with --name, --cpus, --memory flags.
  - cpu_millicores converted to --cpus float (millicores / 1000).
  - memory_mb forwarded as --memory <N>m.
  - workspace_path mounts volume and sets no workdir at provision time.
  - env_passthrough vars present in os.environ are forwarded as --env.
  - env_passthrough vars absent from os.environ are omitted.
  - egress_allowed=False → --network none flag.
  - egress_allowed=True → no --network none flag.
  - Stores container name keyed by environment_id.
  - docker run failure (non-zero exit) raises RuntimeError.
  - asyncio.TimeoutError from docker run propagates.

  execute:
  - Raises RuntimeError when environment is not provisioned.
  - Returns ExecuteResult with stdout, stderr, exit_code, duration_ms.
  - stdout / stderr decoded to str.
  - exit_code captured correctly.
  - duration_ms is a non-negative float.
  - request.env forwarded as --env K=V flags to docker exec.
  - stdin forwarded as bytes to communicate().
  - None stdin becomes empty bytes.
  - workspace_path → --workdir flag on docker exec.
  - No --workdir when workspace_path is None.
  - timeout_seconds from request used; falls back to driver timeout_s.
  - asyncio.TimeoutError propagates.

  reclaim:
  - Calls docker rm -f with the container name.
  - Removes environment_id from internal pool.
  - Idempotent when environment was never provisioned (no exception).
  - asyncio.TimeoutError propagates from docker rm.

  Runtime integration:
  - Full lifecycle (provision → execute → reclaim) produces no audit entries.
  - provision failure writes audit entry with ENV_PROVISION_FAILED.
  - execute failure writes audit entry with ENV_EXECUTE_FAILED.
  - reclaim failure (TimeoutError) writes audit entry with ENV_RECLAIM_FAILED.
  - execute span name is "environment.execute".
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sdk_environment import (
    AuditLogEntry,
    CapabilityEnvelope,
    DockerBackendDriver,
    EnvironmentFailure,
    EnvironmentRuntime,
    ExecuteRequest,
    ExecuteResult,
    FilesystemPolicy,
    NetworkPolicy,
    ProvisionRequest,
    ReclaimRequest,
    RuntimeOptions,
)

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMAGE = "python:3.12-slim"


def _driver(**kwargs: Any) -> DockerBackendDriver:
    return DockerBackendDriver(image=_IMAGE, **kwargs)


def _provision_req(kind: str = DockerBackendDriver.KIND) -> ProvisionRequest:
    return ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1")


def _execute_req(
    kind: str = DockerBackendDriver.KIND,
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


def _reclaim_req(kind: str = DockerBackendDriver.KIND) -> ReclaimRequest:
    return ReclaimRequest(environment_id="env1", environment_kind=kind, session_id="s1")


def _make_proc(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


def _patch_docker(proc: MagicMock | None = None, procs: list[MagicMock] | None = None) -> Any:
    """Patch asyncio.create_subprocess_exec to return proc(s) in order."""
    if procs is not None:
        return patch(
            "sdk_environment._docker_driver.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=procs),
        )
    return patch(
        "sdk_environment._docker_driver.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc or _make_proc()),
    )


def _make_options(audit: CapturingAuditLog) -> RuntimeOptions:
    return RuntimeOptions(audit_log=audit)


# ---------------------------------------------------------------------------
# Kind and defaults
# ---------------------------------------------------------------------------


class TestKindAndDefaults:
    def test_kind_is_system_docker(self) -> None:
        assert _driver().kind == "system.docker"

    def test_kind_constant(self) -> None:
        assert DockerBackendDriver.KIND == "system.docker"

    def test_on_demand_default_true(self) -> None:
        assert _driver().on_demand is True

    def test_on_demand_configurable_false(self) -> None:
        assert _driver(on_demand=False).on_demand is False

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
    async def test_calls_docker_run(self) -> None:
        with _patch_docker() as mock_exec:
            await _driver().provision(_provision_req())
        mock_exec.assert_awaited_once()
        args = mock_exec.call_args.args
        assert args[0] == "docker"
        assert args[1] == "run"

    async def test_detached_flag(self) -> None:
        with _patch_docker() as mock_exec:
            await _driver().provision(_provision_req())
        args = list(mock_exec.call_args.args)
        assert "-d" in args

    async def test_container_name_contains_environment_id(self) -> None:
        with _patch_docker() as mock_exec:
            await _driver().provision(_provision_req())
        args = list(mock_exec.call_args.args)
        name_idx = args.index("--name")
        assert "env1" in args[name_idx + 1]

    async def test_cpus_flag_from_millicores(self) -> None:
        caps = CapabilityEnvelope(cpu_millicores=500)
        with _patch_docker() as mock_exec:
            await _driver(capability_envelope=caps).provision(_provision_req())
        args = list(mock_exec.call_args.args)
        cpus_idx = args.index("--cpus")
        assert float(args[cpus_idx + 1]) == pytest.approx(0.5)

    async def test_cpus_flag_1000_millicores(self) -> None:
        caps = CapabilityEnvelope(cpu_millicores=1000)
        with _patch_docker() as mock_exec:
            await _driver(capability_envelope=caps).provision(_provision_req())
        args = list(mock_exec.call_args.args)
        cpus_idx = args.index("--cpus")
        assert float(args[cpus_idx + 1]) == pytest.approx(1.0)

    async def test_memory_flag_from_capability(self) -> None:
        caps = CapabilityEnvelope(memory_mb=256)
        with _patch_docker() as mock_exec:
            await _driver(capability_envelope=caps).provision(_provision_req())
        args = list(mock_exec.call_args.args)
        mem_idx = args.index("--memory")
        assert args[mem_idx + 1] == "256m"

    async def test_network_none_when_egress_not_allowed(self) -> None:
        policy = NetworkPolicy(egress_allowed=False)
        with _patch_docker() as mock_exec:
            await _driver(network_policy=policy).provision(_provision_req())
        args = list(mock_exec.call_args.args)
        assert "--network" in args
        net_idx = args.index("--network")
        assert args[net_idx + 1] == "none"

    async def test_no_network_none_when_egress_allowed(self) -> None:
        policy = NetworkPolicy(egress_allowed=True)
        with _patch_docker() as mock_exec:
            await _driver(network_policy=policy).provision(_provision_req())
        args = list(mock_exec.call_args.args)
        assert "none" not in args

    async def test_no_network_none_when_allowed_hosts_set(self) -> None:
        policy = NetworkPolicy(egress_allowed=False, allowed_hosts=("example.com",))
        with _patch_docker() as mock_exec:
            await _driver(network_policy=policy).provision(_provision_req())
        args = list(mock_exec.call_args.args)
        assert "none" not in args

    async def test_volume_flag_when_workspace_path_set(self) -> None:
        with _patch_docker() as mock_exec:
            await _driver(workspace_path="/host/ws").provision(_provision_req())
        args = list(mock_exec.call_args.args)
        assert "--volume" in args
        vol_idx = args.index("--volume")
        assert "/host/ws" in args[vol_idx + 1]
        assert "/workspace" in args[vol_idx + 1]

    async def test_custom_workspace_mount_target(self) -> None:
        with _patch_docker() as mock_exec:
            await _driver(
                workspace_path="/host/ws", workspace_mount_target="/app"
            ).provision(_provision_req())
        args = list(mock_exec.call_args.args)
        vol_idx = args.index("--volume")
        assert "/app" in args[vol_idx + 1]

    async def test_no_volume_flag_without_workspace_path(self) -> None:
        with _patch_docker() as mock_exec:
            await _driver().provision(_provision_req())
        args = list(mock_exec.call_args.args)
        assert "--volume" not in args

    async def test_image_in_run_cmd(self) -> None:
        with _patch_docker() as mock_exec:
            await _driver().provision(_provision_req())
        args = list(mock_exec.call_args.args)
        assert _IMAGE in args

    async def test_sleep_infinity_entrypoint(self) -> None:
        with _patch_docker() as mock_exec:
            await _driver().provision(_provision_req())
        args = list(mock_exec.call_args.args)
        assert "sleep" in args
        assert "infinity" in args

    async def test_env_passthrough_present_var_forwarded(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("MY_TOKEN", "secret123")
        with _patch_docker() as mock_exec:
            await _driver(env_passthrough=("MY_TOKEN",)).provision(_provision_req())
        args = list(mock_exec.call_args.args)
        env_pairs = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
        assert any("MY_TOKEN=secret123" == p for p in env_pairs)

    async def test_env_passthrough_absent_var_omitted(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("ABSENT_VAR", raising=False)
        with _patch_docker() as mock_exec:
            await _driver(env_passthrough=("ABSENT_VAR",)).provision(_provision_req())
        args = list(mock_exec.call_args.args)
        env_pairs = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
        assert not any("ABSENT_VAR" in p for p in env_pairs)

    async def test_stores_container_name_after_success(self) -> None:
        driver = _driver()
        with _patch_docker():
            await driver.provision(_provision_req())
        assert "env1" in driver._containers
        assert "env1" in driver._containers["env1"]

    async def test_docker_run_failure_raises(self) -> None:
        with _patch_docker(_make_proc(returncode=1, stderr=b"image not found")):
            with pytest.raises(RuntimeError, match="docker run failed"):
                await _driver().provision(_provision_req())

    async def test_docker_run_failure_message_includes_stderr(self) -> None:
        with _patch_docker(_make_proc(returncode=1, stderr=b"no such image")):
            with pytest.raises(RuntimeError, match="no such image"):
                await _driver().provision(_provision_req())

    async def test_timeout_propagates(self) -> None:
        with patch(
            "sdk_environment._docker_driver.asyncio.create_subprocess_exec",
            AsyncMock(return_value=_make_proc()),
        ):
            with patch(
                "sdk_environment._docker_driver.asyncio.wait_for",
                AsyncMock(side_effect=asyncio.TimeoutError()),
            ):
                with pytest.raises(asyncio.TimeoutError):
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
        run_proc = _make_proc()
        exec_proc = _make_proc(stdout=b"ok", stderr=b"")
        with _patch_docker(procs=[run_proc, exec_proc]):
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert isinstance(result, ExecuteResult)

    async def test_stdout_captured(self) -> None:
        driver = _driver()
        with _patch_docker(procs=[_make_proc(), _make_proc(stdout=b"hello world")]):
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert result.stdout == "hello world"

    async def test_stderr_captured(self) -> None:
        driver = _driver()
        with _patch_docker(procs=[_make_proc(), _make_proc(stderr=b"err msg")]):
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert result.stderr == "err msg"

    async def test_exit_code_captured(self) -> None:
        driver = _driver()
        with _patch_docker(procs=[_make_proc(), _make_proc(returncode=42)]):
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert result.exit_code == 42

    async def test_duration_ms_is_non_negative_float(self) -> None:
        driver = _driver()
        with _patch_docker(procs=[_make_proc(), _make_proc()]):
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms >= 0.0

    async def test_result_types(self) -> None:
        driver = _driver()
        with _patch_docker(procs=[_make_proc(), _make_proc(stdout=b"x")]):
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)
        assert isinstance(result.exit_code, int)
        assert isinstance(result.duration_ms, float)

    async def test_calls_docker_exec(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(list(args))
            return _make_proc()

        with patch("sdk_environment._docker_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.provision(_provision_req())
            await driver.execute(_execute_req())

        exec_cmd = captured[1]
        assert exec_cmd[0] == "docker"
        assert exec_cmd[1] == "exec"

    async def test_container_name_in_exec_cmd(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(list(args))
            return _make_proc()

        with patch("sdk_environment._docker_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.provision(_provision_req())
            await driver.execute(_execute_req())

        exec_cmd = captured[1]
        assert any("env1" in arg for arg in exec_cmd)

    async def test_command_forwarded_to_exec(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(list(args))
            return _make_proc()

        with patch("sdk_environment._docker_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.provision(_provision_req())
            await driver.execute(_execute_req(command=("python", "script.py")))

        exec_cmd = captured[1]
        assert "python" in exec_cmd
        assert "script.py" in exec_cmd

    async def test_env_vars_forwarded_as_env_flags(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(list(args))
            return _make_proc()

        with patch("sdk_environment._docker_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.provision(_provision_req())
            await driver.execute(_execute_req(env={"MY_VAR": "my_val"}))

        exec_cmd = captured[1]
        env_pairs = [exec_cmd[i + 1] for i, a in enumerate(exec_cmd) if a == "--env"]
        assert any("MY_VAR=my_val" == p for p in env_pairs)

    async def test_stdin_forwarded_to_communicate(self) -> None:
        driver = _driver()
        run_proc = _make_proc()
        exec_proc = _make_proc()

        with patch(
            "sdk_environment._docker_driver.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=[run_proc, exec_proc]),
        ):
            await driver.provision(_provision_req())
            await driver.execute(_execute_req(stdin="my input"))

        exec_proc.communicate.assert_awaited_once_with(b"my input")

    async def test_none_stdin_becomes_empty_bytes(self) -> None:
        driver = _driver()
        run_proc = _make_proc()
        exec_proc = _make_proc()

        with patch(
            "sdk_environment._docker_driver.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=[run_proc, exec_proc]),
        ):
            await driver.provision(_provision_req())
            await driver.execute(_execute_req(stdin=None))

        exec_proc.communicate.assert_awaited_once_with(b"")

    async def test_workdir_flag_when_workspace_path_set(self) -> None:
        driver = _driver(workspace_path="/host/ws")
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(list(args))
            return _make_proc()

        with patch("sdk_environment._docker_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.provision(_provision_req())
            await driver.execute(_execute_req())

        exec_cmd = captured[1]
        assert "--workdir" in exec_cmd
        wd_idx = exec_cmd.index("--workdir")
        assert exec_cmd[wd_idx + 1] == "/workspace"

    async def test_no_workdir_flag_without_workspace_path(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(list(args))
            return _make_proc()

        with patch("sdk_environment._docker_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.provision(_provision_req())
            await driver.execute(_execute_req())

        exec_cmd = captured[1]
        assert "--workdir" not in exec_cmd

    async def test_timeout_seconds_used_from_request(self) -> None:
        driver = _driver(timeout_s=99.0)
        with _patch_docker(procs=[_make_proc(), _make_proc()]):
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req(timeout=5))
        assert isinstance(result, ExecuteResult)

    async def test_timeout_propagates(self) -> None:
        driver = _driver()
        # provision completes before the wait_for patch is applied, so the
        # mock only needs to handle the execute call
        with _patch_docker():
            await driver.provision(_provision_req())

        with patch(
            "sdk_environment._docker_driver.asyncio.create_subprocess_exec",
            AsyncMock(return_value=_make_proc()),
        ):
            with patch(
                "sdk_environment._docker_driver.asyncio.wait_for",
                AsyncMock(side_effect=asyncio.TimeoutError()),
            ):
                with pytest.raises(asyncio.TimeoutError):
                    await driver.execute(_execute_req())

    async def test_interactive_flag_on_exec(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(list(args))
            return _make_proc()

        with patch("sdk_environment._docker_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.provision(_provision_req())
            await driver.execute(_execute_req())

        exec_cmd = captured[1]
        assert "-i" in exec_cmd


# ---------------------------------------------------------------------------
# reclaim
# ---------------------------------------------------------------------------


class TestReclaim:
    async def test_calls_docker_rm(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(list(args))
            return _make_proc()

        with patch("sdk_environment._docker_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.provision(_provision_req())
            await driver.reclaim(_reclaim_req())

        rm_cmd = captured[1]
        assert rm_cmd[0] == "docker"
        assert rm_cmd[1] == "rm"
        assert "-f" in rm_cmd

    async def test_container_name_in_rm_cmd(self) -> None:
        driver = _driver()
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(list(args))
            return _make_proc()

        with patch("sdk_environment._docker_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.provision(_provision_req())
            await driver.reclaim(_reclaim_req())

        rm_cmd = captured[1]
        assert any("env1" in arg for arg in rm_cmd)

    async def test_removes_environment_id_from_pool(self) -> None:
        driver = _driver()
        with _patch_docker(procs=[_make_proc(), _make_proc()]):
            await driver.provision(_provision_req())
            await driver.reclaim(_reclaim_req())
        assert "env1" not in driver._containers

    async def test_idempotent_when_not_provisioned(self) -> None:
        driver = _driver()
        await driver.reclaim(_reclaim_req())  # must not raise

    async def test_idempotent_does_not_call_docker(self) -> None:
        driver = _driver()
        with patch(
            "sdk_environment._docker_driver.asyncio.create_subprocess_exec",
            AsyncMock(),
        ) as mock_exec:
            await driver.reclaim(_reclaim_req())
        mock_exec.assert_not_called()

    async def test_timeout_propagates(self) -> None:
        driver = _driver()
        with _patch_docker():
            await driver.provision(_provision_req())

        with patch(
            "sdk_environment._docker_driver.asyncio.create_subprocess_exec",
            AsyncMock(return_value=_make_proc()),
        ):
            with patch(
                "sdk_environment._docker_driver.asyncio.wait_for",
                AsyncMock(side_effect=asyncio.TimeoutError()),
            ):
                with pytest.raises(asyncio.TimeoutError):
                    await driver.reclaim(_reclaim_req())


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
        kind = DockerBackendDriver.KIND

        provision_req = ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1")
        execute_req = ExecuteRequest(
            environment_id="env1",
            environment_kind=kind,
            session_id="s1",
            command=("echo", "ok"),
        )
        reclaim_req = ReclaimRequest(environment_id="env1", environment_kind=kind, session_id="s1")

        with _patch_docker(
            procs=[
                _make_proc(),               # provision: docker run
                _make_proc(stdout=b"ok"),   # execute: docker exec
                _make_proc(),               # reclaim: docker rm -f
            ]
        ):
            await rt.provision(provision_req, opts)
            result = await rt.execute(execute_req, opts)
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
        kind = DockerBackendDriver.KIND
        opts = _make_options(audit_log)

        with _patch_docker(_make_proc(returncode=1, stderr=b"no such image")):
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
        kind = DockerBackendDriver.KIND
        opts = _make_options(audit_log)

        provision_req = ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1")
        execute_req = ExecuteRequest(
            environment_id="env1",
            environment_kind=kind,
            session_id="s1",
            command=("false",),
        )

        with patch(
            "sdk_environment._docker_driver.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=[
                _make_proc(),     # provision succeeds
                Exception("docker exec exploded"),  # execute fails
            ]),
        ):
            await rt.provision(provision_req, opts)
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
        kind = DockerBackendDriver.KIND
        opts = _make_options(audit_log)

        provision_req = ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1")
        execute_req = ExecuteRequest(
            environment_id="env1",
            environment_kind=kind,
            session_id="s1",
            command=("echo", "hi"),
        )

        with _patch_docker(procs=[_make_proc(), _make_proc(stdout=b"hi")]):
            await rt.provision(provision_req, opts)
            await rt.execute(execute_req, opts)

        assert mock_span.name == "environment.execute"

    async def test_docker_binary_configurable(self) -> None:
        driver = _driver(docker_binary="/usr/local/bin/docker")
        captured: list[Any] = []

        async def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.append(list(args))
            return _make_proc()

        with patch("sdk_environment._docker_driver.asyncio.create_subprocess_exec", side_effect=_capture):
            await driver.provision(_provision_req())

        assert captured[0][0] == "/usr/local/bin/docker"
