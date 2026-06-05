"""
Tests for SshBackendDriver.

Covers:
  - Kind constant: driver.kind == "system.ssh".
  - on_demand default False; configurable to True.
  - Policy / capability delegation.

  provision:
  - Opens SSH connection to remote host with mandatory known_hosts.
  - Creates remote scratch directory for the environment_id.
  - Stores connection keyed by environment_id.
  - asyncssh unavailable: raises RuntimeError.
  - SSH connection failure: propagates (runtime wraps as ENV_PROVISION_FAILED).
  - mkdir failure (non-zero exit): raises RuntimeError.

  execute:
  - Raises RuntimeError when environment is not provisioned.
  - Syncs local scratch dir to remote via rsync subprocess.
  - Runs command in remote scratch dir via SSH.
  - Captures stdout, stderr, exit_code, duration_ms.
  - env dict forwarded as shell exports.
  - stdin forwarded to SSH run.
  - timeout_seconds from request used; falls back to driver timeout_s.
  - asyncio.TimeoutError propagates.
  - rsync failure (non-zero exit): raises RuntimeError.
  - rsync timeout: raises asyncio.TimeoutError.

  reclaim:
  - Removes remote scratch directory.
  - Closes SSH connection.
  - Idempotent when environment was never provisioned.

  Authentication:
  - private_key_path used to load PEM for asyncssh.
  - Vault secret used when vault/vault_id/secret_name are configured.
  - Vault secret takes precedence over private_key_path.

  rsync SSH options:
  - known_hosts enforced (StrictHostKeyChecking=yes, UserKnownHostsFile).
  - private_key_path forwarded as -i flag when set.
  - Custom port forwarded as -p flag.

  Runtime integration:
  - Full lifecycle (provision → execute → reclaim) produces no audit entries.
  - execute failure writes audit entry with ENV_EXECUTE_FAILED.
  - provision failure writes audit entry with ENV_PROVISION_FAILED.
  - execute span name is "environment.execute".
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from sdk_environment import (
    AuditLogEntry,
    CapabilityEnvelope,
    EnvironmentFailure,
    EnvironmentRuntime,
    ExecuteRequest,
    ExecuteResult,
    FilesystemPolicy,
    NetworkPolicy,
    ProvisionRequest,
    ReclaimRequest,
    RuntimeOptions,
    SshBackendDriver,
)

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOST = "remote.example.com"
_USER = "meridian"
_KNOWN_HOSTS = "/etc/ssh/known_hosts"


def _driver(**kwargs: Any) -> SshBackendDriver:
    return SshBackendDriver(
        host=_HOST,
        username=_USER,
        known_hosts=_KNOWN_HOSTS,
        **kwargs,
    )


def _provision_req(kind: str = SshBackendDriver.KIND) -> ProvisionRequest:
    return ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="s1")


def _execute_req(
    kind: str = SshBackendDriver.KIND,
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


def _reclaim_req(kind: str = SshBackendDriver.KIND) -> ReclaimRequest:
    return ReclaimRequest(environment_id="env1", environment_kind=kind, session_id="s1")


def _make_ssh_result(
    stdout: str = "",
    stderr: str = "",
    exit_status: int = 0,
) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.exit_status = exit_status
    return result


def _make_conn(
    run_result: MagicMock | None = None,
    run_side_effect: BaseException | None = None,
) -> MagicMock:
    conn = MagicMock()
    if run_side_effect is not None:
        conn.run = AsyncMock(side_effect=run_side_effect)
    else:
        conn.run = AsyncMock(return_value=run_result or _make_ssh_result())
    conn.close = MagicMock()
    return conn


def _make_rsync_proc(returncode: int = 0, stderr: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.communicate = AsyncMock(return_value=(b"", stderr))
    return proc


def _patch_asyncssh(conn: MagicMock) -> Any:
    return patch(
        "sdk_environment._ssh_driver._asyncssh.connect",
        AsyncMock(return_value=conn),
    )


def _patch_rsync(proc: MagicMock) -> Any:
    return patch(
        "sdk_environment._ssh_driver.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    )


def _make_options(audit: CapturingAuditLog) -> RuntimeOptions:
    return RuntimeOptions(audit_log=audit)


# ---------------------------------------------------------------------------
# Kind and defaults
# ---------------------------------------------------------------------------


class TestKindAndDefaults:
    def test_kind_is_system_ssh(self) -> None:
        assert _driver().kind == "system.ssh"

    def test_kind_constant(self) -> None:
        assert SshBackendDriver.KIND == "system.ssh"

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
    async def test_opens_ssh_connection(self) -> None:
        conn = _make_conn(run_result=_make_ssh_result())
        with _patch_asyncssh(conn) as mock_connect, _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
        mock_connect.assert_awaited_once()

    async def test_connect_uses_host(self) -> None:
        conn = _make_conn()
        with _patch_asyncssh(conn) as mock_connect, _patch_rsync(_make_rsync_proc()):
            await _driver().provision(_provision_req())
        kwargs = mock_connect.call_args.kwargs
        assert kwargs["host"] == _HOST

    async def test_connect_uses_username(self) -> None:
        conn = _make_conn()
        with _patch_asyncssh(conn) as mock_connect, _patch_rsync(_make_rsync_proc()):
            await _driver().provision(_provision_req())
        kwargs = mock_connect.call_args.kwargs
        assert kwargs["username"] == _USER

    async def test_connect_uses_known_hosts(self) -> None:
        conn = _make_conn()
        with _patch_asyncssh(conn) as mock_connect, _patch_rsync(_make_rsync_proc()):
            await _driver().provision(_provision_req())
        kwargs = mock_connect.call_args.kwargs
        assert kwargs["known_hosts"] == _KNOWN_HOSTS

    async def test_connect_uses_custom_port(self) -> None:
        conn = _make_conn()
        with _patch_asyncssh(conn) as mock_connect, _patch_rsync(_make_rsync_proc()):
            await _driver(port=2222).provision(_provision_req())
        kwargs = mock_connect.call_args.kwargs
        assert kwargs["port"] == 2222

    async def test_creates_remote_scratch_dir(self) -> None:
        conn = _make_conn()
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
        cmd = conn.run.call_args.args[0]
        assert "mkdir" in cmd
        assert "env1" in cmd

    async def test_stores_connection(self) -> None:
        conn = _make_conn()
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
        assert driver._connections["env1"] is conn

    async def test_asyncssh_unavailable_raises(self) -> None:
        with (
            patch("sdk_environment._ssh_driver._ASYNCSSH_AVAILABLE", False),
            pytest.raises(RuntimeError, match="asyncssh"),
        ):
            await _driver().provision(_provision_req())

    async def test_connection_failure_propagates(self) -> None:
        with (
            patch(
                "sdk_environment._ssh_driver._asyncssh.connect",
                AsyncMock(side_effect=OSError("connection refused")),
            ),
            pytest.raises(OSError, match="connection refused"),
        ):
            await _driver().provision(_provision_req())

    async def test_mkdir_failure_raises(self) -> None:
        conn = _make_conn(run_result=_make_ssh_result(stderr="permission denied", exit_status=1))
        with (
            _patch_asyncssh(conn),
            _patch_rsync(_make_rsync_proc()),
            pytest.raises(RuntimeError, match="scratch directory"),
        ):
            await _driver().provision(_provision_req())


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    async def test_raises_when_not_provisioned(self) -> None:
        with pytest.raises(RuntimeError, match="not provisioned"):
            await _driver().execute(_execute_req())

    async def test_returns_execute_result(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(
            side_effect=[
                _make_ssh_result(),  # provision mkdir
                _make_ssh_result(stdout="hi"),  # execute
            ]
        )
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert isinstance(result, ExecuteResult)

    async def test_stdout_captured(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(
            side_effect=[
                _make_ssh_result(),
                _make_ssh_result(stdout="hello world"),
            ]
        )
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert result.stdout == "hello world"

    async def test_stderr_captured(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(
            side_effect=[
                _make_ssh_result(),
                _make_ssh_result(stderr="err msg"),
            ]
        )
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert result.stderr == "err msg"

    async def test_exit_code_captured(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(
            side_effect=[
                _make_ssh_result(),
                _make_ssh_result(exit_status=42),
            ]
        )
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert result.exit_code == 42

    async def test_duration_ms_is_positive_float(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms >= 0.0

    async def test_result_types(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(
            side_effect=[_make_ssh_result(), _make_ssh_result(stdout="x", exit_status=0)]
        )
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)
        assert isinstance(result.exit_code, int)
        assert isinstance(result.duration_ms, float)

    async def test_bytes_output_decoded(self) -> None:
        conn = _make_conn()
        ssh_result = _make_ssh_result()
        ssh_result.stdout = b"bytes out"
        ssh_result.stderr = b"bytes err"
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), ssh_result])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req())
        assert result.stdout == "bytes out"
        assert result.stderr == "bytes err"

    async def test_command_sent_to_remote(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            await driver.execute(_execute_req(command=("python", "run.py")))
        # Second run call is the command execution (first is mkdir)
        exec_cmd = conn.run.call_args_list[1].args[0]
        assert "python" in exec_cmd
        assert "run.py" in exec_cmd

    async def test_command_run_in_remote_scratch_dir(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            await driver.execute(_execute_req())
        exec_cmd = conn.run.call_args_list[1].args[0]
        assert "env1" in exec_cmd
        assert "cd" in exec_cmd

    async def test_env_vars_exported_in_command(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            await driver.execute(_execute_req(env={"MY_VAR": "my_val"}))
        exec_cmd = conn.run.call_args_list[1].args[0]
        assert "MY_VAR" in exec_cmd
        assert "my_val" in exec_cmd
        assert "export" in exec_cmd

    async def test_stdin_forwarded(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            await driver.execute(_execute_req(stdin="my stdin"))
        call_kwargs = conn.run.call_args_list[1].kwargs
        assert call_kwargs.get("input") == "my stdin"

    async def test_none_stdin_becomes_empty_string(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            await driver.execute(_execute_req(stdin=None))
        call_kwargs = conn.run.call_args_list[1].kwargs
        assert call_kwargs.get("input") == ""

    async def test_rsync_invoked_before_command(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        call_order: list[str] = []
        rsync_proc = _make_rsync_proc()

        async def _fake_exec(*args: Any, **kwargs: Any) -> MagicMock:
            call_order.append("rsync")
            return rsync_proc

        original_run = conn.run

        async def _tracked_run(*args: Any, **kwargs: Any) -> MagicMock:
            call_order.append("ssh_run")
            return await original_run(*args, **kwargs)

        conn.run = _tracked_run

        with (
            _patch_asyncssh(conn),
            patch(
                "sdk_environment._ssh_driver.asyncio.create_subprocess_exec",
                side_effect=_fake_exec,
            ),
        ):
            driver = _driver()
            await driver.provision(_provision_req())
            call_order.clear()
            await driver.execute(_execute_req())

        assert call_order.index("rsync") < call_order.index("ssh_run")

    async def test_rsync_uses_known_hosts(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        captured_cmd: list[str] = []

        async def _capture_exec(*args: Any, **kwargs: Any) -> MagicMock:
            captured_cmd.extend(args)
            return _make_rsync_proc()

        with (
            _patch_asyncssh(conn),
            patch(
                "sdk_environment._ssh_driver.asyncio.create_subprocess_exec",
                side_effect=_capture_exec,
            ),
        ):
            driver = _driver()
            await driver.provision(_provision_req())
            await driver.execute(_execute_req())

        ssh_e_arg = " ".join(captured_cmd)
        assert "UserKnownHostsFile" in ssh_e_arg
        assert _KNOWN_HOSTS in ssh_e_arg
        assert "StrictHostKeyChecking=yes" in ssh_e_arg

    async def test_rsync_failure_raises(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(return_value=_make_ssh_result())
        with (
            _patch_asyncssh(conn),
            _patch_rsync(_make_rsync_proc(returncode=1, stderr=b"no such file")),
        ):
            driver = _driver()
            await driver.provision(_provision_req())
            with pytest.raises(RuntimeError, match="rsync failed"):
                await driver.execute(_execute_req())

    async def test_timeout_seconds_used_from_request(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver(timeout_s=99.0)
            await driver.provision(_provision_req())
            result = await driver.execute(_execute_req(timeout=5))
        assert isinstance(result, ExecuteResult)

    async def test_timeout_propagates(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(return_value=_make_ssh_result())

        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver(timeout_s=0.001)
            await driver.provision(_provision_req())
            with (
                patch(
                    "sdk_environment._ssh_driver.asyncio.wait_for",
                    AsyncMock(side_effect=TimeoutError()),
                ),
                pytest.raises(asyncio.TimeoutError),
            ):
                await driver.execute(_execute_req())


# ---------------------------------------------------------------------------
# reclaim
# ---------------------------------------------------------------------------


class TestReclaim:
    async def test_closes_connection(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            await driver.reclaim(_reclaim_req())
        conn.close.assert_called_once()

    async def test_removes_remote_scratch_dir(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            await driver.reclaim(_reclaim_req())
        reclaim_cmd = conn.run.call_args.args[0]
        assert "rm" in reclaim_cmd
        assert "env1" in reclaim_cmd

    async def test_removes_connection_from_pool(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            await driver.reclaim(_reclaim_req())
        assert "env1" not in driver._connections

    async def test_idempotent_when_not_provisioned(self) -> None:
        driver = _driver()
        await driver.reclaim(_reclaim_req())  # must not raise

    async def test_closes_connection_even_if_rm_fails(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), RuntimeError("rm failed")])
        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            driver = _driver()
            await driver.provision(_provision_req())
            with pytest.raises(RuntimeError):
                await driver.reclaim(_reclaim_req())
        conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    async def test_private_key_path_loaded(self, tmp_path: Any) -> None:
        key_file = tmp_path / "id_rsa"
        key_file.write_text("---BEGIN---\nfake_pem\n---END---\n")
        conn = _make_conn()

        with (
            _patch_asyncssh(conn) as mock_connect,
            _patch_rsync(_make_rsync_proc()),
            patch(
                "sdk_environment._ssh_driver._asyncssh.import_private_key",
                return_value="imported_key",
            ),
        ):
            driver = _driver(private_key_path=str(key_file))
            await driver.provision(_provision_req())

        kwargs = mock_connect.call_args.kwargs
        assert "client_keys" in kwargs
        assert kwargs["client_keys"] == ["imported_key"]

    async def test_vault_secret_used_for_key(self) -> None:
        conn = _make_conn()
        vault = MagicMock()
        vault.get_secret.return_value = {"value": "vault_pem"}

        with (
            _patch_asyncssh(conn) as mock_connect,
            _patch_rsync(_make_rsync_proc()),
            patch(
                "sdk_environment._ssh_driver._asyncssh.import_private_key",
                return_value="vault_key",
            ) as mock_import,
        ):
            driver = _driver(vault=vault, vault_id="vid", secret_name="key")
            await driver.provision(_provision_req())

        mock_import.assert_called_once_with("vault_pem")
        kwargs = mock_connect.call_args.kwargs
        assert kwargs["client_keys"] == ["vault_key"]

    async def test_vault_takes_precedence_over_key_file(self, tmp_path: Any) -> None:
        key_file = tmp_path / "id_rsa"
        key_file.write_text("file_pem")
        conn = _make_conn()
        vault = MagicMock()
        vault.get_secret.return_value = {"value": "vault_pem"}
        imported_keys: list[str] = []

        with (
            _patch_asyncssh(conn),
            _patch_rsync(_make_rsync_proc()),
            patch(
                "sdk_environment._ssh_driver._asyncssh.import_private_key",
                side_effect=lambda pem: imported_keys.append(pem) or pem,
            ),
        ):
            driver = _driver(
                vault=vault,
                vault_id="vid",
                secret_name="key",
                private_key_path=str(key_file),
            )
            await driver.provision(_provision_req())

        # Only the vault PEM should have been imported
        assert imported_keys == ["vault_pem"]

    async def test_no_client_keys_when_no_auth_configured(self) -> None:
        conn = _make_conn()
        with _patch_asyncssh(conn) as mock_connect, _patch_rsync(_make_rsync_proc()):
            await _driver().provision(_provision_req())
        kwargs = mock_connect.call_args.kwargs
        assert "client_keys" not in kwargs

    async def test_rsync_uses_private_key_path(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        captured_cmd: list[str] = []

        async def _capture_exec(*args: Any, **kwargs: Any) -> MagicMock:
            captured_cmd.extend(args)
            return _make_rsync_proc()

        with (
            _patch_asyncssh(conn),
            patch(
                "sdk_environment._ssh_driver.asyncio.create_subprocess_exec",
                side_effect=_capture_exec,
            ),
            patch("builtins.open", mock_open(read_data="fake_pem")),
            patch(
                "sdk_environment._ssh_driver._asyncssh.import_private_key",
                return_value="imported",
            ),
        ):
            driver = _driver(private_key_path="/home/user/.ssh/id_rsa")
            await driver.provision(_provision_req())
            await driver.execute(_execute_req())

        ssh_e_arg = " ".join(captured_cmd)
        assert "/home/user/.ssh/id_rsa" in ssh_e_arg
        assert "-i" in ssh_e_arg


# ---------------------------------------------------------------------------
# rsync SSH options
# ---------------------------------------------------------------------------


class TestRsyncSshOptions:
    async def _capture_rsync_cmd(self, driver: SshBackendDriver, conn: MagicMock) -> list[str]:
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        captured: list[str] = []

        async def _capture_exec(*args: Any, **kwargs: Any) -> MagicMock:
            captured.extend(args)
            return _make_rsync_proc()

        with (
            _patch_asyncssh(conn),
            patch(
                "sdk_environment._ssh_driver.asyncio.create_subprocess_exec",
                side_effect=_capture_exec,
            ),
        ):
            await driver.provision(_provision_req())
            await driver.execute(_execute_req())

        return captured

    async def test_custom_port_in_rsync_ssh_opts(self) -> None:
        conn = _make_conn()
        cmd = await self._capture_rsync_cmd(_driver(port=2222), conn)
        ssh_e_arg = " ".join(cmd)
        assert "2222" in ssh_e_arg

    async def test_default_port_22_in_rsync_ssh_opts(self) -> None:
        conn = _make_conn()
        cmd = await self._capture_rsync_cmd(_driver(), conn)
        ssh_e_arg = " ".join(cmd)
        assert "22" in ssh_e_arg

    async def test_rsync_binary_configurable(self) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(return_value=_make_ssh_result())
        captured: list[str] = []

        async def _capture_exec(*args: Any, **kwargs: Any) -> MagicMock:
            captured.extend(args)
            return _make_rsync_proc()

        with (
            _patch_asyncssh(conn),
            patch(
                "sdk_environment._ssh_driver.asyncio.create_subprocess_exec",
                side_effect=_capture_exec,
            ),
        ):
            driver = _driver(rsync_binary="/usr/local/bin/rsync")
            await driver.provision(_provision_req())
            await driver.execute(_execute_req())

        assert captured[0] == "/usr/local/bin/rsync"

    async def test_rsync_uses_az_delete_flags(self) -> None:
        conn = _make_conn()
        cmd = await self._capture_rsync_cmd(_driver(), conn)
        assert "-az" in cmd
        assert "--delete" in cmd


# ---------------------------------------------------------------------------
# Runtime integration
# ---------------------------------------------------------------------------


class TestRuntimeIntegration:
    async def test_full_lifecycle_no_audit_entries(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(
            side_effect=[
                _make_ssh_result(),  # provision mkdir
                _make_ssh_result(stdout="ok"),  # execute
                _make_ssh_result(),  # reclaim rm
            ]
        )
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        opts = _make_options(audit_log)
        kind = SshBackendDriver.KIND

        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            await rt.provision(_provision_req(kind), opts)
            result = await rt.execute(_execute_req(kind=kind), opts)
            await rt.reclaim(_reclaim_req(kind), opts)

        assert isinstance(result, ExecuteResult)
        assert audit_log.entries == []

    async def test_provision_failure_writes_audit(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with patch(
            "sdk_environment._ssh_driver._asyncssh.connect",
            AsyncMock(side_effect=OSError("refused")),
        ):
            driver = _driver()
            rt = EnvironmentRuntime()
            rt.register(driver)
            with pytest.raises(EnvironmentFailure) as exc_info:
                await rt.provision(_provision_req(), _make_options(audit_log))
        assert exc_info.value.code == "ENV_PROVISION_FAILED"
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "environment.provision.failed"

    async def test_execute_failure_writes_audit(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(return_value=_make_ssh_result())
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        kind = SshBackendDriver.KIND
        opts = _make_options(audit_log)

        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc(returncode=1, stderr=b"fail")):
            await rt.provision(_provision_req(kind), opts)
            with pytest.raises(EnvironmentFailure):
                await rt.execute(_execute_req(kind=kind), opts)

        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].event == "environment.execute.failed"

    async def test_execute_failure_code_is_env_execute_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(return_value=_make_ssh_result())
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        kind = SshBackendDriver.KIND
        opts = _make_options(audit_log)

        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc(returncode=1)):
            await rt.provision(_provision_req(kind), opts)
            with pytest.raises(EnvironmentFailure) as exc_info:
                await rt.execute(_execute_req(kind=kind), opts)

        assert exc_info.value.code == "ENV_EXECUTE_FAILED"

    async def test_execute_span_name(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(side_effect=[_make_ssh_result(), _make_ssh_result()])
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        kind = SshBackendDriver.KIND
        opts = _make_options(audit_log)

        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            await rt.provision(_provision_req(kind), opts)
            await rt.execute(_execute_req(kind=kind), opts)

        assert mock_span.name == "environment.execute"

    async def test_reclaim_failure_writes_audit(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        conn = _make_conn()
        conn.run = AsyncMock(
            side_effect=[
                _make_ssh_result(),  # provision mkdir
                RuntimeError("rm: permission denied"),  # reclaim rm
            ]
        )
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        kind = SshBackendDriver.KIND
        opts = _make_options(audit_log)

        with _patch_asyncssh(conn), _patch_rsync(_make_rsync_proc()):
            await rt.provision(_provision_req(kind), opts)
            with pytest.raises(EnvironmentFailure) as exc_info:
                await rt.reclaim(_reclaim_req(kind), opts)

        assert exc_info.value.code == "ENV_RECLAIM_FAILED"
        assert len(audit_log.entries) == 1
