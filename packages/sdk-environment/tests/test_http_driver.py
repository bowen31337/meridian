"""
Tests for HttpBackendDriver.

Covers:
  - Successful provision / execute / reclaim: correct payloads POSTed,
    ExecuteResult fields mapped from response, no errors raised.
  - Server error response {"error": {...}}: driver raises RuntimeError with
    the server message; runtime wraps it as ENV_*_FAILED and writes the
    audit log.
  - HTTP transport failure (httpx raises): driver lets the exception
    propagate; runtime wraps it as ENV_*_FAILED.
  - Vault auth: when vault/vault_id/secret_name are configured the
    Authorization header is "Bearer <secret_value>"; absent otherwise.
  - Missing Vault secret: resolve returns None; no Authorization header.
  - Empty response body: treated as success for provision and reclaim.
  - execute duration_ms fallback: when the server omits duration_ms the
    driver fills in its own wall-clock measurement (positive float).
  - httpx unavailable: RuntimeError surfaced before any network call.
  - Kind constant: driver.kind == "system.http".
  - on_demand default is True; configurable to False.
  - Policy / capability delegation: driver returns the objects it was
    given at construction.
  - Runtime integration: HttpBackendDriver passes the conformance suite
    contract (provision → execute → reclaim, audit entries, spans).
"""

from __future__ import annotations

import importlib
import sys
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
    HttpBackendDriver,
    NetworkPolicy,
    ProvisionRequest,
    ReclaimRequest,
    RuntimeOptions,
)

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_URL = "https://exec.example.com"


def _driver(**kwargs: Any) -> HttpBackendDriver:
    return HttpBackendDriver(url=_BASE_URL, **kwargs)


def _provision_req(kind: str = HttpBackendDriver.KIND) -> ProvisionRequest:
    return ProvisionRequest(environment_id="env1", environment_kind=kind, session_id="sess1")


def _execute_req(kind: str = HttpBackendDriver.KIND) -> ExecuteRequest:
    return ExecuteRequest(
        environment_id="env1",
        environment_kind=kind,
        session_id="sess1",
        command=("echo", "hello"),
        stdin="stdin-data",
        env={"KEY": "val"},
        timeout_seconds=10,
    )


def _reclaim_req(kind: str = HttpBackendDriver.KIND) -> ReclaimRequest:
    return ReclaimRequest(environment_id="env1", environment_kind=kind, session_id="sess1")


def _ok_response(body: dict[str, Any] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.content = b"{}" if body is None else b"content"
    resp.json.return_value = body or {}
    resp.raise_for_status = MagicMock()
    return resp


def _execute_ok_response(
    stdout: str = "out",
    stderr: str = "err",
    exit_code: int = 0,
    duration_ms: float = 5.0,
) -> MagicMock:
    return _ok_response(
        {"result": {"stdout": stdout, "stderr": stderr, "exit_code": exit_code, "duration_ms": duration_ms}}
    )


def _error_response(code: str = "remote_error", message: str = "boom") -> MagicMock:
    resp = _ok_response({"error": {"code": code, "message": message}})
    resp.content = b'{"error":{}}'
    return resp


def _make_client_mock(response: MagicMock) -> MagicMock:
    client_mock = AsyncMock()
    client_mock.post = AsyncMock(return_value=response)
    client_ctx = MagicMock()
    client_ctx.__aenter__ = AsyncMock(return_value=client_mock)
    client_ctx.__aexit__ = AsyncMock(return_value=False)
    return client_ctx


def _make_options(audit: CapturingAuditLog) -> RuntimeOptions:
    return RuntimeOptions(audit_log=audit)


# ---------------------------------------------------------------------------
# Kind and defaults
# ---------------------------------------------------------------------------


class TestKindAndDefaults:
    def test_kind_is_system_http(self) -> None:
        assert _driver().kind == "system.http"

    def test_kind_constant(self) -> None:
        assert HttpBackendDriver.KIND == "system.http"

    def test_on_demand_default_true(self) -> None:
        assert _driver().on_demand is True

    def test_on_demand_configurable_false(self) -> None:
        assert _driver(on_demand=False).on_demand is False

    def test_network_policy_default(self) -> None:
        policy = _driver().network_policy()
        assert isinstance(policy, NetworkPolicy)

    def test_filesystem_policy_default(self) -> None:
        policy = _driver().filesystem_policy()
        assert isinstance(policy, FilesystemPolicy)

    def test_capability_envelope_default(self) -> None:
        caps = _driver().capability_envelope()
        assert isinstance(caps, CapabilityEnvelope)

    def test_custom_network_policy_returned(self) -> None:
        policy = NetworkPolicy(egress_allowed=True, allowed_hosts=("api.example.com",))
        assert _driver(network_policy=policy).network_policy() is policy

    def test_custom_filesystem_policy_returned(self) -> None:
        policy = FilesystemPolicy(read_globs=("**",))
        assert _driver(filesystem_policy=policy).filesystem_policy() is policy

    def test_custom_capability_envelope_returned(self) -> None:
        caps = CapabilityEnvelope(cpu_millicores=500)
        assert _driver(capability_envelope=caps).capability_envelope() is caps


# ---------------------------------------------------------------------------
# Vault auth
# ---------------------------------------------------------------------------


class TestVaultAuth:
    def _make_vault(self, value: str | None) -> MagicMock:
        vault = MagicMock()
        if value is None:
            vault.get_secret.return_value = None
        else:
            vault.get_secret.return_value = {"value": value, "created_at": "2026-01-01T00:00:00Z"}
        return vault

    async def test_bearer_token_in_header_when_vault_configured(self) -> None:
        vault = self._make_vault("secret-token")
        driver = _driver(vault=vault, vault_id="v1", secret_name="api_key")
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer secret-token"

    async def test_no_auth_header_when_vault_not_configured(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert "Authorization" not in kwargs["headers"]

    async def test_no_auth_header_when_secret_missing_from_vault(self) -> None:
        vault = self._make_vault(None)
        driver = _driver(vault=vault, vault_id="v1", secret_name="missing_key")
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert "Authorization" not in kwargs["headers"]

    async def test_vault_get_secret_called_with_correct_args(self) -> None:
        vault = self._make_vault("tok")
        driver = _driver(vault=vault, vault_id="my-vault", secret_name="my-key")
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        vault.get_secret.assert_called_once_with("my-vault", "my-key")


# ---------------------------------------------------------------------------
# provision — success
# ---------------------------------------------------------------------------


class TestProvisionSuccess:
    async def test_posts_to_provision_sub_path(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.provision(_provision_req())
        args, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert args[0] == f"{_BASE_URL}/provision"

    async def test_payload_contains_environment_id(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.provision(_provision_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert kwargs["json"]["environment_id"] == "env1"

    async def test_payload_contains_environment_kind(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.provision(_provision_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert kwargs["json"]["environment_kind"] == HttpBackendDriver.KIND

    async def test_payload_contains_session_id(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.provision(_provision_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert kwargs["json"]["session_id"] == "sess1"

    async def test_empty_body_succeeds(self) -> None:
        driver = _driver()
        resp = MagicMock()
        resp.content = b""
        resp.raise_for_status = MagicMock()
        client_ctx = _make_client_mock(resp)
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.provision(_provision_req())  # should not raise

    async def test_content_type_header(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.provision(_provision_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert kwargs["headers"]["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# provision — errors
# ---------------------------------------------------------------------------


class TestProvisionErrors:
    async def test_server_error_raises_runtime_error(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_error_response(message="disk full"))
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(RuntimeError, match="disk full"):
                await driver.provision(_provision_req())

    async def test_http_transport_error_propagates(self) -> None:
        import httpx

        driver = _driver()
        client_mock = AsyncMock()
        client_mock.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        client_ctx = MagicMock()
        client_ctx.__aenter__ = AsyncMock(return_value=client_mock)
        client_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(httpx.ConnectError):
                await driver.provision(_provision_req())

    async def test_runtime_wraps_as_env_provision_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        client_ctx = _make_client_mock(_error_response(message="out of capacity"))
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(EnvironmentFailure) as exc_info:
                await rt.provision(_provision_req(), _make_options(audit_log))
        assert exc_info.value.code == "ENV_PROVISION_FAILED"
        assert "out of capacity" in exc_info.value.message

    async def test_runtime_writes_audit_on_provision_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        client_ctx = _make_client_mock(_error_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(EnvironmentFailure):
                await rt.provision(_provision_req(), _make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "environment.provision.failed"


# ---------------------------------------------------------------------------
# execute — success
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    async def test_posts_to_execute_sub_path(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        args, _ = client_ctx.__aenter__.return_value.post.call_args
        assert args[0] == f"{_BASE_URL}/execute"

    async def test_payload_command_serialised_as_list(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert kwargs["json"]["command"] == ["echo", "hello"]

    async def test_payload_stdin_forwarded(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert kwargs["json"]["stdin"] == "stdin-data"

    async def test_payload_env_forwarded(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert kwargs["json"]["env"] == {"KEY": "val"}

    async def test_payload_timeout_seconds_forwarded(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert kwargs["json"]["timeout_seconds"] == 10

    async def test_result_stdout_mapped(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response(stdout="hello world"))
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert result.stdout == "hello world"

    async def test_result_stderr_mapped(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response(stderr="warn: something"))
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert result.stderr == "warn: something"

    async def test_result_exit_code_mapped(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response(exit_code=42))
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert result.exit_code == 42

    async def test_result_duration_ms_from_server(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response(duration_ms=123.4))
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert result.duration_ms == pytest.approx(123.4)

    async def test_result_duration_ms_fallback_when_absent(self) -> None:
        driver = _driver()
        resp = _ok_response({"result": {"stdout": "", "stderr": "", "exit_code": 0}})
        client_ctx = _make_client_mock(resp)
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms >= 0.0

    async def test_returns_execute_result_type(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert isinstance(result, ExecuteResult)

    async def test_stdout_is_str(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert isinstance(result.stdout, str)

    async def test_exit_code_is_int(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert isinstance(result.exit_code, int)

    async def test_duration_ms_is_float(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            result = await driver.execute(_execute_req())
        assert isinstance(result.duration_ms, float)


# ---------------------------------------------------------------------------
# execute — errors
# ---------------------------------------------------------------------------


class TestExecuteErrors:
    async def test_server_error_raises_runtime_error(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_error_response(message="command not found"))
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(RuntimeError, match="command not found"):
                await driver.execute(_execute_req())

    async def test_http_transport_error_propagates(self) -> None:
        import httpx

        driver = _driver()
        client_mock = AsyncMock()
        client_mock.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        client_ctx = MagicMock()
        client_ctx.__aenter__ = AsyncMock(return_value=client_mock)
        client_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(httpx.TimeoutException):
                await driver.execute(_execute_req())

    async def test_runtime_wraps_as_env_execute_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        client_ctx = _make_client_mock(_error_response(message="oom"))
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(EnvironmentFailure) as exc_info:
                await rt.execute(_execute_req(), _make_options(audit_log))
        assert exc_info.value.code == "ENV_EXECUTE_FAILED"
        assert "oom" in exc_info.value.message

    async def test_runtime_writes_audit_on_execute_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        client_ctx = _make_client_mock(_error_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(EnvironmentFailure):
                await rt.execute(_execute_req(), _make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].event == "environment.execute.failed"

    async def test_transport_error_writes_audit(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        import httpx

        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        client_mock = AsyncMock()
        client_mock.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        client_ctx = MagicMock()
        client_ctx.__aenter__ = AsyncMock(return_value=client_mock)
        client_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(EnvironmentFailure):
                await rt.execute(_execute_req(), _make_options(audit_log))
        assert len(audit_log.entries) == 1


# ---------------------------------------------------------------------------
# reclaim — success
# ---------------------------------------------------------------------------


class TestReclaimSuccess:
    async def test_posts_to_reclaim_sub_path(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.reclaim(_reclaim_req())
        args, _ = client_ctx.__aenter__.return_value.post.call_args
        assert args[0] == f"{_BASE_URL}/reclaim"

    async def test_payload_fields_forwarded(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.reclaim(_reclaim_req())
        _, kwargs = client_ctx.__aenter__.return_value.post.call_args
        assert kwargs["json"]["environment_id"] == "env1"
        assert kwargs["json"]["session_id"] == "sess1"

    async def test_empty_body_succeeds(self) -> None:
        driver = _driver()
        resp = MagicMock()
        resp.content = b""
        resp.raise_for_status = MagicMock()
        client_ctx = _make_client_mock(resp)
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.reclaim(_reclaim_req())  # should not raise


# ---------------------------------------------------------------------------
# reclaim — errors
# ---------------------------------------------------------------------------


class TestReclaimErrors:
    async def test_server_error_raises_runtime_error(self) -> None:
        driver = _driver()
        client_ctx = _make_client_mock(_error_response(message="still running"))
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(RuntimeError, match="still running"):
                await driver.reclaim(_reclaim_req())

    async def test_runtime_wraps_as_env_reclaim_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        client_ctx = _make_client_mock(_error_response(message="container locked"))
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(EnvironmentFailure) as exc_info:
                await rt.reclaim(_reclaim_req(), _make_options(audit_log))
        assert exc_info.value.code == "ENV_RECLAIM_FAILED"

    async def test_runtime_writes_audit_on_reclaim_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        client_ctx = _make_client_mock(_error_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            with pytest.raises(EnvironmentFailure):
                await rt.reclaim(_reclaim_req(), _make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].event == "environment.reclaim.failed"


# ---------------------------------------------------------------------------
# httpx unavailable
# ---------------------------------------------------------------------------


class TestHttpxUnavailable:
    async def test_provision_raises_when_httpx_missing(self) -> None:
        with patch("sdk_environment._http_driver._HTTPX_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="httpx"):
                await _driver().provision(_provision_req())

    async def test_execute_raises_when_httpx_missing(self) -> None:
        with patch("sdk_environment._http_driver._HTTPX_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="httpx"):
                await _driver().execute(_execute_req())

    async def test_reclaim_raises_when_httpx_missing(self) -> None:
        with patch("sdk_environment._http_driver._HTTPX_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="httpx"):
                await _driver().reclaim(_reclaim_req())


# ---------------------------------------------------------------------------
# URL trailing slash normalisation
# ---------------------------------------------------------------------------


class TestUrlNormalisation:
    async def test_trailing_slash_stripped(self) -> None:
        driver = HttpBackendDriver(url="https://example.com/exec/")
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await driver.execute(_execute_req())
        args, _ = client_ctx.__aenter__.return_value.post.call_args
        assert args[0] == "https://example.com/exec/execute"


# ---------------------------------------------------------------------------
# Runtime integration (conformance)
# ---------------------------------------------------------------------------


class TestRuntimeIntegration:
    async def test_full_lifecycle_no_audit_entries(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        opts = _make_options(audit_log)
        kind = HttpBackendDriver.KIND

        client_ctx_p = _make_client_mock(_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx_p):
            await rt.provision(_provision_req(kind), opts)

        client_ctx_e = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx_e):
            result = await rt.execute(_execute_req(kind), opts)

        client_ctx_r = _make_client_mock(_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx_r):
            await rt.reclaim(_reclaim_req(kind), opts)

        assert isinstance(result, ExecuteResult)
        assert audit_log.entries == []

    async def test_execute_emits_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        driver = _driver()
        rt = EnvironmentRuntime()
        rt.register(driver)
        client_ctx = _make_client_mock(_execute_ok_response())
        with patch("sdk_environment._http_driver._httpx.AsyncClient", return_value=client_ctx):
            await rt.execute(_execute_req(), _make_options(audit_log))
        assert mock_span.name == "environment.execute"

    async def test_capability_envelope_has_required_fields(self) -> None:
        caps = _driver().capability_envelope()
        assert isinstance(caps.cpu_millicores, int)
        assert isinstance(caps.memory_mb, int)
        assert isinstance(caps.timeout_seconds, int)
        assert isinstance(caps.can_write_filesystem, bool)
        assert isinstance(caps.network, NetworkPolicy)
        assert isinstance(caps.filesystem, FilesystemPolicy)

    async def test_network_policy_has_required_fields(self) -> None:
        policy = _driver().network_policy()
        assert isinstance(policy.egress_allowed, bool)
        assert isinstance(policy.allowed_hosts, tuple)
        assert isinstance(policy.blocked_hosts, tuple)
