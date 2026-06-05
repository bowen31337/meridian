from __future__ import annotations

import time
from typing import Any, Protocol

from ._contract import EnvironmentDriver
from ._types import (
    CapabilityEnvelope,
    ExecuteRequest,
    ExecuteResult,
    FilesystemPolicy,
    NetworkPolicy,
    ProvisionRequest,
    ReclaimRequest,
)

try:
    import httpx as _httpx
except ImportError:
    _httpx = None

_HTTPX_AVAILABLE = _httpx is not None


def _ms_since(start: float) -> float:
    return (time.monotonic() - start) * 1000.0


class VaultCredentialResolver(Protocol):
    """Minimal protocol for fetching a secret value from a vault backend."""

    def get_secret(self, vault_id: str, name: str) -> dict[str, Any] | None: ...


class HttpBackendDriver(EnvironmentDriver):
    """
    Environment backend that delegates execution to a remote HTTP service.

    Each operation posts to a sub-path of the configured base URL:
      POST {url}/provision  — allocate the environment
      POST {url}/execute    — run a command; expects Sandbox return shape
      POST {url}/reclaim    — release the environment

    Request body is the JSON-serialised operation fields.  When a vault,
    vault_id, and secret_name are supplied the secret value is resolved and
    sent as a Bearer token in the Authorization header.

    Expected response shape for execute:
      Success: {"result": {"stdout": str, "stderr": str,
                            "exit_code": int, "duration_ms": float}}
      Error:   {"error": {"code": str, "message": str}}

    For provision and reclaim any 2xx response with no "error" key succeeds.
    On any failure the driver raises; the runtime wraps the exception in
    EnvironmentFailure, surfaces the message to the caller, and writes the
    failure to the audit log.

    Requires httpx; install with:
      pip install 'meridian-sdk-environment[http]'
    """

    KIND = "system.http"

    def __init__(
        self,
        *,
        url: str,
        vault: VaultCredentialResolver | None = None,
        vault_id: str | None = None,
        secret_name: str | None = None,
        timeout_s: float = 30.0,
        on_demand: bool = True,
        network_policy: NetworkPolicy | None = None,
        filesystem_policy: FilesystemPolicy | None = None,
        capability_envelope: CapabilityEnvelope | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._vault = vault
        self._vault_id = vault_id
        self._secret_name = secret_name
        self._timeout_s = timeout_s
        self._on_demand = on_demand
        self._network_policy = network_policy or NetworkPolicy()
        self._filesystem_policy = filesystem_policy or FilesystemPolicy()
        self._capability_envelope = capability_envelope or CapabilityEnvelope()

    @property
    def kind(self) -> str:
        return self.KIND

    @property
    def on_demand(self) -> bool:
        return self._on_demand

    def network_policy(self) -> NetworkPolicy:
        return self._network_policy

    def filesystem_policy(self) -> FilesystemPolicy:
        return self._filesystem_policy

    def capability_envelope(self) -> CapabilityEnvelope:
        return self._capability_envelope

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_token(self) -> str | None:
        if self._vault is None or self._vault_id is None or self._secret_name is None:
            return None
        record = self._vault.get_secret(self._vault_id, self._secret_name)
        if record is None:
            return None
        return record.get("value")

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        token = self._resolve_token()
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _post(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not _HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx is required for HttpBackendDriver. "
                "Install with: pip install 'meridian-sdk-environment[http]'"
            )
        assert _httpx is not None
        async with _httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(
                f"{self._url}/{operation}",
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # EnvironmentDriver
    # ------------------------------------------------------------------

    async def provision(self, request: ProvisionRequest) -> None:
        payload: dict[str, Any] = {
            "environment_id": request.environment_id,
            "environment_kind": request.environment_kind,
            "session_id": request.session_id,
        }
        data = await self._post("provision", payload)
        if "error" in data:
            err = data["error"]
            raise RuntimeError(err.get("message", "provision failed"))

    async def execute(self, request: ExecuteRequest) -> ExecuteResult:
        payload: dict[str, Any] = {
            "environment_id": request.environment_id,
            "environment_kind": request.environment_kind,
            "session_id": request.session_id,
            "command": list(request.command),
            "stdin": request.stdin,
            "env": request.env,
            "timeout_seconds": request.timeout_seconds,
        }
        start = time.monotonic()
        data = await self._post("execute", payload)

        if "error" in data:
            err = data["error"]
            raise RuntimeError(err.get("message", "execute failed"))

        result = data.get("result", {})
        return ExecuteResult(
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            exit_code=int(result.get("exit_code", 0)),
            duration_ms=float(result.get("duration_ms", _ms_since(start))),
        )

    async def reclaim(self, request: ReclaimRequest) -> None:
        payload: dict[str, Any] = {
            "environment_id": request.environment_id,
            "environment_kind": request.environment_kind,
            "session_id": request.session_id,
        }
        data = await self._post("reclaim", payload)
        if "error" in data:
            err = data["error"]
            raise RuntimeError(err.get("message", "reclaim failed"))
