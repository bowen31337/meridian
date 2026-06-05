"""SSH environment backend driver.

Opens an SSH session to a remote host, syncs the local scratch directory to
the remote via rsync (over SSH), runs the entrypoint, and captures
stdout / stderr / exit-code.

Host key verification is mandatory — ``known_hosts`` must point to an
OpenSSH-format known_hosts file.  The driver refuses to connect if the
remote host key is absent from that file (StrictHostKeyChecking=yes).

Authentication (in priority order):
  1. Private key resolved from vault (vault + vault_id + secret_name).
  2. Private key read from ``private_key_path``.
  3. asyncssh default key search (~/.ssh/id_*, SSH agent, etc.).

Each provisioned environment gets an isolated scratch directory on the remote:
  {remote_scratch_base}/{environment_id}/

Requires asyncssh; install with:
  pip install 'meridian-sdk-environment[ssh]'
"""

from __future__ import annotations

import asyncio
import os
import shlex
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

_asyncssh: Any
try:
    import asyncssh as _asyncssh  # pyright: ignore[reportMissingImports]
except ImportError:
    _asyncssh = None

_ASYNCSSH_AVAILABLE = _asyncssh is not None


def _ms_since(start: float) -> float:
    return (time.monotonic() - start) * 1000.0


class VaultCredentialResolver(Protocol):
    """Minimal protocol for fetching a secret value from a vault backend."""

    def get_secret(self, vault_id: str, name: str) -> dict[str, Any] | None: ...


class SshBackendDriver(EnvironmentDriver):
    """
    Environment backend that executes commands on a remote host over SSH.

    Lifecycle:
      provision  — opens an SSH connection to the remote host, creates the
                   per-environment scratch directory on the remote
      execute    — rsyncs the local scratch directory to the remote,
                   runs the command inside the remote scratch directory,
                   captures stdout / stderr / exit-code
      reclaim    — removes the remote scratch directory, closes the SSH
                   connection

    Host key verification is mandatory.  Supply a path to an OpenSSH
    known_hosts file via ``known_hosts``; connections whose host key is
    absent from that file are rejected.

    On any failure the driver raises a native exception; the
    EnvironmentRuntime wraps it as EnvironmentFailure, surfaces the message
    to the caller, and writes the failure to the audit log.

    Requires asyncssh and rsync; install with:
      pip install 'meridian-sdk-environment[ssh]'
    """

    KIND = "system.ssh"

    def __init__(
        self,
        *,
        host: str,
        port: int = 22,
        username: str,
        known_hosts: str,
        private_key_path: str | None = None,
        vault: VaultCredentialResolver | None = None,
        vault_id: str | None = None,
        secret_name: str | None = None,
        local_scratch_base: str = "/tmp/meridian-scratch",
        remote_scratch_base: str = "/tmp/meridian-scratch",
        rsync_binary: str = "rsync",
        timeout_s: float = 30.0,
        on_demand: bool = False,
        network_policy: NetworkPolicy | None = None,
        filesystem_policy: FilesystemPolicy | None = None,
        capability_envelope: CapabilityEnvelope | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._known_hosts = known_hosts
        self._private_key_path = private_key_path
        self._vault = vault
        self._vault_id = vault_id
        self._secret_name = secret_name
        self._local_scratch_base = local_scratch_base.rstrip("/")
        self._remote_scratch_base = remote_scratch_base.rstrip("/")
        self._rsync_binary = rsync_binary
        self._timeout_s = timeout_s
        self._on_demand = on_demand
        self._network_policy = network_policy or NetworkPolicy()
        self._filesystem_policy = filesystem_policy or FilesystemPolicy()
        self._capability_envelope = capability_envelope or CapabilityEnvelope()
        # Active SSH connections keyed by environment_id
        self._connections: dict[str, Any] = {}

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

    def _resolve_private_key_pem(self) -> str | None:
        """Return PEM text of the private key from the vault or key file."""
        if self._vault is not None and self._vault_id and self._secret_name:
            record = self._vault.get_secret(self._vault_id, self._secret_name)
            if record is not None:
                pem = record.get("value")
                if pem:
                    return str(pem)
        if self._private_key_path:
            with open(os.path.expanduser(self._private_key_path)) as fh:
                return fh.read()
        return None

    def _remote_dir(self, environment_id: str) -> str:
        return f"{self._remote_scratch_base}/{environment_id}"

    def _local_dir(self, environment_id: str) -> str:
        return f"{self._local_scratch_base}/{environment_id}"

    async def _open_connection(self, environment_id: str) -> Any:
        if not _ASYNCSSH_AVAILABLE:
            raise RuntimeError(
                "asyncssh is required for SshBackendDriver. "
                "Install with: pip install 'meridian-sdk-environment[ssh]'"
            )
        connect_kwargs: dict[str, Any] = {
            "host": self._host,
            "port": self._port,
            "username": self._username,
            "known_hosts": self._known_hosts,
        }
        pem = self._resolve_private_key_pem()
        if pem is not None:
            connect_kwargs["client_keys"] = [_asyncssh.import_private_key(pem)]
        conn = await _asyncssh.connect(**connect_kwargs)
        self._connections[environment_id] = conn
        return conn

    async def _rsync_to_remote(self, environment_id: str, timeout_s: float) -> None:
        """Sync the local scratch directory to the remote host via rsync over SSH."""
        local_src = self._local_dir(environment_id) + "/"
        remote_dst = f"{self._username}@{self._host}:{self._remote_dir(environment_id)}/"
        ssh_parts = [
            "ssh",
            "-p",
            str(self._port),
            "-o",
            f"UserKnownHostsFile={self._known_hosts}",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
        if self._private_key_path:
            ssh_parts += ["-i", os.path.expanduser(self._private_key_path)]
        cmd = [
            self._rsync_binary,
            "-az",
            "--delete",
            "-e",
            shlex.join(ssh_parts),
            local_src,
            remote_dst,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            proc.kill()
            raise
        if proc.returncode != 0:
            raise RuntimeError(
                f"rsync failed (exit {proc.returncode}): "
                f"{stderr_b.decode(errors='replace').strip()}"
            )

    # ------------------------------------------------------------------
    # EnvironmentDriver
    # ------------------------------------------------------------------

    async def provision(self, request: ProvisionRequest) -> None:
        conn = await self._open_connection(request.environment_id)
        remote_dir = self._remote_dir(request.environment_id)
        result = await conn.run(f"mkdir -p {shlex.quote(remote_dir)}")
        if result.exit_status != 0:
            raise RuntimeError(
                f"Failed to create remote scratch directory {remote_dir!r}: {result.stderr}"
            )

    async def execute(self, request: ExecuteRequest) -> ExecuteResult:
        conn = self._connections.get(request.environment_id)
        if conn is None:
            raise RuntimeError(
                f"Environment {request.environment_id!r} is not provisioned; "
                "call provision() before execute()."
            )
        timeout_s = (
            float(request.timeout_seconds)
            if request.timeout_seconds is not None
            else self._timeout_s
        )

        await self._rsync_to_remote(request.environment_id, timeout_s)

        remote_dir = self._remote_dir(request.environment_id)
        env_exports = " && ".join(f"export {k}={shlex.quote(v)}" for k, v in request.env.items())
        command_str = shlex.join(request.command)
        full_command = (
            f"cd {shlex.quote(remote_dir)}"
            + (f" && {env_exports}" if env_exports else "")
            + f" && {command_str}"
        )

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                conn.run(full_command, input=request.stdin or ""),
                timeout=timeout_s,
            )
        except TimeoutError:
            raise
        duration_ms = _ms_since(start)

        stdout = result.stdout
        stderr = result.stderr
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")

        return ExecuteResult(
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=int(result.exit_status or 0),
            duration_ms=duration_ms,
        )

    async def reclaim(self, request: ReclaimRequest) -> None:
        conn = self._connections.pop(request.environment_id, None)
        if conn is None:
            return
        remote_dir = self._remote_dir(request.environment_id)
        try:
            await conn.run(f"rm -rf {shlex.quote(remote_dir)}")
        finally:
            conn.close()
