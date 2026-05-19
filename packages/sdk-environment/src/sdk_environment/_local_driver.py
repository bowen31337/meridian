"""Local environment backend driver.

Spawns a subprocess in an isolated scratch directory rooted at workspace_path.
Each provisioned environment gets its own subdirectory:
  {workspace_path}/{environment_id}/

Subprocess environment:
  By default the full host process environment (os.environ) is inherited and
  per-execution vars from ExecuteRequest.env are merged on top.  When
  env_passthrough is non-empty only the named vars from os.environ are
  forwarded, providing opt-in env isolation.

Lifecycle:
  provision  — creates the per-environment scratch directory
  execute    — spawns a subprocess with cwd=scratch_dir, captures
               stdout / stderr / exit-code
  reclaim    — removes the scratch directory (shutil.rmtree)

Default backend for v1; no external runtime dependencies beyond the Python
stdlib.

Warm pool semantics (on_demand=False by default): the scratch directory is
provisioned on first use and held warm by WorkerPool for the configured TTL,
avoiding repeated mkdir / rmdir overhead across calls within the same
session.  The WorkerPool reaper reclaims idle environments after the TTL.

On any failure the driver raises; the EnvironmentRuntime wraps the exception
as EnvironmentFailure, surfaces the message to the caller, and writes the
failure to the audit log.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time

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


def _ms_since(start: float) -> float:
    return (time.monotonic() - start) * 1000.0


class LocalBackendDriver(EnvironmentDriver):
    """
    Environment backend that spawns subprocesses in an isolated scratch directory.

    Lifecycle (on_demand=False by default — pool-backed, provision-on-first-use):
      provision  — creates per-environment scratch directory under workspace_path
      execute    — spawns a subprocess in the scratch directory, captures
                   stdout / stderr / exit-code
      reclaim    — removes the scratch directory and all its contents

    On any failure the driver raises; the EnvironmentRuntime wraps the
    exception as EnvironmentFailure, surfaces the message to the caller,
    and writes the failure to the audit log.
    """

    KIND = "system.local"

    def __init__(
        self,
        *,
        workspace_path: str = "/tmp/meridian-local",
        env_passthrough: tuple[str, ...] = (),
        timeout_s: float = 30.0,
        on_demand: bool = False,
        network_policy: NetworkPolicy | None = None,
        filesystem_policy: FilesystemPolicy | None = None,
        capability_envelope: CapabilityEnvelope | None = None,
    ) -> None:
        self._workspace_path = workspace_path.rstrip("/")
        self._env_passthrough = env_passthrough
        self._timeout_s = timeout_s
        self._on_demand = on_demand
        self._network_policy = network_policy or NetworkPolicy()
        self._filesystem_policy = filesystem_policy or FilesystemPolicy()
        self._capability_envelope = capability_envelope or CapabilityEnvelope()
        # Active scratch directories keyed by environment_id
        self._scratch_dirs: dict[str, str] = {}

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

    def _scratch_dir(self, environment_id: str) -> str:
        return f"{self._workspace_path}/{environment_id}"

    def _build_subprocess_env(self, request: ExecuteRequest) -> dict[str, str]:
        if self._env_passthrough:
            env: dict[str, str] = {}
            for var in self._env_passthrough:
                val = os.environ.get(var)
                if val is not None:
                    env[var] = val
        else:
            env = os.environ.copy()
        env.update(request.env)
        return env

    # ------------------------------------------------------------------
    # EnvironmentDriver
    # ------------------------------------------------------------------

    async def provision(self, request: ProvisionRequest) -> None:
        scratch = self._scratch_dir(request.environment_id)
        try:
            os.makedirs(scratch, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to create scratch directory {scratch!r}: {exc}"
            ) from exc
        self._scratch_dirs[request.environment_id] = scratch

    async def execute(self, request: ExecuteRequest) -> ExecuteResult:
        scratch = self._scratch_dirs.get(request.environment_id)
        if scratch is None:
            raise RuntimeError(
                f"Environment {request.environment_id!r} is not provisioned; "
                "call provision() before execute()."
            )
        timeout_s = (
            float(request.timeout_seconds)
            if request.timeout_seconds is not None
            else self._timeout_s
        )
        env = self._build_subprocess_env(request)
        stdin_data = (request.stdin or "").encode()

        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *request.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=scratch,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(stdin_data),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise
        duration_ms = _ms_since(start)

        return ExecuteResult(
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            exit_code=proc.returncode if proc.returncode is not None else 0,
            duration_ms=duration_ms,
        )

    async def reclaim(self, request: ReclaimRequest) -> None:
        scratch = self._scratch_dirs.pop(request.environment_id, None)
        if scratch is None:
            return
        shutil.rmtree(scratch, ignore_errors=True)
