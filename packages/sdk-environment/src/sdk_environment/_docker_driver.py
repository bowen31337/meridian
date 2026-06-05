"""Docker environment backend driver.

Provisions on demand at tool-call time per the Anthropic on-demand pattern.
Each provision starts a detached container from the configured image; execute
runs a command inside it via ``docker exec``; reclaim stops and removes the
container.

Workspace mounting:
  When ``workspace_path`` is supplied the host directory is bind-mounted into
  the container at ``workspace_mount_target`` (default ``/workspace``) and the
  command is run with that directory as the working directory.

Capability envelope enforcement via cgroup flags:
  ``cpu_millicores`` → ``--cpus``   (e.g. 1000 → 1.0)
  ``memory_mb``      → ``--memory`` (e.g. 512 → 512m)

Network isolation:
  When the NetworkPolicy has ``egress_allowed=False`` and no ``allowed_hosts``
  the container is started with ``--network=none``.

Requires the ``docker`` CLI; no additional Python packages are needed.
"""

from __future__ import annotations

import asyncio
import os
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


class DockerBackendDriver(EnvironmentDriver):
    """
    Environment backend that runs commands inside Docker containers.

    Lifecycle (on_demand=True by default — provision → execute → reclaim per call):
      provision  — starts a detached container (docker run -d) with cgroup
                   resource limits from the CapabilityEnvelope, an optional
                   workspace bind-mount, and env_passthrough vars
      execute    — runs the command inside the container (docker exec),
                   captures stdout / stderr / exit-code
      reclaim    — force-removes the container (docker rm -f)

    On any failure the driver raises; the EnvironmentRuntime wraps the
    exception as EnvironmentFailure, surfaces the message to the caller,
    and writes the failure to the audit log.
    """

    KIND = "system.docker"

    def __init__(
        self,
        *,
        image: str,
        workspace_path: str | None = None,
        workspace_mount_target: str = "/workspace",
        env_passthrough: tuple[str, ...] = (),
        docker_binary: str = "docker",
        timeout_s: float = 30.0,
        on_demand: bool = True,
        network_policy: NetworkPolicy | None = None,
        filesystem_policy: FilesystemPolicy | None = None,
        capability_envelope: CapabilityEnvelope | None = None,
    ) -> None:
        self._image = image
        self._workspace_path = workspace_path
        self._workspace_mount_target = workspace_mount_target
        self._env_passthrough = env_passthrough
        self._docker_binary = docker_binary
        self._timeout_s = timeout_s
        self._on_demand = on_demand
        self._network_policy = network_policy or NetworkPolicy()
        self._filesystem_policy = filesystem_policy or FilesystemPolicy()
        self._capability_envelope = capability_envelope or CapabilityEnvelope()
        # Active container names keyed by environment_id
        self._containers: dict[str, str] = {}

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

    def _container_name(self, environment_id: str) -> str:
        return f"meridian-{environment_id}"

    def _build_run_cmd(self, environment_id: str) -> list[str]:
        caps = self._capability_envelope
        policy = self._network_policy
        container_name = self._container_name(environment_id)

        cmd = [self._docker_binary, "run", "-d", "--name", container_name]

        # cgroup resource limits from CapabilityEnvelope
        cmd += ["--cpus", f"{caps.cpu_millicores / 1000.0:.3f}"]
        cmd += ["--memory", f"{caps.memory_mb}m"]

        # Network isolation: no egress + no specific allowed hosts → no network
        if not policy.egress_allowed and not policy.allowed_hosts:
            cmd += ["--network", "none"]

        # Workspace bind-mount: only $WORKSPACE is exposed (physical enforcement layer)
        if self._workspace_path:
            cmd += [
                "--volume",
                f"{self._workspace_path}:{self._workspace_mount_target}",
            ]

        # env_passthrough: pass named host env vars into the container at start
        for var in self._env_passthrough:
            val = os.environ.get(var)
            if val is not None:
                cmd += ["--env", f"{var}={val}"]

        cmd += [self._image, "sleep", "infinity"]
        return cmd

    def _build_exec_cmd(self, container_name: str, request: ExecuteRequest) -> list[str]:
        cmd = [self._docker_binary, "exec", "-i"]

        # Per-execution env vars forwarded via --env flags
        for k, v in request.env.items():
            cmd += ["--env", f"{k}={v}"]

        # Working directory inside the container (when workspace is mounted)
        if self._workspace_path:
            cmd += ["--workdir", self._workspace_mount_target]

        cmd.append(container_name)
        cmd.extend(request.command)
        return cmd

    async def _run_subprocess(
        self,
        cmd: list[str],
        *,
        stdin_data: bytes | None = None,
        timeout_s: float,
    ) -> tuple[int, bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(stdin_data),
                timeout=timeout_s,
            )
        except TimeoutError:
            proc.kill()
            raise
        return proc.returncode if proc.returncode is not None else 0, stdout_b, stderr_b

    # ------------------------------------------------------------------
    # EnvironmentDriver
    # ------------------------------------------------------------------

    async def provision(self, request: ProvisionRequest) -> None:
        cmd = self._build_run_cmd(request.environment_id)
        returncode, _stdout_b, stderr_b = await self._run_subprocess(cmd, timeout_s=self._timeout_s)
        if returncode != 0:
            raise RuntimeError(
                f"docker run failed (exit {returncode}): "
                f"{stderr_b.decode(errors='replace').strip()}"
            )
        self._containers[request.environment_id] = self._container_name(request.environment_id)

    async def execute(self, request: ExecuteRequest) -> ExecuteResult:
        container_name = self._containers.get(request.environment_id)
        if container_name is None:
            raise RuntimeError(
                f"Environment {request.environment_id!r} is not provisioned; "
                "call provision() before execute()."
            )
        timeout_s = (
            float(request.timeout_seconds)
            if request.timeout_seconds is not None
            else self._timeout_s
        )
        cmd = self._build_exec_cmd(container_name, request)
        stdin_data = (request.stdin or "").encode()

        start = time.monotonic()
        try:
            returncode, stdout_b, stderr_b = await self._run_subprocess(
                cmd, stdin_data=stdin_data, timeout_s=timeout_s
            )
        except TimeoutError:
            raise
        duration_ms = _ms_since(start)

        return ExecuteResult(
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            exit_code=returncode,
            duration_ms=duration_ms,
        )

    async def reclaim(self, request: ReclaimRequest) -> None:
        container_name = self._containers.pop(request.environment_id, None)
        if container_name is None:
            return
        cmd = [self._docker_binary, "rm", "-f", container_name]
        # Best-effort cleanup: don't raise on non-zero exit (container may not exist)
        try:
            await self._run_subprocess(cmd, timeout_s=self._timeout_s)
        except TimeoutError:
            raise
