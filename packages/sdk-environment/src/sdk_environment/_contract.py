from __future__ import annotations

from abc import ABC, abstractmethod

from ._types import (
    CapabilityEnvelope,
    ExecuteRequest,
    ExecuteResult,
    FilesystemPolicy,
    NetworkPolicy,
    ProvisionRequest,
    ReclaimRequest,
)


class EnvironmentDriver(ABC):
    """
    Contract every environment backend must implement.

    Register a driver once with EnvironmentRuntime.register(); the runtime
    dispatches provision / execute / reclaim to the correct driver by kind,
    wrapping each call with an OTel span, a structured invocation event, and
    audit-log writes on failure.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Globally unique kind identifier, e.g. 'meridian.python'."""

    @abstractmethod
    async def provision(self, request: ProvisionRequest) -> None:
        """Allocate and initialise an environment instance."""

    @abstractmethod
    async def execute(self, request: ExecuteRequest) -> ExecuteResult:
        """Execute a command inside an active environment instance."""

    @abstractmethod
    async def reclaim(self, request: ReclaimRequest) -> None:
        """Destroy and release an environment instance."""

    @abstractmethod
    def network_policy(self) -> NetworkPolicy:
        """Return the network access policy this driver enforces."""

    @abstractmethod
    def filesystem_policy(self) -> FilesystemPolicy:
        """
        Return the filesystem access policy this driver enforces.

        Container and SSH backends must mount only $WORKSPACE when provisioning,
        providing a physical enforcement layer in addition to the software-level
        FilesystemGate that checks this policy on every fs.* invocation.
        """

    @abstractmethod
    def capability_envelope(self) -> CapabilityEnvelope:
        """Return the resource limits and permission set this driver enforces."""

    @property
    def on_demand(self) -> bool:
        """Return True for container/serverless backends that provision per tool-call.

        When True, WorkerPool will provision → execute → reclaim inline for every
        call rather than keeping the worker alive in the warm pool.
        Defaults to False (pool-backed, provision-on-first-use).
        """
        return False
