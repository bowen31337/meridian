from __future__ import annotations

from abc import ABC, abstractmethod

from ._types import ExecutionContext, SandboxResult, ToolDefinition


class ToolDispatcher(ABC):
    """
    Contract every tool-backend dispatcher must implement.

    Register a dispatcher once with Sandbox.register_dispatcher(); the Sandbox
    routes each execute() call to the dispatcher whose kind matches the tool's
    handler kind, wrapping the call with an OTel span, a structured event, and
    audit-log writes on failure.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Handler kind this dispatcher handles, e.g. 'in_process', 'subprocess'."""

    @abstractmethod
    async def dispatch(
        self,
        tool: ToolDefinition,
        input: dict,
        context: ExecutionContext,
    ) -> SandboxResult:
        """Dispatch a tool call and return its result."""
