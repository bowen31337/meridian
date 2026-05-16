from __future__ import annotations

from abc import ABC, abstractmethod

from ._types import (
    ChannelCapabilities,
    SendRequest,
    SendResult,
    StartRequest,
    StopRequest,
)


class ChannelDriver(ABC):
    """
    Contract every channel backend must implement.

    Register a driver once with ChannelRuntime.register(); the runtime
    dispatches start / send / stop to the correct driver by kind,
    wrapping each call with an OTel span, a structured invocation event, and
    audit-log writes on failure.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Globally unique kind identifier, e.g. 'meridian.slack'."""

    @abstractmethod
    async def start(self, request: StartRequest) -> None:
        """Connect to the channel and begin accepting messages."""

    @abstractmethod
    async def send(self, request: SendRequest) -> SendResult:
        """Send a message over an active channel connection."""

    @abstractmethod
    async def stop(self, request: StopRequest) -> None:
        """Disconnect from the channel and release resources."""

    @abstractmethod
    def capabilities(self) -> ChannelCapabilities:
        """Return the feature and resource limits this driver enforces."""
