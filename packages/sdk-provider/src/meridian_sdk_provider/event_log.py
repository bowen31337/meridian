from __future__ import annotations

from typing import Protocol


class ModelCallEventLog(Protocol):
    """Async write interface for recording model call lifecycle events.

    Implementations write to the session event log.  The router calls
    ``record_started`` once per routing attempt (primary + each fallback)
    immediately before dispatching to the provider.
    """

    async def record_started(
        self,
        *,
        session_id: str,
        routing_rule: str,
        provider_name: str,
        model: str,
    ) -> None: ...


class NoopModelCallEventLog:
    """Fallback used when no event log is wired."""

    async def record_started(
        self,
        *,
        session_id: str,
        routing_rule: str,
        provider_name: str,
        model: str,
    ) -> None:
        pass
