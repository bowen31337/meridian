from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EventLogWriter(ABC):
    """
    Contract every event-log backend must implement.

    Pass a concrete implementation to EventLogRuntime, which wraps each call
    with an OTel span, a structured invocation event, and audit-log writes on
    failure.
    """

    @abstractmethod
    async def append(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> int:
        """Append one event and return its monotonic seq number (0-indexed per session)."""
