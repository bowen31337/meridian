from __future__ import annotations

from storage_event_log import EventLogRuntime


class EventLogModelCallAdapter:
    """Bridges ModelCallEventLog → EventLogRuntime.

    Writes a ``model_call.started`` session event for every routed call.
    Skipped silently when ``session_id`` is empty (non-session callers).
    """

    def __init__(self, runtime: EventLogRuntime) -> None:
        self._runtime = runtime

    async def record_started(
        self,
        *,
        session_id: str,
        routing_rule: str,
        provider_name: str,
        model: str,
    ) -> None:
        await self._runtime.append(
            session_id=session_id,
            event_type="model_call.started",
            data={
                "routing_rule": routing_rule,
                "provider_name": provider_name,
                "model": model,
            },
        )
