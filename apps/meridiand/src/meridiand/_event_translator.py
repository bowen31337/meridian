"""ModelEvent → SessionEvent translation for the System OAuth provider stream.

Contract 4: every ModelEvent emitted by the Claude Code CLI subprocess is
translated into zero or more (EventType, data) pairs so the session event log
remains the authoritative record of what the model produced.

Translation table
-----------------
TextDeltaEvent       → message.delta        {"kind": "text", "text": ..., "model_call_number": N}
ThinkingDeltaEvent   → message.delta        {"kind": "thinking", "thinking": ...,
                                             "model_call_number": N}
ToolUseStartEvent    → (buffered; no immediate row)
ToolInputDeltaEvent  → (appended to tool buffer; no immediate row)
MessageStopEvent     → model_call.completed {"stop_reason": ..., token counts,
                                             "model_call_number": N}
MessageStartEvent    → (no row; model_call.started is written before the call)
MessageDeltaEvent    → (stop_reason captured; no row)

Tool blocks (ToolUseStartEvent + accumulated ToolInputDeltaEvents) are exposed
via the ``tool_blocks`` property after the stream ends.  The caller is
responsible for writing ``tool_call.requested`` events after schema-validation
and hook dispatch.

On event-log append failure the caller receives the exception directly (the
EventLogRuntime already writes to the audit log before raising).
"""

from __future__ import annotations

from typing import Any

from meridian_sdk_provider.types import (
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    ModelEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)


class ModelEventTranslator:
    """Stateful translator for one model-call's event stream.

    Feed each ModelEvent from the stream through :meth:`translate`.  The
    method returns zero or more ``(event_type, data)`` pairs that should be
    appended to the session event log immediately.  Tool blocks are buffered
    internally and exposed via :attr:`tool_blocks` once the stream finishes.

    Parameters
    ----------
    model_call_number:
        1-based index of the current model call within the session.  Included
        in every emitted data dict as ``"model_call_number"``.
    """

    def __init__(self, model_call_number: int = 1) -> None:
        self._model_call_number = model_call_number
        self._tool_blocks: list[dict[str, Any]] = []
        self._current_tool: dict[str, Any] | None = None
        self._stop_reason: str = "end_turn"
        self._start_event: MessageStartEvent | None = None
        self._stop_event: MessageStopEvent | None = None

    # ------------------------------------------------------------------
    # Main translation entry point
    # ------------------------------------------------------------------

    def translate(self, event: ModelEvent) -> list[tuple[str, dict[str, Any]]]:
        """Translate one ModelEvent into zero or more (event_type, data) pairs.

        The returned pairs should be written to the session event log in order.
        Returns an empty list for events that produce no log row (e.g. tool
        input deltas, message start).
        """
        if isinstance(event, MessageStartEvent):
            self._start_event = event
            return []

        if isinstance(event, TextDeltaEvent):
            return [
                (
                    "message.delta",
                    {
                        "kind": "text",
                        "text": event.text,
                        "model_call_number": self._model_call_number,
                    },
                )
            ]

        if isinstance(event, ThinkingDeltaEvent):
            return [
                (
                    "message.delta",
                    {
                        "kind": "thinking",
                        "thinking": event.thinking,
                        "model_call_number": self._model_call_number,
                    },
                )
            ]

        if isinstance(event, ToolUseStartEvent):
            self._current_tool = {"id": event.id, "name": event.name, "input_json": ""}
            self._tool_blocks.append(self._current_tool)
            return []

        if isinstance(event, ToolInputDeltaEvent):
            if self._current_tool is not None and self._current_tool["id"] == event.id:
                self._current_tool["input_json"] += event.partial_json
            return []

        if isinstance(event, MessageDeltaEvent):
            if event.stop_reason is not None:
                self._stop_reason = event.stop_reason
            return []

        if isinstance(event, MessageStopEvent):
            if event.stop_reason is not None:
                self._stop_reason = event.stop_reason
            self._stop_event = event
            return [
                (
                    "model_call.completed",
                    {
                        "stop_reason": self._stop_reason,
                        "input_tokens": event.input_tokens or 0,
                        "output_tokens": event.output_tokens or 0,
                        "cache_creation_tokens": event.cache_creation_input_tokens,
                        "cache_read_tokens": event.cache_read_input_tokens,
                        "model_call_number": self._model_call_number,
                    },
                )
            ]

        return []

    # ------------------------------------------------------------------
    # Accumulated state (read after the stream ends)
    # ------------------------------------------------------------------

    @property
    def tool_blocks(self) -> list[dict[str, Any]]:
        """Tool blocks accumulated from ToolUseStartEvent + ToolInputDeltaEvents.

        Each entry has keys ``id``, ``name``, and ``input_json`` (the raw
        concatenated partial-JSON string from ToolInputDeltaEvents).
        """
        return list(self._tool_blocks)

    @property
    def stop_reason(self) -> str:
        """Stop reason from the latest MessageDeltaEvent or MessageStopEvent."""
        return self._stop_reason

    @property
    def start_event(self) -> MessageStartEvent | None:
        """The MessageStartEvent, if one was received."""
        return self._start_event

    @property
    def stop_event(self) -> MessageStopEvent | None:
        """The MessageStopEvent, if one was received."""
        return self._stop_event
