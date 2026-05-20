"""Unit tests for ModelEventTranslator — Contract 4 event translation.

Tests cover:
  TextDeltaEvent:
  - translate() returns one ("message.delta", data) pair.
  - data["kind"] == "text".
  - data["text"] carries the event text.
  - data["model_call_number"] matches constructor argument.

  ThinkingDeltaEvent:
  - translate() returns one ("message.delta", data) pair.
  - data["kind"] == "thinking".
  - data["thinking"] carries the event thinking text.
  - data["model_call_number"] matches constructor argument.

  ToolUseStartEvent:
  - translate() returns an empty list (buffered, not emitted).
  - tool_blocks contains one entry with id, name, and empty input_json.

  ToolInputDeltaEvent:
  - translate() returns an empty list.
  - tool block's input_json accumulates partial_json strings.
  - Delta for unknown id does not corrupt existing tool block.

  MessageStopEvent:
  - translate() returns one ("model_call.completed", data) pair.
  - data["stop_reason"] matches the event's stop_reason.
  - data["input_tokens"] and data["output_tokens"] are correct.
  - data["cache_creation_tokens"] and data["cache_read_tokens"] are correct.
  - data["model_call_number"] matches constructor argument.
  - stop_event property is set after a MessageStopEvent.

  MessageDeltaEvent:
  - translate() returns an empty list.
  - stop_reason property is updated from MessageDeltaEvent.
  - MessageDeltaEvent stop_reason is overridden by MessageStopEvent stop_reason.

  MessageStartEvent:
  - translate() returns an empty list.
  - start_event property is set.

  Stop_reason resolution:
  - Default stop_reason is "end_turn".
  - MessageDeltaEvent sets stop_reason.
  - MessageStopEvent sets stop_reason.
  - MessageStopEvent overrides an earlier MessageDeltaEvent value.

  Full stream sequence (end_turn):
  - translate() called on MessageStart → TextDelta → MessageStop yields
    [message.delta, model_call.completed] in order.

  Full stream sequence (tool_use):
  - translate() on ToolUseStart → ToolInputDelta(s) → MessageStop produces
    [model_call.completed] with no tool_call.requested rows (caller's responsibility).
  - tool_blocks contains the fully assembled input_json after the stream.

  model_call_number:
  - Default is 1.
  - Emitted in every message.delta and model_call.completed data dict.
"""

from __future__ import annotations

from meridian_sdk_provider.types import (
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)

from meridiand._event_translator import ModelEventTranslator


# ---------------------------------------------------------------------------
# TextDeltaEvent
# ---------------------------------------------------------------------------


class TestTextDeltaTranslation:
    def test_text_delta_emits_message_delta(self) -> None:
        t = ModelEventTranslator()
        pairs = t.translate(TextDeltaEvent(text="hello"))
        assert len(pairs) == 1
        assert pairs[0][0] == "message.delta"

    def test_text_delta_kind_is_text(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(TextDeltaEvent(text="hello"))[0]
        assert data["kind"] == "text"

    def test_text_delta_carries_text(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(TextDeltaEvent(text="world"))[0]
        assert data["text"] == "world"

    def test_text_delta_carries_model_call_number(self) -> None:
        t = ModelEventTranslator(model_call_number=3)
        _, data = t.translate(TextDeltaEvent(text="x"))[0]
        assert data["model_call_number"] == 3


# ---------------------------------------------------------------------------
# ThinkingDeltaEvent
# ---------------------------------------------------------------------------


class TestThinkingDeltaTranslation:
    def test_thinking_delta_emits_message_delta(self) -> None:
        t = ModelEventTranslator()
        pairs = t.translate(ThinkingDeltaEvent(thinking="hmm"))
        assert len(pairs) == 1
        assert pairs[0][0] == "message.delta"

    def test_thinking_delta_kind_is_thinking(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(ThinkingDeltaEvent(thinking="hmm"))[0]
        assert data["kind"] == "thinking"

    def test_thinking_delta_carries_thinking_text(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(ThinkingDeltaEvent(thinking="deep thought"))[0]
        assert data["thinking"] == "deep thought"

    def test_thinking_delta_carries_model_call_number(self) -> None:
        t = ModelEventTranslator(model_call_number=2)
        _, data = t.translate(ThinkingDeltaEvent(thinking="x"))[0]
        assert data["model_call_number"] == 2

    def test_thinking_delta_no_text_key(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(ThinkingDeltaEvent(thinking="x"))[0]
        assert "text" not in data

    def test_text_delta_no_thinking_key(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(TextDeltaEvent(text="x"))[0]
        assert "thinking" not in data


# ---------------------------------------------------------------------------
# ToolUseStartEvent
# ---------------------------------------------------------------------------


class TestToolUseStartTranslation:
    def test_tool_use_start_emits_no_rows(self) -> None:
        t = ModelEventTranslator()
        pairs = t.translate(ToolUseStartEvent(id="tu_1", name="bash"))
        assert pairs == []

    def test_tool_use_start_adds_to_tool_blocks(self) -> None:
        t = ModelEventTranslator()
        t.translate(ToolUseStartEvent(id="tu_1", name="bash"))
        assert len(t.tool_blocks) == 1

    def test_tool_block_has_correct_id(self) -> None:
        t = ModelEventTranslator()
        t.translate(ToolUseStartEvent(id="tu_abc", name="bash"))
        assert t.tool_blocks[0]["id"] == "tu_abc"

    def test_tool_block_has_correct_name(self) -> None:
        t = ModelEventTranslator()
        t.translate(ToolUseStartEvent(id="tu_1", name="read_file"))
        assert t.tool_blocks[0]["name"] == "read_file"

    def test_tool_block_starts_with_empty_input_json(self) -> None:
        t = ModelEventTranslator()
        t.translate(ToolUseStartEvent(id="tu_1", name="bash"))
        assert t.tool_blocks[0]["input_json"] == ""

    def test_multiple_tool_starts_accumulate_separately(self) -> None:
        t = ModelEventTranslator()
        t.translate(ToolUseStartEvent(id="tu_1", name="bash"))
        t.translate(ToolUseStartEvent(id="tu_2", name="read_file"))
        assert len(t.tool_blocks) == 2
        assert t.tool_blocks[0]["id"] == "tu_1"
        assert t.tool_blocks[1]["id"] == "tu_2"


# ---------------------------------------------------------------------------
# ToolInputDeltaEvent
# ---------------------------------------------------------------------------


class TestToolInputDeltaTranslation:
    def test_tool_input_delta_emits_no_rows(self) -> None:
        t = ModelEventTranslator()
        t.translate(ToolUseStartEvent(id="tu_1", name="bash"))
        pairs = t.translate(ToolInputDeltaEvent(id="tu_1", partial_json='{"cmd"'))
        assert pairs == []

    def test_tool_input_delta_accumulates_json(self) -> None:
        t = ModelEventTranslator()
        t.translate(ToolUseStartEvent(id="tu_1", name="bash"))
        t.translate(ToolInputDeltaEvent(id="tu_1", partial_json='{"cmd"'))
        t.translate(ToolInputDeltaEvent(id="tu_1", partial_json=':"ls"}'))
        assert t.tool_blocks[0]["input_json"] == '{"cmd":"ls"}'

    def test_tool_input_delta_unknown_id_ignored(self) -> None:
        t = ModelEventTranslator()
        t.translate(ToolUseStartEvent(id="tu_1", name="bash"))
        t.translate(ToolInputDeltaEvent(id="tu_999", partial_json="noise"))
        assert t.tool_blocks[0]["input_json"] == ""

    def test_tool_input_delta_without_prior_start_ignored(self) -> None:
        t = ModelEventTranslator()
        # No ToolUseStartEvent; should not raise
        pairs = t.translate(ToolInputDeltaEvent(id="tu_1", partial_json='{}'))
        assert pairs == []


# ---------------------------------------------------------------------------
# MessageStopEvent
# ---------------------------------------------------------------------------


class TestMessageStopTranslation:
    def test_message_stop_emits_model_call_completed(self) -> None:
        t = ModelEventTranslator()
        pairs = t.translate(MessageStopEvent(stop_reason="end_turn", input_tokens=10, output_tokens=5))
        assert len(pairs) == 1
        assert pairs[0][0] == "model_call.completed"

    def test_message_stop_data_has_stop_reason(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(MessageStopEvent(stop_reason="end_turn"))[0]
        assert data["stop_reason"] == "end_turn"

    def test_message_stop_data_has_input_tokens(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(MessageStopEvent(stop_reason="end_turn", input_tokens=42))[0]
        assert data["input_tokens"] == 42

    def test_message_stop_data_has_output_tokens(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(MessageStopEvent(stop_reason="end_turn", output_tokens=17))[0]
        assert data["output_tokens"] == 17

    def test_message_stop_null_tokens_default_to_zero(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(MessageStopEvent(stop_reason="end_turn"))[0]
        assert data["input_tokens"] == 0
        assert data["output_tokens"] == 0

    def test_message_stop_data_has_cache_creation_tokens(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(
            MessageStopEvent(stop_reason="end_turn", cache_creation_input_tokens=8)
        )[0]
        assert data["cache_creation_tokens"] == 8

    def test_message_stop_data_has_cache_read_tokens(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(
            MessageStopEvent(stop_reason="end_turn", cache_read_input_tokens=3)
        )[0]
        assert data["cache_read_tokens"] == 3

    def test_message_stop_carries_model_call_number(self) -> None:
        t = ModelEventTranslator(model_call_number=4)
        _, data = t.translate(MessageStopEvent(stop_reason="end_turn"))[0]
        assert data["model_call_number"] == 4

    def test_stop_event_property_set_after_message_stop(self) -> None:
        t = ModelEventTranslator()
        ev = MessageStopEvent(stop_reason="end_turn", input_tokens=1, output_tokens=2)
        t.translate(ev)
        assert t.stop_event is ev


# ---------------------------------------------------------------------------
# MessageDeltaEvent
# ---------------------------------------------------------------------------


class TestMessageDeltaTranslation:
    def test_message_delta_emits_no_rows(self) -> None:
        t = ModelEventTranslator()
        pairs = t.translate(MessageDeltaEvent(stop_reason="tool_use"))
        assert pairs == []

    def test_message_delta_updates_stop_reason(self) -> None:
        t = ModelEventTranslator()
        t.translate(MessageDeltaEvent(stop_reason="tool_use"))
        assert t.stop_reason == "tool_use"

    def test_message_stop_overrides_delta_stop_reason(self) -> None:
        t = ModelEventTranslator()
        t.translate(MessageDeltaEvent(stop_reason="tool_use"))
        t.translate(MessageStopEvent(stop_reason="end_turn"))
        assert t.stop_reason == "end_turn"

    def test_message_delta_none_stop_reason_ignored(self) -> None:
        t = ModelEventTranslator()
        t.translate(MessageDeltaEvent(stop_reason=None))
        assert t.stop_reason == "end_turn"  # default unchanged


# ---------------------------------------------------------------------------
# MessageStartEvent
# ---------------------------------------------------------------------------


class TestMessageStartTranslation:
    def test_message_start_emits_no_rows(self) -> None:
        t = ModelEventTranslator()
        pairs = t.translate(
            MessageStartEvent(type="message_start", model="claude-sonnet-4-6", provider="test")
        )
        assert pairs == []

    def test_start_event_property_set(self) -> None:
        t = ModelEventTranslator()
        ev = MessageStartEvent(type="message_start", model="claude-sonnet-4-6", provider="test")
        t.translate(ev)
        assert t.start_event is ev


# ---------------------------------------------------------------------------
# Stop-reason resolution
# ---------------------------------------------------------------------------


class TestStopReasonResolution:
    def test_default_stop_reason_is_end_turn(self) -> None:
        t = ModelEventTranslator()
        assert t.stop_reason == "end_turn"

    def test_stop_reason_from_message_delta(self) -> None:
        t = ModelEventTranslator()
        t.translate(MessageDeltaEvent(stop_reason="max_tokens"))
        assert t.stop_reason == "max_tokens"

    def test_stop_reason_from_message_stop(self) -> None:
        t = ModelEventTranslator()
        t.translate(MessageStopEvent(stop_reason="tool_use"))
        assert t.stop_reason == "tool_use"

    def test_message_stop_stop_reason_reflected_in_completed_event(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(MessageStopEvent(stop_reason="tool_use"))[0]
        assert data["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# Full stream: end_turn
# ---------------------------------------------------------------------------


class TestFullStreamEndTurn:
    def _feed_stream(self, translator: ModelEventTranslator) -> list[tuple[str, dict]]:
        events = [
            MessageStartEvent(type="message_start", model="claude-sonnet-4-6", provider="test"),
            TextDeltaEvent(text="Hello"),
            TextDeltaEvent(text=", world"),
            MessageStopEvent(stop_reason="end_turn", input_tokens=10, output_tokens=5),
        ]
        result = []
        for ev in events:
            result.extend(translator.translate(ev))
        return result

    def test_end_turn_stream_produces_message_deltas_and_completed(self) -> None:
        t = ModelEventTranslator()
        pairs = self._feed_stream(t)
        types = [p[0] for p in pairs]
        assert types == ["message.delta", "message.delta", "model_call.completed"]

    def test_end_turn_stream_text_correct(self) -> None:
        t = ModelEventTranslator()
        pairs = self._feed_stream(t)
        texts = [p[1]["text"] for p in pairs if p[0] == "message.delta"]
        assert texts == ["Hello", ", world"]

    def test_end_turn_stream_completed_has_stop_reason(self) -> None:
        t = ModelEventTranslator()
        pairs = self._feed_stream(t)
        completed = [p[1] for p in pairs if p[0] == "model_call.completed"]
        assert completed[0]["stop_reason"] == "end_turn"

    def test_end_turn_stream_completed_has_token_counts(self) -> None:
        t = ModelEventTranslator()
        pairs = self._feed_stream(t)
        completed = [p[1] for p in pairs if p[0] == "model_call.completed"][0]
        assert completed["input_tokens"] == 10
        assert completed["output_tokens"] == 5


# ---------------------------------------------------------------------------
# Full stream: tool_use
# ---------------------------------------------------------------------------


class TestFullStreamToolUse:
    def _feed_tool_stream(self, translator: ModelEventTranslator) -> list[tuple[str, dict]]:
        events = [
            MessageStartEvent(type="message_start", model="claude-sonnet-4-6", provider="test"),
            ToolUseStartEvent(id="tu_1", name="bash"),
            ToolInputDeltaEvent(id="tu_1", partial_json='{"cmd"'),
            ToolInputDeltaEvent(id="tu_1", partial_json=':"ls"}'),
            MessageStopEvent(stop_reason="tool_use"),
        ]
        result = []
        for ev in events:
            result.extend(translator.translate(ev))
        return result

    def test_tool_use_stream_emits_only_model_call_completed(self) -> None:
        t = ModelEventTranslator()
        pairs = self._feed_tool_stream(t)
        types = [p[0] for p in pairs]
        assert types == ["model_call.completed"]

    def test_tool_use_stream_completed_stop_reason(self) -> None:
        t = ModelEventTranslator()
        pairs = self._feed_tool_stream(t)
        assert pairs[0][1]["stop_reason"] == "tool_use"

    def test_tool_blocks_assembled_after_stream(self) -> None:
        t = ModelEventTranslator()
        self._feed_tool_stream(t)
        assert len(t.tool_blocks) == 1
        block = t.tool_blocks[0]
        assert block["id"] == "tu_1"
        assert block["name"] == "bash"
        assert block["input_json"] == '{"cmd":"ls"}'


# ---------------------------------------------------------------------------
# Full stream: thinking + text (mixed)
# ---------------------------------------------------------------------------


class TestFullStreamWithThinking:
    def test_thinking_and_text_deltas_in_order(self) -> None:
        t = ModelEventTranslator()
        events = [
            ThinkingDeltaEvent(thinking="let me think"),
            TextDeltaEvent(text="answer"),
            MessageStopEvent(stop_reason="end_turn"),
        ]
        pairs: list[tuple[str, dict]] = []
        for ev in events:
            pairs.extend(t.translate(ev))
        types = [p[0] for p in pairs]
        assert types == ["message.delta", "message.delta", "model_call.completed"]

    def test_thinking_delta_kind_in_mixed_stream(self) -> None:
        t = ModelEventTranslator()
        t.translate(ThinkingDeltaEvent(thinking="thought"))
        t.translate(TextDeltaEvent(text="reply"))
        pairs: list[tuple[str, dict]] = []
        for ev in [ThinkingDeltaEvent(thinking="thought"), TextDeltaEvent(text="reply")]:
            pairs.extend(t.translate(ev))
        kinds = [p[1]["kind"] for p in pairs if p[0] == "message.delta"]
        # order: thinking first, then text (from the second round of translate calls)
        assert "thinking" in kinds
        assert "text" in kinds


# ---------------------------------------------------------------------------
# model_call_number propagation
# ---------------------------------------------------------------------------


class TestModelCallNumber:
    def test_default_model_call_number_is_one(self) -> None:
        t = ModelEventTranslator()
        _, data = t.translate(TextDeltaEvent(text="x"))[0]
        assert data["model_call_number"] == 1

    def test_model_call_number_in_thinking_delta(self) -> None:
        t = ModelEventTranslator(model_call_number=5)
        _, data = t.translate(ThinkingDeltaEvent(thinking="x"))[0]
        assert data["model_call_number"] == 5

    def test_model_call_number_in_model_call_completed(self) -> None:
        t = ModelEventTranslator(model_call_number=7)
        _, data = t.translate(MessageStopEvent(stop_reason="end_turn"))[0]
        assert data["model_call_number"] == 7
