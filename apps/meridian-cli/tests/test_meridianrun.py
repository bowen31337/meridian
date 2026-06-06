"""Tests for the meridianrun command.

Invariants verified:
  1. invocation — OTel span created with name run.stream and audit entry written
  2. success    — span.add_event("run.stream.completed") called after normal exit
  3. --no-follow — exits after first poll without looping
  4. terminal phase — exits when session.phase_change reaches a terminal phase
  5. failure    — DaemonError writes error audit entry, prints to stderr, exits 1
  6. renderer   — phase_change, message.delta, tool_call.* rendered to stdout
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from meridian_cli.__main__ import cli
from meridian_cli._client import DaemonClient, DaemonError
from meridian_cli.meridianrun import _Renderer
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(mock_client: MagicMock, args: list[str]) -> object:
    runner = CliRunner()
    with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
        return runner.invoke(cli, ["meridianrun"] + args, catch_exceptions=False)


def _make_event(etype: str, data: dict, seq: int = 0) -> dict:
    return {"seq": seq, "ts": "2026-05-18T00:00:00Z", "type": etype, "data": data}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> MagicMock:
    return MagicMock(spec=DaemonClient)


@pytest.fixture()
def mock_tracer() -> tuple[MagicMock, MagicMock]:
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = lambda *_: span
    tracer.start_as_current_span.return_value.__exit__ = lambda *_: False
    return span, tracer


@pytest.fixture(autouse=True)
def _patch_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = lambda *_: span
    tracer.start_as_current_span.return_value.__exit__ = lambda *_: False
    monkeypatch.setattr("meridian_cli.meridianrun.get_tracer", lambda: tracer)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("meridian_cli.meridianrun.time.sleep", lambda _: None)


# ---------------------------------------------------------------------------
# 1. Invocation
# ---------------------------------------------------------------------------


def test_invocation_writes_audit(mock_client: MagicMock) -> None:
    done_event = _make_event("session.phase_change", {"prev_phase": "running", "phase": "done"})
    mock_client.request.return_value = [done_event]

    with patch("meridian_cli.meridianrun.write_audit") as mock_audit:
        result = _invoke(mock_client, ["--no-follow", "sess-1"])

    assert result.exit_code == 0
    mock_audit.assert_any_call(
        "info",
        "run.stream.invoked",
        {"operation": "run.stream", "session_id": "sess-1"},
    )


def test_invocation_starts_otel_span(
    mock_client: MagicMock,
    mock_tracer: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _span, tracer = mock_tracer
    monkeypatch.setattr("meridian_cli.meridianrun.get_tracer", lambda: tracer)
    mock_client.request.return_value = []

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["--no-follow", "sess-1"])

    assert result.exit_code == 0
    tracer.start_as_current_span.assert_called_once_with(
        "run.stream",
        attributes={"operation": "run.stream", "session.id": "sess-1"},
    )


# ---------------------------------------------------------------------------
# 2. Success
# ---------------------------------------------------------------------------


def test_success_records_completed_event(
    mock_client: MagicMock,
    mock_tracer: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_span, tracer = mock_tracer
    monkeypatch.setattr("meridian_cli.meridianrun.get_tracer", lambda: tracer)
    mock_client.request.return_value = []

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["--no-follow", "sess-1"])

    assert result.exit_code == 0
    mock_span.add_event.assert_called_with("run.stream.completed")


# ---------------------------------------------------------------------------
# 3. --no-follow exits after one poll
# ---------------------------------------------------------------------------


def test_no_follow_polls_once(mock_client: MagicMock) -> None:
    mock_client.request.return_value = []

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["--no-follow", "sess-1"])

    assert result.exit_code == 0
    mock_client.request.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Terminal phase stops the loop
# ---------------------------------------------------------------------------


def test_follow_stops_on_done_phase(mock_client: MagicMock) -> None:
    done_event = _make_event("session.phase_change", {"prev_phase": "running", "phase": "done"})
    # First call returns the terminal event; a second call would mean we didn't stop
    mock_client.request.side_effect = [
        [done_event],
        [],  # should never be reached
    ]

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code == 0
    assert mock_client.request.call_count == 1


def test_follow_stops_on_cancelled_phase(mock_client: MagicMock) -> None:
    cancelled = _make_event("session.phase_change", {"prev_phase": "running", "phase": "cancelled"})
    mock_client.request.side_effect = [[cancelled], []]

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code == 0
    assert mock_client.request.call_count == 1


def test_follow_stops_on_error_phase(mock_client: MagicMock) -> None:
    err_phase = _make_event("session.phase_change", {"prev_phase": "running", "phase": "error"})
    mock_client.request.side_effect = [[err_phase], []]

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code == 0
    assert mock_client.request.call_count == 1


def test_follow_continues_on_non_terminal_phase(mock_client: MagicMock) -> None:
    running = _make_event("session.phase_change", {"prev_phase": "idle", "phase": "running"})
    done = _make_event("session.phase_change", {"prev_phase": "running", "phase": "done"}, seq=1)
    mock_client.request.side_effect = [[running], [done]]

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code == 0
    assert mock_client.request.call_count == 2


# ---------------------------------------------------------------------------
# 5. DaemonError → failure path
# ---------------------------------------------------------------------------


def test_daemon_error_exits_nonzero(mock_client: MagicMock) -> None:
    mock_client.request.side_effect = DaemonError("daemon_unreachable", "connection refused")

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code != 0


def test_daemon_error_writes_error_audit(mock_client: MagicMock) -> None:
    mock_client.request.side_effect = DaemonError("daemon_unreachable", "connection refused")

    with patch("meridian_cli.meridianrun.write_audit") as mock_audit:
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code != 0
    mock_audit.assert_any_call(
        "error",
        "run.stream.failed",
        {
            "code": "daemon_unreachable",
            "message": "connection refused",
            "session_id": "sess-1",
        },
    )


def test_daemon_error_prints_to_stderr(mock_client: MagicMock) -> None:
    mock_client.request.side_effect = DaemonError("daemon_unreachable", "connection refused")

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code != 0
    assert "daemon_unreachable" in result.stderr
    assert "connection refused" in result.stderr


def test_daemon_error_records_otel_failure(
    mock_client: MagicMock,
    mock_tracer: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_span, tracer = mock_tracer
    monkeypatch.setattr("meridian_cli.meridianrun.get_tracer", lambda: tracer)
    mock_client.request.side_effect = DaemonError("daemon_unreachable", "connection refused")

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code != 0
    mock_span.set_status.assert_called_once()
    mock_span.add_event.assert_any_call(
        "meridian.cli.failure",
        {"error.code": "daemon_unreachable", "error.message": "connection refused"},
    )


# ---------------------------------------------------------------------------
# 6. Renderer unit tests
# ---------------------------------------------------------------------------


class _FakeConsole:
    """Capture rich console output for assertions."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._pending = ""

    def print(self, *args: object, end: str = "\n") -> None:
        text = " ".join(str(a) for a in args)
        self._pending += text
        if end == "\n":
            self.lines.append(self._pending)
            self._pending = ""

    def flush(self) -> None:
        if self._pending:
            self.lines.append(self._pending)
            self._pending = ""


@pytest.fixture()
def fake_console() -> _FakeConsole:
    return _FakeConsole()


@pytest.fixture()
def renderer(fake_console: _FakeConsole) -> _Renderer:
    return _Renderer(fake_console)  # type: ignore[arg-type]


def test_renderer_phase_change(renderer: _Renderer, fake_console: _FakeConsole) -> None:
    renderer.render(_make_event("session.phase_change", {"prev_phase": "idle", "phase": "running"}))
    assert any("running" in line for line in fake_console.lines)
    assert any("idle" in line for line in fake_console.lines)


def test_renderer_message_delta_text(renderer: _Renderer, fake_console: _FakeConsole) -> None:
    renderer.render(
        _make_event("message.delta", {"delta": {"type": "text_delta", "text": "Hello"}})
    )
    renderer.close()
    combined = " ".join(fake_console.lines)
    assert "Hello" in combined
    assert "Assistant" in combined


def test_renderer_thinking_collapsed(renderer: _Renderer, fake_console: _FakeConsole) -> None:
    renderer.render(
        _make_event(
            "message.delta",
            {"content": [{"type": "thinking", "thinking": "secret thoughts"}]},
        )
    )
    combined = " ".join(fake_console.lines)
    assert "<thinking>" in combined
    assert "secret thoughts" not in combined


def test_renderer_tool_call_requested(renderer: _Renderer, fake_console: _FakeConsole) -> None:
    renderer.render(
        _make_event(
            "tool_call.requested",
            {"tool_name": "bash", "arguments": {"command": "ls"}},
        )
    )
    combined = " ".join(fake_console.lines)
    assert "bash" in combined
    assert "⟳" in combined


def test_renderer_tool_call_result(renderer: _Renderer, fake_console: _FakeConsole) -> None:
    renderer.render(_make_event("tool_call.result", {"result": "file1.txt\nfile2.txt"}))
    combined = " ".join(fake_console.lines)
    assert "✓" in combined
    assert "file1.txt" in combined


def test_renderer_tool_call_error(renderer: _Renderer, fake_console: _FakeConsole) -> None:
    renderer.render(_make_event("tool_call.error", {"error": "permission denied"}))
    combined = " ".join(fake_console.lines)
    assert "✗" in combined
    assert "permission denied" in combined


def test_renderer_budget_warning(renderer: _Renderer, fake_console: _FakeConsole) -> None:
    renderer.render(_make_event("budget.warning", {"message": "80% used"}))
    combined = " ".join(fake_console.lines)
    assert "⚠" in combined
    assert "80% used" in combined


def test_renderer_budget_exceeded(renderer: _Renderer, fake_console: _FakeConsole) -> None:
    renderer.render(_make_event("budget.exceeded", {"message": "limit hit"}))
    combined = " ".join(fake_console.lines)
    assert "⊘" in combined


def test_renderer_error_event(renderer: _Renderer, fake_console: _FakeConsole) -> None:
    renderer.render(_make_event("error", {"message": "something broke"}))
    combined = " ".join(fake_console.lines)
    assert "something broke" in combined


def test_renderer_flush_ends_streaming_line(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(
        _make_event("message.delta", {"delta": {"type": "text_delta", "text": "partial"}})
    )
    # Before flush, no newline yet
    assert fake_console._pending != "" or any("partial" in line for line in fake_console.lines)
    renderer.close()
    # After close, pending is flushed
    combined = " ".join(fake_console.lines)
    assert "partial" in combined


def test_renderer_seq_tracking_via_events_endpoint(mock_client: MagicMock) -> None:
    """The since= query param advances as events are consumed."""
    evt0 = _make_event("session.created", {}, seq=0)
    evt1 = _make_event("session.phase_change", {"prev_phase": "idle", "phase": "done"}, seq=1)
    mock_client.request.return_value = [evt0, evt1]

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code == 0
    first_positional_args = mock_client.request.call_args_list[0].args
    assert any("since=-1" in str(a) for a in first_positional_args)


# ---------------------------------------------------------------------------
# Additional renderer branches
# ---------------------------------------------------------------------------


def test_renderer_message_added_user_role(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(
        _make_event("message.added", {"role": "user", "content": "hi there"})
    )
    combined = " ".join(fake_console.lines)
    assert "hi there" in combined
    assert "User" in combined


def test_renderer_message_added_non_user_silent(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(
        _make_event("message.added", {"role": "assistant", "content": "X"})
    )
    assert fake_console.lines == []


def test_renderer_tool_call_args_dict_truncated(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(
        _make_event(
            "tool_call.requested",
            {
                "tool_name": "fn",
                "arguments": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            },
        )
    )
    combined = " ".join(fake_console.lines)
    assert "fn" in combined
    assert "…" in combined  # truncation marker


def test_renderer_tool_call_args_non_dict(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(
        _make_event(
            "tool_call.requested",
            {"name": "fn2", "args": "raw-string-args"},
        )
    )
    combined = " ".join(fake_console.lines)
    assert "raw-string-args" in combined


def test_renderer_tool_call_dispatched(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(_make_event("tool_call.dispatched", {}))
    combined = " ".join(fake_console.lines)
    assert "dispatching" in combined


def test_renderer_tool_call_result_non_str(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(_make_event("tool_call.result", {"output": {"x": 1, "y": 2}}))
    combined = " ".join(fake_console.lines)
    assert "{" in combined and "x" in combined


def test_renderer_model_call_started(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(_make_event("model_call.started", {}))
    combined = " ".join(fake_console.lines)
    assert "model call" in combined


def test_renderer_usage_delta_with_tokens(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(_make_event("usage.delta", {"prompt_tokens": 10, "completion_tokens": 5}))
    combined = " ".join(fake_console.lines)
    assert "+10p" in combined and "+5c" in combined


def test_renderer_usage_delta_zero_tokens_silent(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(_make_event("usage.delta", {"prompt_tokens": 0, "completion_tokens": 0}))
    assert fake_console.lines == []


def test_renderer_checkpoint_created(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(_make_event("checkpoint.created", {}))
    combined = " ".join(fake_console.lines)
    assert "checkpoint.created" in combined


def test_renderer_render_delta_via_data_content(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """delta is None but data.content has blocks."""
    renderer.render(
        _make_event(
            "message.delta",
            {"content": [{"type": "text", "text": "from-data-content"}]},
        )
    )
    renderer.close()
    combined = " ".join(fake_console.lines)
    assert "from-data-content" in combined


def test_renderer_render_delta_text_delta_block(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    renderer.render(
        _make_event(
            "message.delta",
            {"delta": {"content": [{"type": "text_delta", "delta": "fragment"}]}},
        )
    )
    renderer.close()
    combined = " ".join(fake_console.lines)
    assert "fragment" in combined


def test_renderer_render_delta_thinking_already_in_thinking(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """Two consecutive thinking blocks — the second doesn't re-emit the marker."""
    renderer.render(
        _make_event(
            "message.delta",
            {"content": [{"type": "thinking", "thinking": "first"}]},
        )
    )
    renderer.render(
        _make_event(
            "message.delta",
            {"content": [{"type": "thinking", "thinking": "second"}]},
        )
    )
    combined = " ".join(fake_console.lines)
    # Only one <thinking> marker even though two events
    assert combined.count("<thinking>") == 1


def test_renderer_render_delta_thinking_after_streaming(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """thinking block after streaming text emits the newline-break path."""
    renderer.render(
        _make_event("message.delta", {"delta": {"type": "text_delta", "text": "hi"}})
    )
    renderer.render(
        _make_event(
            "message.delta",
            {"content": [{"type": "thinking", "thinking": "x"}]},
        )
    )
    renderer.close()
    combined = " ".join(fake_console.lines)
    assert "<thinking>" in combined


def test_renderer_render_delta_flat_content_string(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """Flat string content fallback path (no delta dict)."""
    renderer.render(_make_event("message.delta", {"content": "literal"}))
    renderer.close()
    combined = " ".join(fake_console.lines)
    assert "literal" in combined


def test_renderer_seq_not_advanced_when_lower(mock_client: MagicMock) -> None:
    """Events with seq <= last_seq don't advance last_seq."""
    e_high = _make_event("session.created", {}, seq=10)
    done = _make_event(
        "session.phase_change", {"prev_phase": "running", "phase": "done"}, seq=11
    )
    e_low = _make_event("usage.delta", {"prompt_tokens": 1}, seq=5)
    mock_client.request.return_value = [e_high, e_low, done]
    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["--no-follow", "s"])
    assert result.exit_code == 0


def test_keyboard_interrupt_during_loop(mock_client: MagicMock) -> None:
    """KeyboardInterrupt closes renderer gracefully."""
    mock_client.request.side_effect = KeyboardInterrupt
    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])
    assert result.exit_code == 0


def test_renderer_silent_event_falls_through(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """Events not matched by any branch (e.g. hook.invoked) emit nothing."""
    renderer.render(_make_event("hook.invoked", {}))
    assert fake_console.lines == []


def test_renderer_delta_text_empty_returns_early(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """delta.type is text with empty text — emits nothing but returns."""
    renderer.render(_make_event("message.delta", {"delta": {"type": "text", "text": ""}}))
    assert fake_console.lines == []


def test_renderer_delta_text_block_empty_loops(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """content_blocks with text block of empty text — no emit but loop continues."""
    renderer.render(
        _make_event(
            "message.delta",
            {"content": [{"type": "text", "text": ""}, {"type": "text", "text": "real"}]},
        )
    )
    renderer.close()
    combined = " ".join(fake_console.lines)
    assert "real" in combined


def test_renderer_delta_flat_with_dict_text(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """delta is dict, type not text — falls through to fallback which finds delta.text."""
    renderer.render(
        _make_event("message.delta", {"delta": {"type": "other", "text": "in-delta"}})
    )
    renderer.close()
    combined = " ".join(fake_console.lines)
    assert "in-delta" in combined


def test_renderer_delta_no_text_anywhere_silent(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """No text in delta or content — emit_text never called."""
    renderer.render(_make_event("message.delta", {"delta": {"type": "other"}, "content": 123}))
    assert fake_console.lines == []


def test_renderer_unknown_block_type_silently_skipped(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """A content block with unknown type (e.g. 'tool_use') is silently skipped."""
    renderer.render(
        _make_event(
            "message.delta",
            {"content": [{"type": "tool_use", "id": "x"}, {"type": "text", "text": "hi"}]},
        )
    )
    renderer.close()
    combined = " ".join(fake_console.lines)
    assert "hi" in combined


def test_renderer_two_text_blocks_both_with_text(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """Two text content blocks both populated — loop iterates after first emit."""
    renderer.render(
        _make_event(
            "message.delta",
            {"content": [{"type": "text", "text": "one "}, {"type": "text", "text": "two"}]},
        )
    )
    renderer.close()
    combined = " ".join(fake_console.lines)
    assert "one" in combined and "two" in combined


def test_renderer_emit_text_while_already_streaming(
    renderer: _Renderer, fake_console: _FakeConsole
) -> None:
    """Two consecutive text emissions — second skips the 'Assistant:' prelude."""
    renderer.render(_make_event("message.delta", {"delta": {"type": "text_delta", "text": "a"}}))
    renderer.render(_make_event("message.delta", {"delta": {"type": "text_delta", "text": "b"}}))
    renderer.close()
    combined = " ".join(fake_console.lines)
    assert "a" in combined and "b" in combined
    assert combined.count("Assistant") == 1
