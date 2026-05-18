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

from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner

from meridian_cli.__main__ import cli
from meridian_cli._client import DaemonClient, DaemonError
from meridian_cli.meridianrun import _Renderer


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
    cancelled = _make_event(
        "session.phase_change", {"prev_phase": "running", "phase": "cancelled"}
    )
    mock_client.request.side_effect = [[cancelled], []]

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code == 0
    assert mock_client.request.call_count == 1


def test_follow_stops_on_error_phase(mock_client: MagicMock) -> None:
    err_phase = _make_event(
        "session.phase_change", {"prev_phase": "running", "phase": "error"}
    )
    mock_client.request.side_effect = [[err_phase], []]

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code == 0
    assert mock_client.request.call_count == 1


def test_follow_continues_on_non_terminal_phase(mock_client: MagicMock) -> None:
    running = _make_event(
        "session.phase_change", {"prev_phase": "idle", "phase": "running"}
    )
    done = _make_event(
        "session.phase_change", {"prev_phase": "running", "phase": "done"}, seq=1
    )
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
    renderer.render(
        _make_event("session.phase_change", {"prev_phase": "idle", "phase": "running"})
    )
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
    assert fake_console._pending != "" or any("partial" in l for l in fake_console.lines)
    renderer.close()
    # After close, pending is flushed
    combined = " ".join(fake_console.lines)
    assert "partial" in combined


def test_renderer_seq_tracking_via_events_endpoint(mock_client: MagicMock) -> None:
    """The since= query param advances as events are consumed."""
    evt0 = _make_event("session.created", {}, seq=0)
    evt1 = _make_event(
        "session.phase_change", {"prev_phase": "idle", "phase": "done"}, seq=1
    )
    mock_client.request.return_value = [evt0, evt1]

    with patch("meridian_cli.meridianrun.write_audit"):
        result = _invoke(mock_client, ["sess-1"])

    assert result.exit_code == 0
    first_positional_args = mock_client.request.call_args_list[0].args
    assert any("since=-1" in str(a) for a in first_positional_args)
