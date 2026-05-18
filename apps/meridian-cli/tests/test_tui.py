"""Tests for the meridiantui command.

Invariants verified:
  1. invocation — OTel span created with name tui.launch and audit entry written
  2. success    — span.add_event("tui.launch.completed") called on normal exit
  3. failure    — app.run() raising writes error audit entry, prints to stderr, exits 1
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from meridian_cli.__main__ import cli
from meridian_cli._client import DaemonClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(mock_client: MagicMock) -> object:
    runner = CliRunner(mix_stderr=False)
    with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
        return runner.invoke(cli, ["meridiantui"], catch_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> MagicMock:
    return MagicMock(spec=DaemonClient)


@pytest.fixture()
def mock_tracer() -> tuple[MagicMock, MagicMock]:
    """Returns (mock_span, mock_tracer) with the context-manager protocol wired."""
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = lambda *_: span
    tracer.start_as_current_span.return_value.__exit__ = lambda *_: False
    return span, tracer


@pytest.fixture(autouse=True)
def _patch_tui_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress real OTel SDK requirement for every test in this module."""
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = lambda *_: span
    tracer.start_as_current_span.return_value.__exit__ = lambda *_: False
    monkeypatch.setattr("meridian_cli.tui.get_tracer", lambda: tracer)


# ---------------------------------------------------------------------------
# Invocation — audit log written
# ---------------------------------------------------------------------------


def test_meridiantui_invocation_writes_audit(mock_client: MagicMock) -> None:
    with (
        patch("meridian_cli.tui.MeridianTuiApp.run"),
        patch("meridian_cli.tui.write_audit") as mock_audit,
    ):
        result = _invoke(mock_client)

    assert result.exit_code == 0
    mock_audit.assert_any_call("info", "tui.launch.invoked", {"operation": "tui.launch"})


def test_meridiantui_invocation_starts_otel_span(
    mock_client: MagicMock,
    mock_tracer: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _span, tracer = mock_tracer
    monkeypatch.setattr("meridian_cli.tui.get_tracer", lambda: tracer)

    with (
        patch("meridian_cli.tui.MeridianTuiApp.run"),
        patch("meridian_cli.tui.write_audit"),
    ):
        result = _invoke(mock_client)

    assert result.exit_code == 0
    tracer.start_as_current_span.assert_called_once_with(
        "tui.launch", attributes={"operation": "tui.launch"}
    )


# ---------------------------------------------------------------------------
# Success — span completed event
# ---------------------------------------------------------------------------


def test_meridiantui_success_records_completed_event(
    mock_client: MagicMock,
    mock_tracer: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_span, tracer = mock_tracer
    monkeypatch.setattr("meridian_cli.tui.get_tracer", lambda: tracer)

    with (
        patch("meridian_cli.tui.MeridianTuiApp.run"),
        patch("meridian_cli.tui.write_audit"),
    ):
        result = _invoke(mock_client)

    assert result.exit_code == 0
    mock_span.add_event.assert_called_with("tui.launch.completed")


# ---------------------------------------------------------------------------
# Failure — app.run() raises
# ---------------------------------------------------------------------------


def test_meridiantui_failure_exits_nonzero(mock_client: MagicMock) -> None:
    with (
        patch(
            "meridian_cli.tui.MeridianTuiApp.run",
            side_effect=RuntimeError("terminal unavailable"),
        ),
        patch("meridian_cli.tui.write_audit"),
    ):
        result = _invoke(mock_client)

    assert result.exit_code != 0


def test_meridiantui_failure_writes_error_audit(mock_client: MagicMock) -> None:
    with (
        patch(
            "meridian_cli.tui.MeridianTuiApp.run",
            side_effect=RuntimeError("terminal unavailable"),
        ),
        patch("meridian_cli.tui.write_audit") as mock_audit,
    ):
        result = _invoke(mock_client)

    assert result.exit_code != 0
    mock_audit.assert_any_call(
        "error",
        "tui.launch.failed",
        {"code": "tui_launch_failed", "message": "terminal unavailable"},
    )


def test_meridiantui_failure_prints_error_to_stderr(mock_client: MagicMock) -> None:
    with (
        patch(
            "meridian_cli.tui.MeridianTuiApp.run",
            side_effect=RuntimeError("terminal unavailable"),
        ),
        patch("meridian_cli.tui.write_audit"),
    ):
        result = _invoke(mock_client)

    assert result.exit_code != 0
    assert "tui_launch_failed" in result.stderr
    assert "terminal unavailable" in result.stderr


def test_meridiantui_failure_records_otel_failure(
    mock_client: MagicMock,
    mock_tracer: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_span, tracer = mock_tracer
    monkeypatch.setattr("meridian_cli.tui.get_tracer", lambda: tracer)

    with (
        patch(
            "meridian_cli.tui.MeridianTuiApp.run",
            side_effect=RuntimeError("terminal unavailable"),
        ),
        patch("meridian_cli.tui.write_audit"),
    ):
        result = _invoke(mock_client)

    assert result.exit_code != 0
    mock_span.set_status.assert_called_once()
    mock_span.add_event.assert_any_call(
        "meridian.cli.failure",
        {"error.code": "tui_launch_failed", "error.message": "terminal unavailable"},
    )
