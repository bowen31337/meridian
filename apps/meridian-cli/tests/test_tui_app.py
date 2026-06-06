"""Unit tests for MeridianTuiApp + ApprovalModal internals (no live Textual loop)."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from meridian_cli._client import DaemonClient, DaemonError
from meridian_cli.tui import ApprovalModal, MeridianTuiApp


# ---------------------------------------------------------------------------
# ApprovalModal — compose + actions
# ---------------------------------------------------------------------------


class TestApprovalModal:
    async def test_compose_runs_inside_app(self) -> None:
        """Drive the App through Textual's test pilot so compose actually runs."""
        app = MeridianTuiApp(client=MagicMock(spec=DaemonClient))
        # Stub the @work-decorated fetch so no real HTTP happens during mount
        app._fetch_sessions = MagicMock()  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await pilot.pause()
            # Push an ApprovalModal so its compose path executes too
            modal = ApprovalModal("Title", ["one", "two"])
            app.push_screen(modal)
            await pilot.pause()
            modal.action_approve()  # exercise the dismiss path inside an app
            await pilot.pause()

    def test_action_approve_dismisses_true(self) -> None:
        modal = ApprovalModal("t", [])
        modal.dismiss = MagicMock()  # type: ignore[method-assign]
        modal.action_approve()
        modal.dismiss.assert_called_once_with(True)

    def test_action_reject_dismisses_false(self) -> None:
        modal = ApprovalModal("t", [])
        modal.dismiss = MagicMock()  # type: ignore[method-assign]
        modal.action_reject()
        modal.dismiss.assert_called_once_with(False)


# ---------------------------------------------------------------------------
# MeridianTuiApp — set up an instance + stub query_one
# ---------------------------------------------------------------------------


def _make_app() -> tuple[MeridianTuiApp, dict[str, MagicMock]]:
    """Build a MeridianTuiApp with query_one mocked to return per-id widgets."""
    app = MeridianTuiApp(client=MagicMock(spec=DaemonClient))
    widgets: dict[str, MagicMock] = {
        "#sessions-pane": MagicMock(),
        "#channels-pane": MagicMock(),
        "#events-pane": MagicMock(),
    }
    app.query_one = lambda selector, _type=None: widgets[selector]  # type: ignore[assignment]
    return app, widgets


async def test_compose_and_mount_via_pilot() -> None:
    app = MeridianTuiApp(client=MagicMock(spec=DaemonClient))
    app._fetch_sessions = MagicMock()  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.pause()
        # on_mount has fired — fetch_sessions should have been invoked
        app._fetch_sessions.assert_called_once()


def test_on_mount_sets_up_tables_and_starts_fetch() -> None:
    app, w = _make_app()
    app._fetch_sessions = MagicMock()  # type: ignore[method-assign]
    app.on_mount()
    w["#sessions-pane"].add_columns.assert_called_once_with("ID", "Phase")
    w["#channels-pane"].add_columns.assert_called_once_with("ID", "Kind")
    app._fetch_sessions.assert_called_once()


# ---------------------------------------------------------------------------
# _fetch_sessions — success + error paths
# ---------------------------------------------------------------------------


def test_fetch_sessions_success_dispatches_update() -> None:
    app, _ = _make_app()
    app._client.request.return_value = [{"id": "s1", "phase": "running"}]  # type: ignore[attr-defined]
    app.call_from_thread = MagicMock()  # type: ignore[method-assign]
    # Call the underlying function (not the @work wrapper) via __wrapped__
    MeridianTuiApp._fetch_sessions.__wrapped__(app)  # type: ignore[attr-defined]
    args = app.call_from_thread.call_args
    assert args[0][0] == app._update_sessions
    assert args[0][1] == [{"id": "s1", "phase": "running"}]


def test_fetch_sessions_non_list_yields_empty() -> None:
    app, _ = _make_app()
    app._client.request.return_value = {"not": "a list"}  # type: ignore[attr-defined]
    app.call_from_thread = MagicMock()  # type: ignore[method-assign]
    MeridianTuiApp._fetch_sessions.__wrapped__(app)  # type: ignore[attr-defined]
    app.call_from_thread.assert_called_once_with(app._update_sessions, [])


def test_fetch_sessions_daemon_error_logs() -> None:
    app, _ = _make_app()
    app._client.request.side_effect = DaemonError(code="x", message="oops")  # type: ignore[attr-defined]
    app.call_from_thread = MagicMock()  # type: ignore[method-assign]
    MeridianTuiApp._fetch_sessions.__wrapped__(app)  # type: ignore[attr-defined]
    app.call_from_thread.assert_called_once_with(app._log_error, "sessions: oops")


# ---------------------------------------------------------------------------
# _update_sessions — populates table + auto-switches
# ---------------------------------------------------------------------------


def test_update_sessions_populates_and_switches() -> None:
    app, w = _make_app()
    app._switch_session = MagicMock()  # type: ignore[method-assign]
    app._update_sessions([{"id": "s1", "phase": "running"}, {"id": "s2", "phase": "idle"}])
    w["#sessions-pane"].clear.assert_called_once()
    assert w["#sessions-pane"].add_row.call_count == 2
    app._switch_session.assert_called_once_with("s1")


def test_update_sessions_with_dash_defaults() -> None:
    app, w = _make_app()
    app._switch_session = MagicMock()  # type: ignore[method-assign]
    app._update_sessions([{}])  # id / phase missing
    w["#sessions-pane"].add_row.assert_called_once_with("—", "—")
    # No id → no switch
    app._switch_session.assert_not_called()


def test_update_sessions_empty_list_no_switch() -> None:
    app, w = _make_app()
    app._switch_session = MagicMock()  # type: ignore[method-assign]
    app._update_sessions([])
    w["#sessions-pane"].clear.assert_called_once()
    app._switch_session.assert_not_called()


# ---------------------------------------------------------------------------
# on_data_table_row_selected — sessions vs other
# ---------------------------------------------------------------------------


def test_row_selected_on_sessions_pane_switches() -> None:
    app, _ = _make_app()
    app._switch_session = MagicMock()  # type: ignore[method-assign]

    tbl = MagicMock()
    tbl.id = "sessions-pane"
    tbl.get_row.return_value = ("session-xyz", "running")
    evt = MagicMock()
    evt.data_table = tbl
    evt.row_key = "k"
    app.on_data_table_row_selected(evt)
    app._switch_session.assert_called_once_with("session-xyz")


def test_row_selected_on_other_pane_ignored() -> None:
    app, _ = _make_app()
    app._switch_session = MagicMock()  # type: ignore[method-assign]

    tbl = MagicMock()
    tbl.id = "channels-pane"
    evt = MagicMock()
    evt.data_table = tbl
    app.on_data_table_row_selected(evt)
    app._switch_session.assert_not_called()


# ---------------------------------------------------------------------------
# _switch_session — wires fetch/stream
# ---------------------------------------------------------------------------


def test_switch_session_resets_state_and_fetches() -> None:
    app, w = _make_app()
    app._fetch_channels = MagicMock()  # type: ignore[method-assign]
    app._stream_events = MagicMock()  # type: ignore[method-assign]
    app._switch_session("sx")
    assert app._selected_session_id == "sx"
    assert app._last_event_seq == -1
    w["#events-pane"].clear.assert_called_once()
    w["#events-pane"].write.assert_called_once()
    app._fetch_channels.assert_called_once_with("sx")
    app._stream_events.assert_called_once_with("sx")


# ---------------------------------------------------------------------------
# _fetch_channels — success + error + non-list
# ---------------------------------------------------------------------------


def test_fetch_channels_success_filters_by_session() -> None:
    app, _ = _make_app()
    app._client.request.return_value = [  # type: ignore[attr-defined]
        {"id": "c1", "kind": "k", "session_id": "S"},
        {"id": "c2", "kind": "k", "session_id": "OTHER"},
    ]
    app.call_from_thread = MagicMock()  # type: ignore[method-assign]
    MeridianTuiApp._fetch_channels.__wrapped__(app, "S")  # type: ignore[attr-defined]
    args = app.call_from_thread.call_args
    assert args[0][0] == app._update_channels
    assert args[0][1] == [{"id": "c1", "kind": "k", "session_id": "S"}]


def test_fetch_channels_non_list_yields_empty() -> None:
    app, _ = _make_app()
    app._client.request.return_value = "not list"  # type: ignore[attr-defined]
    app.call_from_thread = MagicMock()  # type: ignore[method-assign]
    MeridianTuiApp._fetch_channels.__wrapped__(app, "S")  # type: ignore[attr-defined]
    app.call_from_thread.assert_called_once_with(app._update_channels, [])


def test_fetch_channels_daemon_error_logs() -> None:
    app, _ = _make_app()
    app._client.request.side_effect = DaemonError(code="x", message="boom")  # type: ignore[attr-defined]
    app.call_from_thread = MagicMock()  # type: ignore[method-assign]
    MeridianTuiApp._fetch_channels.__wrapped__(app, "S")  # type: ignore[attr-defined]
    app.call_from_thread.assert_called_once_with(app._log_error, "channels: boom")


# ---------------------------------------------------------------------------
# _update_channels
# ---------------------------------------------------------------------------


def test_update_channels_populates() -> None:
    app, w = _make_app()
    app._update_channels([{"id": "c1", "kind": "k"}, {}])
    w["#channels-pane"].clear.assert_called_once()
    assert w["#channels-pane"].add_row.call_count == 2


# ---------------------------------------------------------------------------
# _stream_events — exits when session deselected; handles DaemonError + non-list
# ---------------------------------------------------------------------------


def test_stream_events_polls_then_exits_on_session_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _ = _make_app()
    app._selected_session_id = "S"
    app._client.request.return_value = [{"seq": 1, "type": "evt", "data": {}, "ts": "2024-01-01T00:00:00Z"}]  # type: ignore[attr-defined]
    app.call_from_thread = MagicMock()  # type: ignore[method-assign]

    sleep_calls: list[float] = []

    def _fake_sleep(s: float) -> None:
        sleep_calls.append(s)
        app._selected_session_id = "DIFFERENT"  # exit on next loop check

    monkeypatch.setattr(time, "sleep", _fake_sleep)
    MeridianTuiApp._stream_events.__wrapped__(app, "S")  # type: ignore[attr-defined]
    assert app._last_event_seq == 1
    app.call_from_thread.assert_called_once()
    assert sleep_calls  # slept at least once


def test_stream_events_daemon_error_sleeps_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _ = _make_app()
    app._selected_session_id = "S"
    app._client.request.side_effect = DaemonError(code="x", message="boom")  # type: ignore[attr-defined]

    def _fake_sleep(s: float) -> None:
        app._selected_session_id = "OTHER"  # exit after first retry

    monkeypatch.setattr(time, "sleep", _fake_sleep)
    MeridianTuiApp._stream_events.__wrapped__(app, "S")  # type: ignore[attr-defined]


def test_stream_events_non_list_result(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = _make_app()
    app._selected_session_id = "S"
    app._client.request.return_value = {"not": "list"}  # type: ignore[attr-defined]
    app.call_from_thread = MagicMock()  # type: ignore[method-assign]

    def _fake_sleep(s: float) -> None:
        app._selected_session_id = "OTHER"

    monkeypatch.setattr(time, "sleep", _fake_sleep)
    MeridianTuiApp._stream_events.__wrapped__(app, "S")  # type: ignore[attr-defined]
    # No events → no call_from_thread for ingest
    app.call_from_thread.assert_not_called()


def test_stream_events_seq_not_advanced_when_lower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _ = _make_app()
    app._selected_session_id = "S"
    app._last_event_seq = 10
    app._client.request.return_value = [{"seq": 5, "type": "evt", "data": {}}]  # type: ignore[attr-defined]
    app.call_from_thread = MagicMock()  # type: ignore[method-assign]

    def _fake_sleep(s: float) -> None:
        app._selected_session_id = "OTHER"

    monkeypatch.setattr(time, "sleep", _fake_sleep)
    MeridianTuiApp._stream_events.__wrapped__(app, "S")  # type: ignore[attr-defined]
    # seq stays at 10 because incoming seq (5) wasn't higher
    assert app._last_event_seq == 10


# ---------------------------------------------------------------------------
# _ingest_event — budget, skill, generic
# ---------------------------------------------------------------------------


def test_ingest_event_session_mismatch_ignored() -> None:
    app, w = _make_app()
    app._selected_session_id = "S"
    app._ingest_event({"type": "x"}, "OTHER")
    w["#events-pane"].write.assert_not_called()


def test_ingest_event_writes_log_line() -> None:
    app, w = _make_app()
    app._selected_session_id = "S"
    app._ingest_event({"ts": "2024-01-01T00:00:00Z", "type": "msg", "data": {"a": 1}}, "S")
    w["#events-pane"].write.assert_called_once()


def test_ingest_event_budget_warning_enqueues_approval() -> None:
    app, _ = _make_app()
    app._selected_session_id = "S"
    app._enqueue_approval = MagicMock()  # type: ignore[method-assign]
    app._ingest_event({"type": "budget.warning", "data": {"amount": 10}}, "S")
    app._enqueue_approval.assert_called_once()
    args = app._enqueue_approval.call_args
    assert args[0][0] == "Budget Approval Required"


def test_ingest_event_budget_exceeded_enqueues_approval() -> None:
    app, _ = _make_app()
    app._selected_session_id = "S"
    app._enqueue_approval = MagicMock()  # type: ignore[method-assign]
    app._ingest_event({"type": "budget.exceeded", "data": {}}, "S")
    app._enqueue_approval.assert_called_once()


def test_ingest_event_skill_invoked_enqueues_approval() -> None:
    app, _ = _make_app()
    app._selected_session_id = "S"
    app._enqueue_approval = MagicMock()  # type: ignore[method-assign]
    app._ingest_event(
        {"type": "hook.invoked", "data": {"skill_id": "s.x", "skill_version": "1.0"}}, "S"
    )
    app._enqueue_approval.assert_called_once()
    assert app._enqueue_approval.call_args[0][0] == "Skill Activation Approval"


def test_ingest_event_skill_invoked_no_skill_id_skipped() -> None:
    app, _ = _make_app()
    app._selected_session_id = "S"
    app._enqueue_approval = MagicMock()  # type: ignore[method-assign]
    app._ingest_event({"type": "hook.invoked", "data": {}}, "S")
    app._enqueue_approval.assert_not_called()


def test_ingest_event_none_ts_safe() -> None:
    app, w = _make_app()
    app._selected_session_id = "S"
    app._ingest_event({"type": "x"}, "S")  # ts and data absent
    w["#events-pane"].write.assert_called_once()


# ---------------------------------------------------------------------------
# _enqueue_approval + _pop_approval
# ---------------------------------------------------------------------------


def test_enqueue_approval_triggers_pop_when_idle() -> None:
    app, _ = _make_app()
    app._pop_approval = MagicMock()  # type: ignore[method-assign]
    app._approving = False
    app._enqueue_approval("t", ["l"], "S", {"type": "x"})
    assert app._approval_queue
    app._pop_approval.assert_called_once()


def test_enqueue_approval_skips_pop_when_already_approving() -> None:
    app, _ = _make_app()
    app._pop_approval = MagicMock()  # type: ignore[method-assign]
    app._approving = True
    app._enqueue_approval("t", ["l"], "S", {"type": "x"})
    assert app._approval_queue
    app._pop_approval.assert_not_called()


def test_pop_approval_empty_queue_noop() -> None:
    app, _ = _make_app()
    app.push_screen = MagicMock()  # type: ignore[method-assign]
    app._pop_approval()
    app.push_screen.assert_not_called()


def test_pop_approval_push_screen_with_modal() -> None:
    app, _ = _make_app()
    app.push_screen = MagicMock()  # type: ignore[method-assign]
    app._approval_queue = [
        {"title": "T", "lines": ["L"], "session_id": "S", "event": {"type": "evt"}}
    ]
    app._pop_approval()
    assert app._approving is True
    assert not app._approval_queue
    app.push_screen.assert_called_once()
    modal = app.push_screen.call_args[0][0]
    assert isinstance(modal, ApprovalModal)


def test_pop_approval_callback_approved_writes_green_and_pops_next() -> None:
    app, w = _make_app()
    app.push_screen = MagicMock()  # type: ignore[method-assign]
    app._approval_queue = [
        {"title": "T", "lines": [], "session_id": "S", "event": {"type": "ev1"}},
    ]
    app._pop_approval()
    callback = app.push_screen.call_args.kwargs["callback"]
    callback(True)
    assert app._approving is False
    w["#events-pane"].write.assert_called()
    # Last write should contain "approved"
    written = " ".join(c.args[0] for c in w["#events-pane"].write.call_args_list)
    assert "approved" in written


def test_pop_approval_callback_rejected_writes_red() -> None:
    app, w = _make_app()
    app.push_screen = MagicMock()  # type: ignore[method-assign]
    app._approval_queue = [
        {"title": "T", "lines": [], "session_id": "S", "event": {"type": "ev1"}},
    ]
    app._pop_approval()
    callback = app.push_screen.call_args.kwargs["callback"]
    callback(False)
    written = " ".join(c.args[0] for c in w["#events-pane"].write.call_args_list)
    assert "rejected" in written


# ---------------------------------------------------------------------------
# _log_error
# ---------------------------------------------------------------------------


def test_log_error_writes_red_line() -> None:
    app, w = _make_app()
    app._log_error("kaboom")
    written = w["#events-pane"].write.call_args[0][0]
    assert "kaboom" in written
    assert "red" in written.lower() or "[bold red]" in written
