"""meridiantui — interactive TUI gateway for the Meridian daemon.

Three-pane layout: session list | channel view | live event tail.
Keyboard-driven approval prompts surface for budget.warning,
budget.exceeded, and hook.invoked (skill-activation) events.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import click
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Label, RichLog, Static
from textual import work

from ._audit import write_audit
from ._client import DaemonClient, DaemonError
from ._telemetry import get_tracer, record_failure, record_invocation_event

_POLL_INTERVAL = 2.0
_BUDGET_EVENT_TYPES = frozenset({"budget.warning", "budget.exceeded"})
_SKILL_EVENT_TYPES = frozenset({"hook.invoked"})


# ---------------------------------------------------------------------------
# Approval modal
# ---------------------------------------------------------------------------


class ApprovalModal(ModalScreen[bool]):
    """Keyboard-driven modal prompt for budget / skill-activation approval."""

    BINDINGS = [
        Binding("a", "approve", "Approve", priority=True),
        Binding("r", "reject", "Reject", priority=True),
        Binding("escape", "reject", "Cancel"),
    ]

    def __init__(self, title: str, lines: list[str]) -> None:
        super().__init__()
        self._modal_title = title
        self._lines = lines

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label(self._modal_title, id="approval-title")
            yield Static("")
            for line in self._lines:
                yield Label(line)
            yield Static("")
            yield Label("[dim]a[/dim] approve   [dim]r[/dim] reject", markup=True)

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# Main TUI application
# ---------------------------------------------------------------------------


class MeridianTuiApp(App[None]):
    """Multi-session TUI gateway with live event tail and approval prompts."""

    CSS = """
    Screen {
        background: $surface;
    }
    #panes {
        height: 1fr;
    }
    #sessions-container {
        width: 26;
        border: solid $accent;
    }
    #channels-container {
        width: 30;
        border: solid $accent;
    }
    #events-container {
        width: 1fr;
        border: solid $accent;
    }
    .pane-title {
        height: 1;
        background: $accent;
        color: $text;
        content-align: center middle;
        text-style: bold;
    }
    #sessions-pane, #channels-pane, #events-pane {
        height: 1fr;
    }
    ApprovalModal {
        align: center middle;
    }
    #approval-dialog {
        padding: 1 2;
        background: $surface;
        border: solid $warning;
        width: 64;
        height: auto;
    }
    #approval-title {
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("tab", "focus_next", "Next Pane"),
        Binding("shift+tab", "focus_previous", "Prev Pane"),
    ]

    def __init__(self, client: DaemonClient) -> None:
        super().__init__()
        self._client = client
        self._selected_session_id: str | None = None
        self._last_event_seq: int = -1
        self._approval_queue: list[dict[str, Any]] = []
        self._approving = False

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="panes"):
            with Vertical(id="sessions-container"):
                yield Label("Sessions", classes="pane-title")
                yield DataTable(id="sessions-pane")
            with Vertical(id="channels-container"):
                yield Label("Channels", classes="pane-title")
                yield DataTable(id="channels-pane")
            with Vertical(id="events-container"):
                yield Label("Events", classes="pane-title")
                yield RichLog(id="events-pane", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        sessions_tbl = self.query_one("#sessions-pane", DataTable)
        sessions_tbl.add_columns("ID", "Phase")
        sessions_tbl.cursor_type = "row"

        channels_tbl = self.query_one("#channels-pane", DataTable)
        channels_tbl.add_columns("ID", "Kind")
        channels_tbl.cursor_type = "row"

        self._fetch_sessions()

    # ------------------------------------------------------------------
    # Session list
    # ------------------------------------------------------------------

    @work(thread=True)
    def _fetch_sessions(self) -> None:
        try:
            result = self._client.request("GET", "/v1/x/sessions")
        except DaemonError as exc:
            self.call_from_thread(self._log_error, f"sessions: {exc.message}")
            return
        sessions = result if isinstance(result, list) else []
        self.call_from_thread(self._update_sessions, sessions)

    def _update_sessions(self, sessions: list[dict[str, Any]]) -> None:
        tbl = self.query_one("#sessions-pane", DataTable)
        tbl.clear()
        for sess in sessions:
            tbl.add_row(sess.get("id", "—"), sess.get("phase", "—"))
        if sessions:
            first_id = sessions[0].get("id")
            if first_id:
                self._switch_session(first_id)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "sessions-pane":
            return
        row = event.data_table.get_row(event.row_key)
        self._switch_session(str(row[0]))

    def _switch_session(self, session_id: str) -> None:
        self._selected_session_id = session_id
        self._last_event_seq = -1
        log = self.query_one("#events-pane", RichLog)
        log.clear()
        log.write(f"[dim]session: {session_id}[/dim]")
        self._fetch_channels(session_id)
        self._stream_events(session_id)

    # ------------------------------------------------------------------
    # Channel view
    # ------------------------------------------------------------------

    @work(thread=True)
    def _fetch_channels(self, session_id: str) -> None:
        try:
            result = self._client.request("GET", "/v1/x/channels")
        except DaemonError as exc:
            self.call_from_thread(self._log_error, f"channels: {exc.message}")
            return
        all_channels = result if isinstance(result, list) else []
        channels = [c for c in all_channels if c.get("session_id") == session_id]
        self.call_from_thread(self._update_channels, channels)

    def _update_channels(self, channels: list[dict[str, Any]]) -> None:
        tbl = self.query_one("#channels-pane", DataTable)
        tbl.clear()
        for ch in channels:
            tbl.add_row(ch.get("id", "—"), ch.get("kind", "—"))

    # ------------------------------------------------------------------
    # Live event tail
    # ------------------------------------------------------------------

    @work(thread=True, exclusive=True)
    def _stream_events(self, session_id: str) -> None:
        while self._selected_session_id == session_id:
            try:
                result = self._client.request(
                    "GET",
                    f"/v1/sessions/{session_id}/events?since={self._last_event_seq}",
                )
            except DaemonError:
                time.sleep(_POLL_INTERVAL)
                continue
            events = result if isinstance(result, list) else []
            for evt in events:
                seq = evt.get("seq", 0)
                if seq > self._last_event_seq:
                    self._last_event_seq = seq
                self.call_from_thread(self._ingest_event, evt, session_id)
            time.sleep(_POLL_INTERVAL)

    def _ingest_event(self, evt: dict[str, Any], session_id: str) -> None:
        if session_id != self._selected_session_id:
            return
        ts = (evt.get("ts") or "")[:19]
        etype = evt.get("type", "?")
        data = evt.get("data") or {}
        log = self.query_one("#events-pane", RichLog)
        log.write(
            f"[dim]{ts}[/dim] [bold]{etype}[/bold] "
            f"{json.dumps(data, separators=(',', ':'))}"
        )

        if etype in _BUDGET_EVENT_TYPES:
            self._enqueue_approval(
                "Budget Approval Required",
                [
                    f"Session: {session_id}",
                    f"Event:   {etype}",
                    f"Detail:  {json.dumps(data)}",
                ],
                session_id,
                evt,
            )
        elif etype in _SKILL_EVENT_TYPES and data.get("skill_id"):
            self._enqueue_approval(
                "Skill Activation Approval",
                [
                    f"Session: {session_id}",
                    f"Skill:   {data.get('skill_id')}",
                    f"Version: {data.get('skill_version', 'latest')}",
                ],
                session_id,
                evt,
            )

    # ------------------------------------------------------------------
    # Approval queue
    # ------------------------------------------------------------------

    def _enqueue_approval(
        self,
        title: str,
        lines: list[str],
        session_id: str,
        evt: dict[str, Any],
    ) -> None:
        self._approval_queue.append(
            {"title": title, "lines": lines, "session_id": session_id, "event": evt}
        )
        if not self._approving:
            self._pop_approval()

    def _pop_approval(self) -> None:
        if not self._approval_queue:
            return
        item = self._approval_queue.pop(0)
        self._approving = True

        def _on_result(approved: bool) -> None:
            self._approving = False
            log = self.query_one("#events-pane", RichLog)
            etype = item["event"].get("type", "")
            if approved:
                log.write(f"[bold green]{etype} approved[/bold green]")
            else:
                log.write(f"[bold red]{etype} rejected[/bold red]")
            self._pop_approval()

        self.push_screen(ApprovalModal(item["title"], item["lines"]), callback=_on_result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_error(self, message: str) -> None:
        log = self.query_one("#events-pane", RichLog)
        log.write(f"[bold red]error: {message}[/bold red]")


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command("meridiantui")
@click.pass_context
def meridiantui(ctx: click.Context) -> None:
    """Launch the interactive TUI gateway with multi-session navigation."""
    client: DaemonClient = ctx.find_root().obj

    span_name = "tui.launch"
    tracer = get_tracer()
    with tracer.start_as_current_span(
        span_name, attributes={"operation": "tui.launch"}
    ) as span:
        record_invocation_event(
            span,
            {
                "event.name": f"{span_name}.invocation",
                "operation": "tui.launch",
            },
        )
        write_audit("info", f"{span_name}.invoked", {"operation": "tui.launch"})

        try:
            app = MeridianTuiApp(client=client)
            app.run()
        except Exception as exc:
            code = "tui_launch_failed"
            message = str(exc)
            record_failure(span, code, message)
            write_audit(
                "error",
                f"{span_name}.failed",
                {"code": code, "message": message},
            )
            click.echo(f"error: [{code}] {message}", err=True)
            sys.exit(1)

        span.add_event(f"{span_name}.completed")
