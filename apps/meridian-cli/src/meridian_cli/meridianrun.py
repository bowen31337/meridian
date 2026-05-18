"""meridianrun — stream session events to terminal with a human-friendly TTY renderer.

Renders:
  - token streaming   (message.delta → inline text, flushed on message.added)
  - tool-call inlining (requested → dispatched → result / error)
  - collapsed thinking blocks
  - color-coded phase transitions
"""

from __future__ import annotations

import sys
import time
from typing import Any

import click
from rich.console import Console

from ._audit import write_audit
from ._client import DaemonClient, DaemonError
from ._telemetry import get_tracer, record_failure, record_invocation_event

_POLL_INTERVAL = 1.0
_TERMINAL_PHASES = frozenset({"done", "cancelled", "error"})

_PHASE_STYLES: dict[str, str] = {
    "idle": "dim",
    "running": "bold cyan",
    "waiting": "bold yellow",
    "done": "bold green",
    "error": "bold red",
    "cancelled": "dim red",
}


def _phase_style(phase: str) -> str:
    return _PHASE_STYLES.get(phase, "bold blue")


# ---------------------------------------------------------------------------
# TTY renderer
# ---------------------------------------------------------------------------


class _Renderer:
    """Stateful renderer that converts session events into terminal output."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._streaming = False   # True while accumulating message.delta tokens
        self._in_thinking = False  # True while inside a thinking block

    def render(self, event: dict[str, Any]) -> None:
        etype = event.get("type", "")
        data = event.get("data") or {}
        ts = (event.get("ts") or "")[:19]

        if etype == "session.created":
            self._console.print(f"[dim]{ts} session created[/dim]")

        elif etype == "session.phase_change":
            self._flush()
            prev = data.get("prev_phase", "?")
            phase = data.get("phase", "?")
            ps = _phase_style(phase)
            self._console.print(
                f"[dim]{ts}[/dim] [bold]Phase:[/bold] [dim]{prev}[/dim]"
                f" → [{ps}]{phase}[/{ps}]"
            )

        elif etype == "message.delta":
            self._render_delta(data)

        elif etype == "message.added":
            self._flush()
            role = data.get("role", "")
            content = data.get("content", "")
            if role == "user" and isinstance(content, str) and content:
                self._console.print(
                    f"[dim]{ts}[/dim] [bold magenta]User:[/bold magenta] {content}"
                )

        elif etype == "tool_call.requested":
            self._flush()
            name = data.get("tool_name") or data.get("name") or "?"
            args_raw = data.get("arguments") or data.get("args") or {}
            if isinstance(args_raw, dict):
                args_str = ", ".join(
                    f"{k}={repr(v)}" for k, v in list(args_raw.items())[:3]
                )
                if len(args_raw) > 3:
                    args_str += ", …"
            else:
                args_str = str(args_raw)[:80]
            self._console.print(
                f"[dim]{ts}[/dim] [bold yellow]⟳ {name}[/bold yellow]"
                f"([dim]{args_str}[/dim])"
            )

        elif etype == "tool_call.dispatched":
            self._console.print("  [dim]dispatching…[/dim]")

        elif etype == "tool_call.result":
            result = data.get("result") or data.get("output") or ""
            if isinstance(result, str):
                snippet = result[:120].replace("\n", " ")
            else:
                import json
                snippet = json.dumps(result)[:120]
            self._console.print(f"  [bold green]✓[/bold green] [dim]{snippet}[/dim]")

        elif etype == "tool_call.error":
            error = data.get("error") or data.get("message") or "unknown error"
            self._console.print(f"  [bold red]✗[/bold red] {error}")

        elif etype == "model_call.started":
            self._console.print(f"[dim]{ts} ↑ model call…[/dim]")

        elif etype == "usage.delta":
            p = data.get("prompt_tokens") or 0
            c = data.get("completion_tokens") or 0
            if p or c:
                self._console.print(f"  [dim]tokens: +{p}p +{c}c[/dim]")

        elif etype == "budget.warning":
            self._flush()
            detail = data.get("message") or data.get("detail") or str(data)
            self._console.print(f"[bold yellow]⚠ Budget warning:[/bold yellow] {detail}")

        elif etype == "budget.exceeded":
            self._flush()
            detail = data.get("message") or data.get("detail") or str(data)
            self._console.print(f"[bold red]⊘ Budget exceeded:[/bold red] {detail}")

        elif etype == "error":
            self._flush()
            message = data.get("message") or str(data)
            self._console.print(f"[bold red]⚠ Error:[/bold red] {message}")

        elif etype in ("checkpoint.created", "child_session.spawned", "child_session.completed"):
            self._flush()
            self._console.print(f"[dim]{ts} {etype}[/dim]")

        # hook.invoked, hook.verdict, memory.read, memory.write,
        # acp.inbound, acp.outbound, channel.inbound, channel.outbound,
        # model_call.completed — intentionally silent

    def _render_delta(self, data: dict[str, Any]) -> None:
        """Render a message.delta event with thinking-block collapsing."""
        # Try structured content blocks first (Anthropic streaming format)
        content_blocks = None
        delta = data.get("delta")
        if isinstance(delta, dict):
            content_blocks = delta.get("content")
            if content_blocks is None and delta.get("type") in ("text", "text_delta"):
                text = delta.get("text") or ""
                if text:
                    self._emit_text(text)
                return

        if content_blocks is None:
            content_blocks = data.get("content")

        if isinstance(content_blocks, list):
            for block in content_blocks:
                btype = block.get("type", "")
                if btype == "thinking":
                    if not self._in_thinking:
                        if self._streaming:
                            self._console.print("")
                        self._console.print("[dim]<thinking>…</thinking>[/dim]")
                        self._in_thinking = True
                elif btype in ("text", "text_delta"):
                    text = block.get("text") or block.get("delta") or ""
                    if text:
                        self._in_thinking = False
                        self._emit_text(text)
            return

        # Flat text delta fallback
        text = ""
        if isinstance(delta, dict):
            text = delta.get("text") or ""
        if not text and isinstance(data.get("content"), str):
            text = data["content"]
        if text:
            self._in_thinking = False
            self._emit_text(text)

    def _emit_text(self, text: str) -> None:
        if not self._streaming:
            self._console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
            self._streaming = True
        self._console.print(text, end="")

    def _flush(self) -> None:
        """End any in-progress streaming line."""
        if self._streaming:
            self._console.print("")
            self._streaming = False
        self._in_thinking = False

    def close(self) -> None:
        self._flush()


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command("meridianrun")
@click.argument("session_id")
@click.option(
    "--follow/--no-follow",
    default=True,
    help="Keep tailing until the session reaches a terminal phase.",
)
@click.pass_context
def meridianrun(ctx: click.Context, session_id: str, follow: bool) -> None:
    """Stream session events to terminal with human-friendly rendering."""
    client: DaemonClient = ctx.find_root().obj
    console = Console(highlight=False)

    span_name = "run.stream"
    tracer = get_tracer()
    with tracer.start_as_current_span(
        span_name,
        attributes={"operation": "run.stream", "session.id": session_id},
    ) as span:
        record_invocation_event(
            span,
            {
                "event.name": f"{span_name}.invocation",
                "operation": "run.stream",
                "session_id": session_id,
            },
        )
        write_audit(
            "info",
            f"{span_name}.invoked",
            {"operation": "run.stream", "session_id": session_id},
        )

        renderer = _Renderer(console)
        last_seq = -1
        terminal_phase: str | None = None

        try:
            while True:
                try:
                    result = client.request(
                        "GET",
                        f"/v1/sessions/{session_id}/events?since={last_seq}",
                    )
                except DaemonError as exc:
                    renderer.close()
                    code = exc.code
                    message = exc.message
                    record_failure(span, code, message)
                    write_audit(
                        "error",
                        f"{span_name}.failed",
                        {"code": code, "message": message, "session_id": session_id},
                    )
                    click.echo(f"error: [{code}] {message}", err=True)
                    sys.exit(1)

                events = result if isinstance(result, list) else []
                for evt in events:
                    seq = evt.get("seq", 0)
                    if seq > last_seq:
                        last_seq = seq
                    renderer.render(evt)
                    if evt.get("type") == "session.phase_change":
                        terminal_phase = (evt.get("data") or {}).get("phase")

                if not follow or terminal_phase in _TERMINAL_PHASES:
                    break

                time.sleep(_POLL_INTERVAL)

        except KeyboardInterrupt:
            renderer.close()

        else:
            renderer.close()

        span.add_event(f"{span_name}.completed")
