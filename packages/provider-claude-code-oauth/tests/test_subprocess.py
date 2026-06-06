"""Unit tests for _subprocess helpers and CliSubprocessManager internals.

These exercise branches not reachable through the integration-style stub:
``_opts_to_dict`` block conversion, ``_parse_event`` variants, and the manager's
internal I/O / lifecycle helpers driven with in-memory fakes (no real CLI).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from meridian_sdk_provider.types import (
    MessageStartEvent,
    MessageStopEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)

from meridian_provider_claude_code_oauth._subprocess import (
    CliCallTimeoutError,
    CliSubprocessError,
    CliSubprocessManager,
    _opts_to_dict,
    _parse_event,
)

# ---------------------------------------------------------------------------
# In-memory subprocess fakes
# ---------------------------------------------------------------------------


class _FakeStdin:
    def __init__(self) -> None:
        self.written: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        pass


class _FakeStdout:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeProc:
    def __init__(self, lines: list[bytes] | None = None, returncode: int | None = None) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(lines or [])
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.returncode = 0
        return 0


def _line(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj) + "\n").encode()


# ---------------------------------------------------------------------------
# _opts_to_dict — list-content block conversion + optional fields
# ---------------------------------------------------------------------------


def test_opts_to_dict_converts_blocks_and_optional_fields() -> None:
    blocks = [
        SimpleNamespace(type="text", text="plain", cache_control=None),
        SimpleNamespace(type="text", text="cached", cache_control={"type": "ephemeral"}),
        SimpleNamespace(type="tool_use", id="t1", name="fn", input={"a": 1}),
        SimpleNamespace(type="tool_result", tool_use_id="t1", content="done"),
        SimpleNamespace(type="tool_result", tool_use_id="t2", content=["a", "b"]),
        SimpleNamespace(type="thinking", thinking="reason", signature="sig"),
        SimpleNamespace(type="image"),  # unknown block — skipped
    ]
    msg = SimpleNamespace(role="user", content=blocks)
    opts = SimpleNamespace(
        model="claude-x",
        messages=[msg],
        max_tokens=256,
        system="be terse",
        tools=[SimpleNamespace(name="fn", description="d", input_schema={"type": "object"})],
        temperature=0.3,
        enable_thinking=True,
        thinking_budget_tokens=2000,
        session_id="sess-1",
    )

    d = _opts_to_dict(opts)  # type: ignore[arg-type]

    content = d["messages"][0]["content"]
    assert {"type": "text", "text": "plain"} in content
    assert {"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}} in content
    assert {"type": "tool_use", "id": "t1", "name": "fn", "input": {"a": 1}} in content
    assert {"type": "tool_result", "tool_use_id": "t1", "content": "done"} in content
    tr_list = next(b for b in content if b.get("tool_use_id") == "t2")
    assert tr_list["content"] == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]
    assert {"type": "thinking", "thinking": "reason", "signature": "sig"} in content

    assert d["system"] == "be terse"
    assert d["temperature"] == 0.3
    assert d["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert d["session_id"] == "sess-1"
    assert d["tools"][0]["name"] == "fn"


# ---------------------------------------------------------------------------
# _parse_event — all branches
# ---------------------------------------------------------------------------


def test_parse_event_message_start() -> None:
    ev = _parse_event({"type": "message_start", "model": "m", "input_tokens": 3}, "p")
    assert isinstance(ev, MessageStartEvent)
    assert ev.model == "m"
    assert ev.provider == "p"


def test_parse_event_text_delta() -> None:
    ev = _parse_event({"type": "text_delta", "text": "hi"}, "p")
    assert isinstance(ev, TextDeltaEvent)
    assert ev.text == "hi"


def test_parse_event_thinking_delta() -> None:
    ev = _parse_event({"type": "thinking_delta", "thinking": "hmm"}, "p")
    assert isinstance(ev, ThinkingDeltaEvent)
    assert ev.thinking == "hmm"


def test_parse_event_tool_use_start() -> None:
    ev = _parse_event({"type": "tool_use_start", "id": "x", "name": "fn"}, "p")
    assert isinstance(ev, ToolUseStartEvent)
    assert ev.id == "x"
    assert ev.name == "fn"


def test_parse_event_tool_input_delta() -> None:
    ev = _parse_event({"type": "tool_input_delta", "id": "x", "partial_json": "{}"}, "p")
    assert isinstance(ev, ToolInputDeltaEvent)
    assert ev.partial_json == "{}"


def test_parse_event_message_stop() -> None:
    ev = _parse_event(
        {"type": "message_stop", "input_tokens": 1, "output_tokens": 2, "stop_reason": "end"},
        "p",
    )
    assert isinstance(ev, MessageStopEvent)
    assert ev.stop_reason == "end"


def test_parse_event_unknown_returns_none() -> None:
    assert _parse_event({"type": "pong"}, "p") is None


# ---------------------------------------------------------------------------
# _read_events — stdout-level branches
# ---------------------------------------------------------------------------


def _manager() -> CliSubprocessManager:
    return CliSubprocessManager("claude", "1.0.0", health_interval_s=9999)


async def test_read_events_closed_stdout_raises() -> None:
    mgr = _manager()
    mgr._proc = _FakeProc(lines=[])  # readline → b"" → closed
    with pytest.raises(CliSubprocessError, match="closed stdout"):
        async for _ in mgr._read_events("cid"):
            pass


async def test_read_events_invalid_json_raises() -> None:
    mgr = _manager()
    mgr._proc = _FakeProc(lines=[b"not json {{{\n"])
    with pytest.raises(CliSubprocessError, match="invalid JSON"):
        async for _ in mgr._read_events("cid"):
            pass


async def test_read_events_skips_none_events_until_done() -> None:
    mgr = _manager()
    mgr._proc = _FakeProc(
        lines=[
            _line({"type": "pong", "id": "x"}),  # parses to None → loop continues
            _line({"type": "text_delta", "text": "hi"}),
            _line({"type": "done"}),
        ]
    )
    events = [e async for e in mgr._read_events("cid")]
    assert len(events) == 1
    assert isinstance(events[0], TextDeltaEvent)


# ---------------------------------------------------------------------------
# call() — cancellation kills and respawns
# ---------------------------------------------------------------------------


async def test_call_cancelled_kills_and_respawns() -> None:
    mgr = _manager()
    mgr._proc = _FakeProc()

    async def _noop() -> None:
        pass

    mgr._ensure_alive = _noop  # type: ignore[method-assign]

    async def _read(_cid: str) -> Any:
        raise asyncio.CancelledError()
        yield  # pragma: no cover - makes this an async generator

    mgr._read_events = _read  # type: ignore[method-assign]

    killed: list[int] = []

    async def _ks() -> None:
        killed.append(1)

    mgr._kill_and_spawn = _ks  # type: ignore[method-assign]

    opts = SimpleNamespace(
        model="m",
        messages=[SimpleNamespace(role="user", content="hi")],
        max_tokens=10,
        system=None,
        tools=None,
        temperature=None,
        enable_thinking=False,
        thinking_budget_tokens=None,
        session_id=None,
    )
    with pytest.raises(asyncio.CancelledError):
        async for _ in mgr.call(opts):  # type: ignore[arg-type]
            pass
    assert killed == [1]
    assert not mgr._call_active


# ---------------------------------------------------------------------------
# I/O + lifecycle helper branches
# ---------------------------------------------------------------------------


async def test_flush_stdin_noop_when_no_proc() -> None:
    mgr = _manager()
    mgr._proc = None
    await mgr._flush_stdin()  # 318 false branch — no error


async def test_ensure_alive_noop_when_alive() -> None:
    mgr = _manager()
    mgr._proc = _FakeProc(returncode=None)
    called: list[int] = []

    async def _spawn() -> None:
        called.append(1)

    mgr._spawn = _spawn  # type: ignore[method-assign]
    await mgr._ensure_alive()
    assert called == []


async def test_spawn_noop_when_already_alive() -> None:
    mgr = _manager()
    proc = _FakeProc(returncode=None)
    mgr._proc = proc
    await mgr._spawn()  # early return — must not replace the live process
    assert mgr._proc is proc


async def test_kill_noop_when_no_proc() -> None:
    mgr = _manager()
    mgr._proc = None
    await mgr._kill()
    assert mgr._proc is None


async def test_kill_suppresses_shutdown_write_error() -> None:
    mgr = CliSubprocessManager("claude", "1.0.0", health_interval_s=9999, sigkill_grace_s=0.5)
    proc = _FakeProc(returncode=None)
    mgr._proc = proc

    def _boom(_line: str) -> None:
        raise RuntimeError("stdin broken")

    mgr._write_line = _boom  # type: ignore[method-assign]
    await mgr._kill()
    assert proc.terminated
    assert mgr._proc is None


class _HangThenKillProc(_FakeProc):
    async def wait(self) -> int:
        if self.killed:
            return -9
        await asyncio.sleep(10)
        return 0  # pragma: no cover


async def test_kill_escalates_to_sigkill_on_grace_timeout() -> None:
    mgr = CliSubprocessManager("claude", "1.0.0", health_interval_s=9999, sigkill_grace_s=0.01)
    proc = _HangThenKillProc(returncode=None)
    mgr._proc = proc
    await mgr._kill()
    assert proc.terminated
    assert proc.killed
    assert mgr._proc is None


# ---------------------------------------------------------------------------
# start()/stop() — health task lifecycle
# ---------------------------------------------------------------------------


async def test_start_creates_health_task_and_stop_cancels_it() -> None:
    mgr = _manager()

    async def _spawn() -> None:
        mgr._proc = _FakeProc(returncode=None)

    mgr._spawn = _spawn  # type: ignore[method-assign]
    await mgr.start()
    assert mgr._health_task is not None
    await mgr.stop()
    assert mgr._health_task is None
    assert mgr._proc is None


async def test_start_keeps_existing_live_health_task() -> None:
    mgr = _manager()

    async def _spawn() -> None:
        mgr._proc = _FakeProc(returncode=None)

    mgr._spawn = _spawn  # type: ignore[method-assign]

    async def _idle() -> None:
        await asyncio.sleep(3600)

    existing = asyncio.create_task(_idle())
    mgr._health_task = existing
    await mgr.start()  # 269->exit: task already alive → not replaced
    assert mgr._health_task is existing
    existing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await existing


async def test_spawn_invokes_create_subprocess_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _manager()
    mgr._proc = None
    created = _FakeProc(returncode=None)
    captured: list[tuple[Any, ...]] = []

    async def _fake_exec(*args: Any, **_kw: Any) -> _FakeProc:
        captured.append(args)
        return created

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    await mgr._spawn()
    assert mgr._proc is created
    assert captured[0][0] == "claude"
    assert "--server" in captured[0]


# ---------------------------------------------------------------------------
# _health_loop — skip-when-active vs run-check, then cancel
# ---------------------------------------------------------------------------


async def test_health_loop_skips_when_call_active_then_runs_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = CliSubprocessManager("claude", "1.0.0", health_interval_s=5)
    checks: list[int] = []

    async def _check() -> None:
        checks.append(1)

    mgr._do_health_check = _check  # type: ignore[method-assign]

    state = {"i": 0}

    async def _fake_sleep(_secs: float) -> None:
        state["i"] += 1
        if state["i"] == 1:
            mgr._call_active = True  # first cycle: skip
        elif state["i"] == 2:
            mgr._call_active = False  # second cycle: run check
        else:
            raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await mgr._health_loop()
    assert checks == [1]


# ---------------------------------------------------------------------------
# _do_health_check — pong-mismatch and empty-read both respawn
# ---------------------------------------------------------------------------


async def test_health_check_respawns_when_no_pong_line() -> None:
    mgr = CliSubprocessManager("claude", "1.0.0", health_interval_s=9999, health_timeout_s=1)
    mgr._proc = _FakeProc(lines=[])  # readline → b"" (falsy)
    spawned: list[int] = []

    async def _ks() -> None:
        spawned.append(1)

    mgr._kill_and_spawn = _ks  # type: ignore[method-assign]
    await mgr._do_health_check()
    assert spawned == [1]


async def test_health_check_respawns_on_non_pong_message() -> None:
    mgr = CliSubprocessManager("claude", "1.0.0", health_interval_s=9999, health_timeout_s=1)
    mgr._proc = _FakeProc(lines=[_line({"type": "garbage", "id": "x"})])
    spawned: list[int] = []

    async def _ks() -> None:
        spawned.append(1)

    mgr._kill_and_spawn = _ks  # type: ignore[method-assign]
    await mgr._do_health_check()
    assert spawned == [1]


async def test_read_events_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = CliSubprocessManager("claude", "1.0.0", health_interval_s=9999, call_timeout_s=0.01)
    mgr._proc = _FakeProc(lines=[])

    async def _slow_readline() -> bytes:
        await asyncio.sleep(10)
        return b""  # pragma: no cover

    mgr._proc.stdout.readline = _slow_readline  # type: ignore[method-assign]
    with pytest.raises(CliCallTimeoutError):
        async for _ in mgr._read_events("cid"):
            pass
