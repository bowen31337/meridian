"""
Tests for the headless one-shot CliSubprocessManager (claude -p mode).

Covers:
  - _build_prompt: single- and multi-message serialization, block flattening.
  - _map_model: opus/sonnet/haiku aliases, default (None) for others.
  - _build_args: base argv, Contract 1 disallowed built-ins, --model,
    --append-system-prompt, and the MCP tool-bridge wiring (--mcp-config,
    --allowed-tools, cwd) when opts.metadata carries meridian_tools.
  - call(): success event stream, empty text, usage/model extraction,
    non-zero exit, invalid JSON, is_error result, timeout, and cwd passthrough.
  - start()/stop() are no-ops.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from meridian_sdk_provider.types import Message, ModelCallOpts

from meridian_provider_claude_code_oauth import _subprocess
from meridian_provider_claude_code_oauth._disallowed_tools import ALL_CLAUDE_CODE_BUILTIN_TOOLS
from meridian_provider_claude_code_oauth._subprocess import (
    CliCallTimeoutError,
    CliSubprocessError,
    CliSubprocessManager,
    _build_prompt,
    _flatten_content,
    _map_model,
)


def _mgr(call_timeout_s: float = 120.0) -> CliSubprocessManager:
    return CliSubprocessManager("claude", "v1", call_timeout_s=call_timeout_s)


def _opts(
    *,
    model: str = "claude-sonnet-4-6",
    content: str = "hi",
    system: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelCallOpts:
    return ModelCallOpts(
        model=model,
        messages=[Message(role="user", content=content)],
        system=system,
        metadata=metadata or {},
    )


def _result_json(
    *,
    text: str = "hello",
    inp: int = 5,
    out: int = 6,
    stop: str = "end_turn",
    model: str = "claude-opus-4-7",
    is_error: bool = False,
) -> bytes:
    return json.dumps(
        {
            "type": "result",
            "is_error": is_error,
            "result": text,
            "stop_reason": stop,
            "usage": {"input_tokens": inp, "output_tokens": out},
            "modelUsage": {model: {}},
        }
    ).encode()


class _FakeProc:
    def __init__(
        self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0, hang: bool = False
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(10)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


def _patch_exec(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc, captured: dict[str, Any]) -> None:
    async def _fake(*args: Any, **kwargs: Any) -> _FakeProc:
        captured["args"] = list(args)
        captured["cwd"] = kwargs.get("cwd")
        return proc

    monkeypatch.setattr(_subprocess.asyncio, "create_subprocess_exec", _fake)


async def _collect(mgr: CliSubprocessManager, opts: ModelCallOpts) -> list[Any]:
    return [ev async for ev in mgr.call(opts)]


# ---------------------------------------------------------------------------
# Prompt + model mapping
# ---------------------------------------------------------------------------


class TestPromptAndModel:
    def test_single_message_prompt_is_content(self) -> None:
        assert _build_prompt(_opts(content="hello there")) == "hello there"

    def test_multi_message_prompt_is_role_tagged(self) -> None:
        opts = ModelCallOpts(
            model="m",
            messages=[Message(role="user", content="a"), Message(role="assistant", content="b")],
        )
        assert _build_prompt(opts) == "user: a\n\nassistant: b"

    def test_flatten_content_str(self) -> None:
        assert _flatten_content("plain") == "plain"

    def test_flatten_content_blocks(self) -> None:
        block = type("B", (), {"text": "block text"})()
        assert _flatten_content([block]) == "block text"

    def test_map_model_aliases(self) -> None:
        assert _map_model("claude-opus-4-7") == "opus"
        assert _map_model("claude:claude-sonnet-4-6") == "sonnet"
        assert _map_model("claude-haiku-4-5") == "haiku"

    def test_map_model_unknown_is_none(self) -> None:
        assert _map_model("gpt-4o") is None
        assert _map_model(None) is None


# ---------------------------------------------------------------------------
# _build_args
# ---------------------------------------------------------------------------


class TestBuildArgs:
    def test_base_argv_no_tools(self) -> None:
        args, cwd = _mgr()._build_args(_opts(system="be brief"))
        assert args[:5] == ["claude", "-p", "hi", "--output-format", "json"]
        assert "--disallowed-tools" in args
        assert "Bash" in args  # Contract 1: built-ins disabled
        assert "--model" in args and "sonnet" in args
        assert "--append-system-prompt" in args and "be brief" in args
        assert "--mcp-config" not in args
        assert cwd is None

    def test_no_system_prompt_flag_when_absent(self) -> None:
        args, _ = _mgr()._build_args(_opts(system=None))
        assert "--append-system-prompt" not in args

    def test_unknown_model_omits_model_flag(self) -> None:
        args, _ = _mgr()._build_args(_opts(model="gpt-4o"))
        assert "--model" not in args

    def test_tool_metadata_adds_mcp_bridge(self) -> None:
        meta = {
            "meridian_tools": {
                "agent_id": "agent_X",
                "storage_root": "/root",
                "tools": ["exec", "read"],
                "workspace": "/ws",
            }
        }
        args, cwd = _mgr()._build_args(_opts(metadata=meta))
        assert "--mcp-config" in args
        cfg = json.loads(args[args.index("--mcp-config") + 1])
        bridge = cfg["mcpServers"]["meridian"]
        assert bridge["args"][:2] == ["-m", "meridiand._agent_tool_server"]
        assert "agent_X" in bridge["args"] and "/root" in bridge["args"]
        assert "mcp__meridian__exec" in args
        assert "mcp__meridian__read" in args
        assert "Bash" in args  # built-ins still disabled alongside MCP tools
        assert cwd == "/ws"

    def test_empty_tools_list_is_no_bridge(self) -> None:
        meta = {"meridian_tools": {"agent_id": "a", "storage_root": "/r", "tools": []}}
        args, cwd = _mgr()._build_args(_opts(metadata=meta))
        assert "--mcp-config" not in args
        assert cwd is None

    def test_extra_dirs_become_add_dir_flags(self) -> None:
        meta = {
            "meridian_tools": {
                "agent_id": "a",
                "storage_root": "/r",
                "tools": ["read"],
                "workspace": "/ws",
                "extra_dirs": ["/Users/bob/dev", "/data"],
            }
        }
        args, _ = _mgr()._build_args(_opts(metadata=meta))
        # each granted dir is passed to the CLI as --add-dir <dir>
        pairs = [(args[i], args[i + 1]) for i, a in enumerate(args[:-1]) if a == "--add-dir"]
        assert ("--add-dir", "/Users/bob/dev") in pairs
        assert ("--add-dir", "/data") in pairs

    def test_no_add_dir_without_extra_dirs(self) -> None:
        meta = {
            "meridian_tools": {
                "agent_id": "a",
                "storage_root": "/r",
                "tools": ["read"],
                "workspace": "/ws",
            }
        }
        args, _ = _mgr()._build_args(_opts(metadata=meta))
        assert "--add-dir" not in args

    def test_native_web_tools_allowed_by_name(self) -> None:
        # web_search / web_fetch map to the CLI's native tools, not the MCP bridge.
        meta = {
            "meridian_tools": {
                "agent_id": "a",
                "storage_root": "/r",
                "tools": ["read", "web_search", "web_fetch"],
                "workspace": "/ws",
            }
        }
        args, cwd = _mgr()._build_args(_opts(metadata=meta))
        assert "WebSearch" in args
        assert "WebFetch" in args
        assert "mcp__meridian__web_search" not in args  # not bridged
        assert "mcp__meridian__read" in args  # ordinary tool still bridged
        assert "--mcp-config" in args  # bridge present for the non-web tool
        assert cwd == "/ws"

    def test_only_web_tools_needs_no_bridge(self) -> None:
        meta = {
            "meridian_tools": {
                "agent_id": "a",
                "storage_root": "/r",
                "tools": ["web_search"],
            }
        }
        args, cwd = _mgr()._build_args(_opts(metadata=meta))
        assert "--mcp-config" not in args  # nothing to bridge
        assert "--allowed-tools" in args
        assert "WebSearch" in args
        assert cwd is None
        assert "Bash" in args  # Contract 1 builtins still disabled


# ---------------------------------------------------------------------------
# call()
# ---------------------------------------------------------------------------


class TestCall:
    async def test_success_event_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        _patch_exec(monkeypatch, _FakeProc(stdout=_result_json(text="pong")), captured)
        events = await _collect(_mgr(), _opts())
        assert [e.type for e in events] == ["message_start", "text_delta", "message_stop"]
        assert events[0].model == "claude-opus-4-7"
        assert events[0].input_tokens == 5
        assert events[1].text == "pong"
        assert events[2].output_tokens == 6
        assert events[2].stop_reason == "end_turn"

    async def test_empty_text_yields_no_delta(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_exec(monkeypatch, _FakeProc(stdout=_result_json(text="")), {})
        events = await _collect(_mgr(), _opts())
        assert [e.type for e in events] == ["message_start", "message_stop"]

    async def test_model_falls_back_to_opts_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.dumps({"result": "x", "usage": {}, "stop_reason": "end_turn"}).encode()
        _patch_exec(monkeypatch, _FakeProc(stdout=payload), {})
        events = await _collect(_mgr(), _opts(model="claude-sonnet-4-6"))
        assert events[0].model == "claude-sonnet-4-6"

    async def test_nonzero_exit_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_exec(monkeypatch, _FakeProc(stderr=b"boom", returncode=2), {})
        with pytest.raises(CliSubprocessError, match="exited with code 2"):
            await _collect(_mgr(), _opts())

    async def test_invalid_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_exec(monkeypatch, _FakeProc(stdout=b"not json"), {})
        with pytest.raises(CliSubprocessError, match="invalid JSON"):
            await _collect(_mgr(), _opts())

    async def test_is_error_result_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_exec(monkeypatch, _FakeProc(stdout=_result_json(text="nope", is_error=True)), {})
        with pytest.raises(CliSubprocessError, match="reported an error"):
            await _collect(_mgr(), _opts())

    async def test_timeout_raises_and_kills(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = _FakeProc(hang=True)
        _patch_exec(monkeypatch, proc, {})
        with pytest.raises(CliCallTimeoutError):
            await _collect(_mgr(call_timeout_s=0.01), _opts())
        assert proc.killed is True

    async def test_cwd_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        _patch_exec(monkeypatch, _FakeProc(stdout=_result_json()), captured)
        meta = {
            "meridian_tools": {
                "agent_id": "a",
                "storage_root": "/r",
                "tools": ["exec"],
                "workspace": "/ws",
            }
        }
        await _collect(_mgr(), _opts(metadata=meta))
        assert captured["cwd"] == "/ws"

    async def test_cancellation_kills_and_reraises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = _FakeProc(hang=True)
        _patch_exec(monkeypatch, proc, {})
        agen = _mgr().call(_opts())
        task = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.05)  # let the call reach communicate()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert proc.killed is True


# ---------------------------------------------------------------------------
# Lifecycle no-ops + Contract 1 constant
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_stop_are_noops(self) -> None:
        mgr = _mgr()
        await mgr.start()
        await mgr.stop()

    def test_disallow_constant_has_builtins(self) -> None:
        assert {"Bash", "Read", "Write", "Edit"} <= set(ALL_CLAUDE_CODE_BUILTIN_TOOLS)

    def test_disallowed_tool_error_constructs(self) -> None:
        from meridian_provider_claude_code_oauth._subprocess import DisallowedToolError

        err = DisallowedToolError("nope")
        assert isinstance(err, CliSubprocessError)
        assert "nope" in str(err)
