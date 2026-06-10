"""
Tests for AgentResponder (_agent_responder) — inbound message -> LLM reply.

Covers: persona + agent-tool-context loading, reply generation (with tool
metadata), and dispatch (full flow, app unbound, model failure fallback, empty
reply, inbound/outbound error auditing, persona + tool-context propagation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meridiand._agent_responder import _FALLBACK_REPLY, AgentResponder

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Event:
    def __init__(self, text: str) -> None:
        self.type = "text_delta"
        self.text = text


class _FakeRouter:
    def __init__(self, *, text: str = "reply", raise_exc: Exception | None = None) -> None:
        self._text = text
        self._raise = raise_exc
        self.last_opts: Any = None

    async def call(self, opts: Any) -> Any:
        self.last_opts = opts
        if self._raise is not None:
            raise self._raise
        if self._text:
            yield _Event(self._text)


class _RecordingAudit:
    def __init__(self) -> None:
        self.entries: list[Any] = []

    def write(self, entry: Any) -> None:
        self.entries.append(entry)


class _FakeApp:
    """Raw ASGI app for the inbound/outbound routes the responder calls."""

    def __init__(
        self,
        *,
        inbound_raise: bool = False,
        outbound_raise: bool = False,
        session_id: str = "sess_1",
    ) -> None:
        self.inbound_raise = inbound_raise
        self.outbound_raise = outbound_raise
        self.session_id = session_id
        self.posts: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        path = scope["path"]
        self.posts.append((path, json.loads(body or b"{}")))
        if path.endswith("/inbound"):
            if self.inbound_raise:
                raise RuntimeError("inbound boom")
            payload = json.dumps({"session_id": self.session_id}).encode()
        else:
            if self.outbound_raise:
                raise RuntimeError("outbound boom")
            payload = b'{"delivered": true}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def _seed(root: Path, *, agent_id: str | None = "agent_t", tools: list[str] | None = None) -> None:
    (root / "channels").mkdir(parents=True, exist_ok=True)
    channel: dict[str, Any] = {"id": "ch1", "kind": "meridian.telegram"}
    if agent_id is not None:
        channel["default_agent_id"] = agent_id
        (root / "agents").mkdir(parents=True, exist_ok=True)
        (root / "environments").mkdir(parents=True, exist_ok=True)
        ws = root / "ws"
        ws.mkdir(exist_ok=True)
        (root / "environments" / "env_t.json").write_text(
            json.dumps({"id": "env_t", "workspace_path": str(ws)})
        )
        (root / "agents" / f"{agent_id}.json").write_text(
            json.dumps(
                {
                    "id": agent_id,
                    "default_environment_id": "env_t",
                    "version": {
                        "instructions": "Be a terse assistant.",
                        "tools": [{"name": t} for t in (tools if tools is not None else ["read"])],
                        "capabilities": [f"fs.read[{ws}/**]"],
                    },
                }
            )
        )
    (root / "channels" / "ch1.json").write_text(json.dumps(channel))


def _responder(root: Path | None, router: Any = None, audit: Any = None) -> AgentResponder:
    return AgentResponder(
        model_router=router or _FakeRouter(),
        model="claude:claude-opus-4-7",
        storage_root=root,
        audit_log=audit,
    )


# ---------------------------------------------------------------------------
# Persona + agent context loading
# ---------------------------------------------------------------------------


class TestLoadPersona:
    def test_default_when_no_storage(self) -> None:
        r = _responder(None)
        assert r._load_persona("ch1") == r._system_prompt

    def test_default_when_channel_missing(self, tmp_path: Path) -> None:
        r = _responder(tmp_path)
        assert r._load_persona("ch1") == r._system_prompt

    def test_default_when_no_agent_id(self, tmp_path: Path) -> None:
        _seed(tmp_path, agent_id=None)
        r = _responder(tmp_path)
        assert r._load_persona("ch1") == r._system_prompt

    def test_returns_agent_instructions(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        r = _responder(tmp_path)
        assert r._load_persona("ch1") == "Be a terse assistant."


class TestLoadAgentContext:
    def test_none_without_storage(self) -> None:
        assert _responder(None)._load_agent_context("ch1") is None

    def test_none_without_agent(self, tmp_path: Path) -> None:
        _seed(tmp_path, agent_id=None)
        assert _responder(tmp_path)._load_agent_context("ch1") is None

    def test_none_when_no_tools(self, tmp_path: Path) -> None:
        _seed(tmp_path, tools=[])
        assert _responder(tmp_path)._load_agent_context("ch1") is None

    def test_full_context_with_workspace(self, tmp_path: Path) -> None:
        _seed(tmp_path, tools=["read", "write"])
        ctx = _responder(tmp_path)._load_agent_context("ch1")
        assert ctx is not None
        assert ctx["agent_id"] == "agent_t"
        assert ctx["tools"] == ["read", "write"]
        assert ctx["workspace"].endswith("/ws")

    def test_none_on_malformed_agent_record(self, tmp_path: Path) -> None:
        (tmp_path / "channels").mkdir(parents=True, exist_ok=True)
        (tmp_path / "channels" / "ch1.json").write_text(
            json.dumps({"id": "ch1", "default_agent_id": "agent_bad"})
        )
        (tmp_path / "agents").mkdir(parents=True, exist_ok=True)
        (tmp_path / "agents" / "agent_bad.json").write_text("{ not valid json")
        assert _responder(tmp_path)._load_agent_context("ch1") is None


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    async def test_noop_when_unbound(self, tmp_path: Path) -> None:
        await _responder(tmp_path).dispatch(
            channel_id="ch1", sender_id="u", content="hi", content_type="text/plain"
        )  # no app -> returns silently

    async def test_full_flow_delivers_reply(self, tmp_path: Path) -> None:
        _seed(tmp_path, tools=["read"])
        router = _FakeRouter(text="hello back")
        app = _FakeApp()
        r = _responder(tmp_path, router=router)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        outbound = [p for p in app.posts if p[0].endswith("/outbound")][0]
        assert outbound[1]["content"] == "hello back"
        assert outbound[1]["session_id"] == "sess_1"
        # persona + tool context propagated to the model call
        assert router.last_opts.system == "Be a terse assistant."
        assert router.last_opts.metadata["meridian_tools"]["agent_id"] == "agent_t"

    async def test_model_failure_sends_fallback(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        audit = _RecordingAudit()
        router = _FakeRouter(raise_exc=RuntimeError("rate limit"))
        app = _FakeApp()
        r = _responder(tmp_path, router=router, audit=audit)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        outbound = [p for p in app.posts if p[0].endswith("/outbound")][0]
        assert outbound[1]["content"] == _FALLBACK_REPLY
        assert any(e.detail.get("stage") == "model" for e in audit.entries)

    async def test_empty_reply_becomes_fallback(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        app = _FakeApp()
        r = _responder(tmp_path, router=_FakeRouter(text=""))
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        outbound = [p for p in app.posts if p[0].endswith("/outbound")][0]
        assert outbound[1]["content"] == _FALLBACK_REPLY

    async def test_inbound_error_is_audited_and_still_replies(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        audit = _RecordingAudit()
        app = _FakeApp(inbound_raise=True)
        r = _responder(tmp_path, router=_FakeRouter(text="ok"), audit=audit)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert any(e.detail.get("stage") == "inbound" for e in audit.entries)
        assert any(p[0].endswith("/outbound") for p in app.posts)

    async def test_outbound_error_is_audited(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        audit = _RecordingAudit()
        app = _FakeApp(outbound_raise=True)
        r = _responder(tmp_path, router=_FakeRouter(text="ok"), audit=audit)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert any(e.detail.get("stage") == "outbound" for e in audit.entries)


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------


class TestConversationMemory:
    def _history(self, root: Path, channel_id: str = "ch1", sender_id: str = "u") -> list[Any]:
        path = root / "conversations" / channel_id / f"{sender_id}.json"
        return json.loads(path.read_text()) if path.exists() else []

    async def test_persists_user_and_assistant_turns(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        r = _responder(tmp_path, router=_FakeRouter(text="hello"))
        r.bind(_FakeApp())
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        hist = self._history(tmp_path)
        assert [m["role"] for m in hist] == ["user", "assistant"]
        assert hist[0]["content"] == "hi"
        assert hist[1]["content"] == "hello"

    async def test_prior_turns_passed_to_next_call(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        router = _FakeRouter(text="ack")
        r = _responder(tmp_path, router=router)
        r.bind(_FakeApp())
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="first", content_type="text/plain"
        )
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="second", content_type="text/plain"
        )
        msgs = router.last_opts.messages  # ModelCallOpts coerces dicts to Message objects
        assert [m.role for m in msgs] == ["user", "assistant", "user"]
        assert msgs[0].content == "first"
        assert msgs[2].content == "second"

    async def test_history_trimmed(self, tmp_path: Path) -> None:
        from meridiand._agent_responder import _MAX_HISTORY_MESSAGES

        _seed(tmp_path)
        r = _responder(tmp_path, router=_FakeRouter(text="x"))
        r.bind(_FakeApp())
        for i in range(_MAX_HISTORY_MESSAGES):  # each dispatch adds 2 messages
            await r.dispatch(
                channel_id="ch1", sender_id="u", content=f"m{i}", content_type="text/plain"
            )
        assert len(self._history(tmp_path)) == _MAX_HISTORY_MESSAGES

    async def test_model_failure_keeps_user_turn_only(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        r = _responder(tmp_path, router=_FakeRouter(raise_exc=RuntimeError("down")))
        r.bind(_FakeApp())
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        hist = self._history(tmp_path)
        assert [m["role"] for m in hist] == ["user"]

    async def test_corrupt_history_starts_fresh(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        conv = tmp_path / "conversations" / "ch1"
        conv.mkdir(parents=True)
        (conv / "u.json").write_text("{ not a list")
        r = _responder(tmp_path, router=_FakeRouter(text="ok"))
        r.bind(_FakeApp())
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert [m["role"] for m in self._history(tmp_path)] == ["user", "assistant"]

    def test_no_history_without_storage(self) -> None:
        r = _responder(None)
        assert r._load_history("ch1", "u") == []
        r._save_history("ch1", "u", [{"role": "user", "content": "x"}])  # no-op, no error

    async def test_save_failure_is_audited(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        # Make the per-sender history dir path collide with a file so mkdir fails.
        (tmp_path / "conversations").mkdir()
        (tmp_path / "conversations" / "ch1").write_text("not a dir")
        audit = _RecordingAudit()
        r = _responder(tmp_path, router=_FakeRouter(text="ok"), audit=audit)
        r.bind(_FakeApp())
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert any(e.detail.get("stage") == "memory" for e in audit.entries)
