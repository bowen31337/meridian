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
    def __init__(
        self,
        *,
        text: str = "reply",
        facts_json: str | None = None,
        raise_exc: Exception | None = None,
        extract_raise: bool = False,
    ) -> None:
        self._text = text
        self._facts = facts_json
        self._raise = raise_exc
        self._extract_raise = extract_raise
        self.last_opts: Any = None
        self.last_reply_opts: Any = None
        self.calls: list[Any] = []

    async def call(self, opts: Any) -> Any:
        self.last_opts = opts
        self.calls.append(opts)
        is_extract = "haiku" in (opts.model or "")
        if not is_extract:
            self.last_reply_opts = opts
        if self._raise is not None or (is_extract and self._extract_raise):
            raise self._raise or RuntimeError("extract boom")
        text = self._facts if (is_extract and self._facts is not None) else self._text
        if text:
            yield _Event(text)


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
        inbound_status: int = 200,
        quarantined: bool = False,
        outbound_raise: bool = False,
        memory_raise: bool = False,
        write_raise: bool = False,
        memories: list[str] | None = None,
        session_id: str = "sess_1",
    ) -> None:
        self.inbound_raise = inbound_raise
        self.inbound_status = inbound_status
        self.quarantined = quarantined
        self.outbound_raise = outbound_raise
        self.memory_raise = memory_raise
        self.write_raise = write_raise
        self.memories = memories or []
        self.session_id = session_id
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.mem_writes: list[dict[str, Any]] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        path = scope["path"]
        parsed = json.loads(body or b"{}")
        self.posts.append((path, parsed))
        status = 200
        if path.endswith("/inbound"):
            if self.inbound_raise:
                raise RuntimeError("inbound boom")
            status = self.inbound_status
            payload = json.dumps(
                {"session_id": self.session_id, "quarantined": self.quarantined}
            ).encode()
        elif path.endswith("/query_runs"):
            if self.memory_raise:
                raise RuntimeError("memory boom")
            payload = json.dumps({"results": [{"content": m} for m in self.memories]}).encode()
        elif path.endswith("/write"):
            if self.write_raise:
                raise RuntimeError("write boom")
            self.mem_writes.append(parsed)
            payload = b'{"id": "mem_1"}'
        else:
            if self.outbound_raise:
                raise RuntimeError("outbound boom")
            payload = b'{"delivered": true}'
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def _seed(
    root: Path,
    *,
    agent_id: str | None = "agent_t",
    tools: list[str] | None = None,
    memory_store_id: str | None = None,
    caps: list[str] | None = None,
) -> None:
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
                        "capabilities": caps if caps is not None else [f"fs.read[{ws}/**]"],
                        "memory_store_refs": [memory_store_id] if memory_store_id else [],
                    },
                }
            )
        )
    (root / "channels" / "ch1.json").write_text(json.dumps(channel))


def _seed_skill(
    root: Path,
    *,
    skill_id: str = "skill_1",
    name: str = "Changelog Writer",
    instructions: str = "Write crisp changelogs.",
    agent_id: str = "agent_t",
    status: str = "active",
    version_id: str | None = None,
    version_instructions: str | None = None,
) -> None:
    """Create a skill record + (optionally pinned version) + an activation."""
    (root / "skills").mkdir(parents=True, exist_ok=True)
    (root / "skill_activations").mkdir(parents=True, exist_ok=True)
    (root / "skills" / f"{skill_id}.json").write_text(
        json.dumps(
            {
                "id": skill_id,
                "name": name,
                "description": "d",
                "version": {"id": f"{skill_id}_v1", "instructions": instructions},
            }
        )
    )
    if version_id is not None and version_instructions is not None:
        (root / "skill_versions").mkdir(parents=True, exist_ok=True)
        (root / "skill_versions" / f"{version_id}.json").write_text(
            json.dumps({"id": version_id, "instructions": version_instructions})
        )
    (root / "skill_activations" / f"act_{skill_id}.json").write_text(
        json.dumps(
            {
                "id": f"act_{skill_id}",
                "agent_id": agent_id,
                "skill_id": skill_id,
                "skill_version_id": version_id,
                "status": status,
                "requested_at": "2026-06-10T00:00:00+00:00",
            }
        )
    )


def _responder(
    root: Path | None,
    router: Any = None,
    audit: Any = None,
    extract_facts: bool = False,
    intelligent_routing: bool = False,
) -> AgentResponder:
    return AgentResponder(
        model_router=router or _FakeRouter(),
        model="claude:claude-opus-4-7",
        storage_root=root,
        audit_log=audit,
        extract_facts=extract_facts,
        intelligent_routing=intelligent_routing,
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

    def test_extra_dirs_excludes_workspace_includes_granted(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed(
            tmp_path,
            tools=["read"],
            caps=[
                f"fs.read[{ws}/**]",  # workspace -> not an extra dir
                "exec.shell",  # non-fs -> skipped
                "net.fetch[*]",  # non-fs -> skipped
                "garbage!!",  # unparseable -> skipped
                "fs.read[/opt/data/**]",  # granted root -> included (once)
                "fs.write[/opt/data/**]",  # same root -> deduped
            ],
        )
        ctx = _responder(tmp_path)._load_agent_context("ch1")
        assert ctx is not None
        assert ctx["extra_dirs"] == ["/opt/data"]

    def test_none_on_malformed_agent_record(self, tmp_path: Path) -> None:
        (tmp_path / "channels").mkdir(parents=True, exist_ok=True)
        (tmp_path / "channels" / "ch1.json").write_text(
            json.dumps({"id": "ch1", "default_agent_id": "agent_bad"})
        )
        (tmp_path / "agents").mkdir(parents=True, exist_ok=True)
        (tmp_path / "agents" / "agent_bad.json").write_text("{ not valid json")
        assert _responder(tmp_path)._load_agent_context("ch1") is None

    def test_web_tools_forwarded_with_net_fetch_grant(self, tmp_path: Path) -> None:
        _seed(
            tmp_path,
            tools=["read", "web_search", "web_fetch"],
            caps=["fs.read[/ws/**]", "net.fetch[*]"],
        )
        ctx = _responder(tmp_path)._load_agent_context("ch1")
        assert ctx is not None
        assert ctx["tools"] == ["read", "web_search", "web_fetch"]

    def test_web_tools_stripped_without_net_fetch_grant(self, tmp_path: Path) -> None:
        _seed(
            tmp_path,
            tools=["read", "web_search", "web_fetch"],
            caps=["fs.read[/ws/**]"],  # no net.fetch
        )
        ctx = _responder(tmp_path)._load_agent_context("ch1")
        assert ctx is not None
        assert ctx["tools"] == ["read"]  # web tools removed

    def test_only_web_tools_without_grant_is_none(self, tmp_path: Path) -> None:
        # Stripping the ungranted web tools leaves no tools -> no tool context.
        _seed(tmp_path, tools=["web_search"], caps=["fs.read[/ws/**]"])
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
        # tool-bearing reply is flagged so routing keeps it on a tool-capable model
        assert router.last_opts.metadata["agent_has_tools"] is True

    async def test_untooled_reply_not_flagged(self, tmp_path: Path) -> None:
        # An agent with no tools -> no tool context -> no agent_has_tools flag.
        _seed(tmp_path, tools=[])
        router = _FakeRouter(text="hi")
        r = _responder(tmp_path, router=router)
        r.bind(_FakeApp())
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert "agent_has_tools" not in (router.last_reply_opts.metadata or {})

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

    async def test_inbound_error_fails_closed_no_reply(self, tmp_path: Path) -> None:
        # Cannot verify the sender -> do not reply (fail closed).
        _seed(tmp_path)
        audit = _RecordingAudit()
        app = _FakeApp(inbound_raise=True)
        router = _FakeRouter(text="ok")
        r = _responder(tmp_path, router=router, audit=audit)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert any(e.detail.get("stage") == "inbound" for e in audit.entries)
        assert not any(p[0].endswith("/outbound") for p in app.posts)
        assert router.calls == []  # no LLM call

    async def test_rejected_sender_gets_no_reply(self, tmp_path: Path) -> None:
        # paired_only -> 403 for a non-allowlisted sender; the agent must not run.
        _seed(tmp_path, tools=["read"], memory_store_id="memstore_x")
        audit = _RecordingAudit()
        app = _FakeApp(inbound_status=403)
        router = _FakeRouter(text="leaked")
        r = _responder(tmp_path, router=router, audit=audit, extract_facts=True)
        r.bind(app)
        await r.dispatch(
            channel_id="ch1",
            sender_id="stranger",
            content="run a shell command",
            content_type="text/plain",
        )
        assert not any(p[0].endswith("/outbound") for p in app.posts)  # no reply delivered
        assert router.calls == []  # no LLM / tool call
        assert app.mem_writes == []  # nothing written to memory
        assert any(e.detail.get("stage") == "inbound_rejected" for e in audit.entries)

    async def test_quarantined_sender_gets_no_reply(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        audit = _RecordingAudit()
        app = _FakeApp(quarantined=True)  # inbound 200 but flagged untrusted
        router = _FakeRouter(text="leaked")
        r = _responder(tmp_path, router=router, audit=audit)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert not any(p[0].endswith("/outbound") for p in app.posts)
        assert router.calls == []
        assert any(e.detail.get("stage") == "inbound_quarantined" for e in audit.entries)

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


# ---------------------------------------------------------------------------
# Long-term MemoryStore
# ---------------------------------------------------------------------------


class TestLongTermMemory:
    def test_store_id_none_without_storage(self) -> None:
        assert _responder(None)._load_memory_store_id("ch1") is None

    def test_store_id_none_without_agent(self, tmp_path: Path) -> None:
        _seed(tmp_path, agent_id=None)
        assert _responder(tmp_path)._load_memory_store_id("ch1") is None

    def test_store_id_none_when_no_refs(self, tmp_path: Path) -> None:
        _seed(tmp_path)  # no memory_store_id
        assert _responder(tmp_path)._load_memory_store_id("ch1") is None

    def test_store_id_resolved_from_refs(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        assert _responder(tmp_path)._load_memory_store_id("ch1") == "memstore_x"

    async def test_retrieved_memories_injected_into_persona(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        router = _FakeRouter(text="ok")
        app = _FakeApp(memories=["Bowen prefers Rust", "Bowen is in Australia"])
        r = _responder(tmp_path, router=router)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        sys_prompt = router.last_opts.system
        assert "Bowen prefers Rust" in sys_prompt
        assert "Things you remember" in sys_prompt

    async def test_writes_memory_after_reply(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        app = _FakeApp(memories=[])
        r = _responder(tmp_path, router=_FakeRouter(text="ok"))
        r.bind(app)
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="I love Substrate", content_type="text/plain"
        )
        assert len(app.mem_writes) == 1
        assert app.mem_writes[0]["content"] == "I love Substrate"
        assert app.mem_writes[0]["key"].startswith("tg-")

    async def test_no_write_on_model_failure(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        app = _FakeApp(memories=[])
        r = _responder(tmp_path, router=_FakeRouter(raise_exc=RuntimeError("down")))
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="x", content_type="text/plain")
        assert app.mem_writes == []

    async def test_no_memory_calls_without_store(self, tmp_path: Path) -> None:
        _seed(tmp_path)  # agent has no memory_store_refs
        app = _FakeApp()
        r = _responder(tmp_path, router=_FakeRouter(text="ok"))
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert not any("/query_runs" in p[0] or "/write" in p[0] for p in app.posts)

    async def test_retrieval_failure_is_audited(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        audit = _RecordingAudit()
        app = _FakeApp(memory_raise=True)
        r = _responder(tmp_path, router=_FakeRouter(text="ok"), audit=audit)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert any(e.detail.get("stage") == "memory_retrieve" for e in audit.entries)

    async def test_write_failure_is_audited(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        audit = _RecordingAudit()
        app = _FakeApp(write_raise=True)
        r = _responder(tmp_path, router=_FakeRouter(text="ok"), audit=audit)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert any(e.detail.get("stage") == "memory_write" for e in audit.entries)

    def test_store_id_none_on_malformed_agent(self, tmp_path: Path) -> None:
        (tmp_path / "channels").mkdir(parents=True, exist_ok=True)
        (tmp_path / "channels" / "ch1.json").write_text(
            json.dumps({"id": "ch1", "default_agent_id": "agent_bad"})
        )
        (tmp_path / "agents").mkdir(parents=True, exist_ok=True)
        (tmp_path / "agents" / "agent_bad.json").write_text("{ broken")
        assert _responder(tmp_path)._load_memory_store_id("ch1") is None


# ---------------------------------------------------------------------------
# Fact extraction + dialectic reconciliation
# ---------------------------------------------------------------------------


class TestFactExtraction:
    def test_parse_facts_valid(self) -> None:
        r = _responder(None)
        assert r._parse_facts('["a", "b"]') == ["a", "b"]

    def test_parse_facts_with_surrounding_prose(self) -> None:
        r = _responder(None)
        assert r._parse_facts('Sure! ["x", "y"] done') == ["x", "y"]

    def test_parse_facts_non_json(self) -> None:
        r = _responder(None)
        assert r._parse_facts("no array here") == []

    def test_parse_facts_non_list(self) -> None:
        r = _responder(None)
        assert r._parse_facts('{"not": "a list"}') == []

    def test_parse_facts_invalid_json_in_brackets(self) -> None:
        r = _responder(None)
        assert r._parse_facts("[not, valid, json]") == []

    def test_parse_facts_filters_and_caps(self) -> None:
        r = AgentResponder(model_router=_FakeRouter(), model="m", max_facts=2)
        assert r._parse_facts('["a", "", 5, "b", "c"]') == ["a", "b"]

    async def test_extracts_and_dialectic_writes(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        router = _FakeRouter(
            text="ok", facts_json='["Bowen prefers Rust", "Bowen builds Meridian"]'
        )
        app = _FakeApp()
        r = _responder(tmp_path, router=router, extract_facts=True)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hey", content_type="text/plain")
        assert [w["content"] for w in app.mem_writes] == [
            "Bowen prefers Rust",
            "Bowen builds Meridian",
        ]
        assert all(w["dialectic"] is True for w in app.mem_writes)

    async def test_no_facts_no_writes(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        router = _FakeRouter(text="ok", facts_json="[]")
        app = _FakeApp()
        r = _responder(tmp_path, router=router, extract_facts=True)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert app.mem_writes == []

    async def test_extraction_failure_is_audited(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        audit = _RecordingAudit()
        router = _FakeRouter(text="ok", extract_raise=True)
        app = _FakeApp()
        r = _responder(tmp_path, router=router, audit=audit, extract_facts=True)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        assert any(e.detail.get("stage") == "memory_extract" for e in audit.entries)
        assert app.mem_writes == []

    async def test_reply_uses_default_model_not_extract_model(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        router = _FakeRouter(text="hello", facts_json="[]")
        app = _FakeApp()
        r = _responder(tmp_path, router=router, extract_facts=True)
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        # reply call used opus; extraction call used haiku
        assert "opus" in router.calls[0].model
        assert "haiku" in router.calls[1].model


# ---------------------------------------------------------------------------
# Slash commands (openclaw-style menu)
# ---------------------------------------------------------------------------


def _outbound(app: _FakeApp) -> list[str]:
    return [p[1]["content"] for p in app.posts if p[0].endswith("/outbound")]


class TestSlashCommands:
    async def test_help_lists_commands_without_llm(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        router = _FakeRouter(text="should-not-run")
        app = _FakeApp()
        r = _responder(tmp_path, router=router)
        r.bind(app)
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="/help", content_type="text/plain"
        )
        out = _outbound(app)
        assert len(out) == 1
        assert "/help" in out[0] and "/remember" in out[0]
        assert router.calls == []  # handled locally, no model call

    async def test_start_greets(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        app = _FakeApp()
        r = _responder(tmp_path, router=_FakeRouter())
        r.bind(app)
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="/start", content_type="text/plain"
        )
        assert "personal assistant" in _outbound(app)[0]

    async def test_new_without_storage_is_noop(self) -> None:
        # /new must not blow up when the responder has no storage root.
        app = _FakeApp()
        r = _responder(None, router=_FakeRouter())
        r.bind(app)
        await r.dispatch(channel_id="ch1", sender_id="u", content="/new", content_type="text/plain")
        assert "Fresh start" in _outbound(app)[0]

    async def test_command_with_botname_suffix(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        app = _FakeApp()
        r = _responder(tmp_path, router=_FakeRouter())
        r.bind(app)
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="/help@mybot", content_type="text/plain"
        )
        assert "/remember" in _outbound(app)[0]

    async def test_new_clears_history(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        r = _responder(tmp_path, router=_FakeRouter(text="hi"))
        r.bind(_FakeApp())
        # build up some history
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        hist_path = tmp_path / "conversations" / "ch1" / "u.json"
        assert hist_path.exists()
        # /new wipes it
        app2 = _FakeApp()
        r.bind(app2)
        await r.dispatch(channel_id="ch1", sender_id="u", content="/new", content_type="text/plain")
        assert not hist_path.exists()
        assert "Fresh start" in _outbound(app2)[0]

    async def test_whoami_returns_sender(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        app = _FakeApp()
        r = _responder(tmp_path, router=_FakeRouter())
        r.bind(app)
        await r.dispatch(
            channel_id="ch1", sender_id="2069029798", content="/whoami", content_type="text/plain"
        )
        assert "2069029798" in _outbound(app)[0]

    async def test_remember_writes_memory(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        app = _FakeApp()
        r = _responder(tmp_path, router=_FakeRouter())
        r.bind(app)
        await r.dispatch(
            channel_id="ch1",
            sender_id="u",
            content="/remember I live in Sydney",
            content_type="text/plain",
        )
        assert len(app.mem_writes) == 1
        assert app.mem_writes[0]["content"] == "I live in Sydney"
        assert app.mem_writes[0]["dialectic"] is True
        assert "I live in Sydney" in _outbound(app)[0]

    async def test_remember_without_args_prompts(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        app = _FakeApp()
        r = _responder(tmp_path, router=_FakeRouter())
        r.bind(app)
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="/remember", content_type="text/plain"
        )
        assert app.mem_writes == []
        assert "Tell me what to remember" in _outbound(app)[0]

    async def test_remember_without_store(self, tmp_path: Path) -> None:
        _seed(tmp_path)  # no memory_store_id
        app = _FakeApp()
        r = _responder(tmp_path, router=_FakeRouter())
        r.bind(app)
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="/remember x", content_type="text/plain"
        )
        assert app.mem_writes == []
        assert "memory store" in _outbound(app)[0]

    async def test_unknown_bare_command_nudges(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        router = _FakeRouter(text="should-not-run")
        app = _FakeApp()
        r = _responder(tmp_path, router=router)
        r.bind(app)
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="/bogus", content_type="text/plain"
        )
        assert "Unknown command" in _outbound(app)[0]
        assert router.calls == []

    async def test_slash_with_prose_goes_to_llm(self, tmp_path: Path) -> None:
        # A message that merely starts with "/" but has prose is a real question.
        _seed(tmp_path)
        router = _FakeRouter(text="that path is fine")
        app = _FakeApp()
        r = _responder(tmp_path, router=router)
        r.bind(app)
        await r.dispatch(
            channel_id="ch1",
            sender_id="u",
            content="/etc/hosts is broken, why?",
            content_type="text/plain",
        )
        assert len(router.calls) == 1  # routed to the model
        assert _outbound(app)[0] == "that path is fine"


# ---------------------------------------------------------------------------
# Quoted/replied text as nearest context
# ---------------------------------------------------------------------------


class TestQuoteContext:
    async def test_quote_folded_into_user_message(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        router = _FakeRouter(text="ok")
        r = _responder(tmp_path, router=router)
        r.bind(_FakeApp())
        await r.dispatch(
            channel_id="ch1",
            sender_id="u",
            content="what does this mean?",
            content_type="text/plain",
            quote="the deploy failed at step 3",
        )
        msg = router.last_opts.messages[-1]
        assert "the deploy failed at step 3" in msg.content
        assert "what does this mean?" in msg.content

    async def test_no_quote_leaves_message_plain(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        router = _FakeRouter(text="ok")
        r = _responder(tmp_path, router=router)
        r.bind(_FakeApp())
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="hello", content_type="text/plain"
        )
        assert router.last_opts.messages[-1].content == "hello"

    async def test_quoted_context_persisted_in_history(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        r = _responder(tmp_path, router=_FakeRouter(text="ok"))
        r.bind(_FakeApp())
        await r.dispatch(
            channel_id="ch1",
            sender_id="u",
            content="explain",
            content_type="text/plain",
            quote="some earlier note",
        )
        hist = json.loads((tmp_path / "conversations" / "ch1" / "u.json").read_text())
        assert "some earlier note" in hist[0]["content"]


# ---------------------------------------------------------------------------
# Active skills (Meridian Skill resource injection)
# ---------------------------------------------------------------------------


class TestActiveSkills:
    def test_active_skill_resolved(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        _seed_skill(tmp_path, name="Changelog Writer", instructions="Write crisp changelogs.")
        skills = _responder(tmp_path)._load_active_skills("ch1")
        assert skills == [("Changelog Writer", "Write crisp changelogs.")]

    def test_pending_skill_ignored(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        _seed_skill(tmp_path, status="pending")
        assert _responder(tmp_path)._load_active_skills("ch1") == []

    def test_revoked_skill_ignored(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        _seed_skill(tmp_path, status="revoked")
        assert _responder(tmp_path)._load_active_skills("ch1") == []

    def test_activation_for_other_agent_ignored(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        _seed_skill(tmp_path, agent_id="someone_else")
        assert _responder(tmp_path)._load_active_skills("ch1") == []

    def test_pinned_version_instructions_preferred(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        _seed_skill(
            tmp_path,
            instructions="old text",
            version_id="skillver_pinned",
            version_instructions="pinned version text",
        )
        skills = _responder(tmp_path)._load_active_skills("ch1")
        assert skills[0][1] == "pinned version text"

    def test_missing_pinned_version_falls_back_to_skill_record(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        # activation pins a version_id whose file does not exist -> use skill record
        _seed_skill(tmp_path, instructions="fallback text", version_id="skillver_gone")
        skills = _responder(tmp_path)._load_active_skills("ch1")
        assert skills[0][1] == "fallback text"

    def test_skill_without_instructions_skipped(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        _seed_skill(tmp_path, instructions="   ")  # blank
        assert _responder(tmp_path)._load_active_skills("ch1") == []

    def test_corrupt_activation_skipped(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        _seed_skill(tmp_path, skill_id="skill_ok", instructions="good")
        (tmp_path / "skill_activations" / "act_bad.json").write_text("{ not json")
        skills = _responder(tmp_path)._load_active_skills("ch1")
        assert skills == [("Changelog Writer", "good")]

    def test_no_activations_dir(self, tmp_path: Path) -> None:
        _seed(tmp_path)  # no skills seeded
        assert _responder(tmp_path)._load_active_skills("ch1") == []

    def test_no_storage_returns_empty(self) -> None:
        assert _responder(None)._load_active_skills("ch1") == []

    def test_no_agent_id_returns_empty(self, tmp_path: Path) -> None:
        _seed(tmp_path, agent_id=None)
        _seed_skill(tmp_path)
        assert _responder(tmp_path)._load_active_skills("ch1") == []

    def test_missing_channel_returns_empty(self, tmp_path: Path) -> None:
        # no channel file at all -> outer guard returns []
        assert _responder(tmp_path)._load_active_skills("ch1") == []

    def test_activation_without_skill_id_skipped(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        (tmp_path / "skill_activations").mkdir(parents=True, exist_ok=True)
        (tmp_path / "skill_activations" / "act_x.json").write_text(
            json.dumps({"id": "act_x", "agent_id": "agent_t", "status": "active"})
        )
        assert _responder(tmp_path)._load_active_skills("ch1") == []

    def test_pinned_version_only_no_skill_record(self, tmp_path: Path) -> None:
        # version file exists but skill record absent -> name falls back to skill_id
        _seed(tmp_path)
        (tmp_path / "skill_versions").mkdir(parents=True, exist_ok=True)
        (tmp_path / "skill_versions" / "skillver_x.json").write_text(
            json.dumps({"id": "skillver_x", "instructions": "ver only"})
        )
        (tmp_path / "skill_activations").mkdir(parents=True, exist_ok=True)
        (tmp_path / "skill_activations" / "act_y.json").write_text(
            json.dumps(
                {
                    "id": "act_y",
                    "agent_id": "agent_t",
                    "skill_id": "skill_y",
                    "skill_version_id": "skillver_x",
                    "status": "active",
                }
            )
        )
        skills = _responder(tmp_path)._load_active_skills("ch1")
        assert skills == [("skill_y", "ver only")]

    async def test_active_skill_injected_into_prompt(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        _seed_skill(tmp_path, name="Changelog Writer", instructions="Write crisp changelogs.")
        router = _FakeRouter(text="ok")
        r = _responder(tmp_path, router=router)
        r.bind(_FakeApp())
        await r.dispatch(channel_id="ch1", sender_id="u", content="hi", content_type="text/plain")
        system = router.last_opts.system
        assert "Active skills" in system
        assert "Changelog Writer" in system
        assert "Write crisp changelogs." in system


# ---------------------------------------------------------------------------
# Intelligent routing (capability-tier classification)
# ---------------------------------------------------------------------------


class TestIntelligentRoutingDispatch:
    async def test_disabled_sets_no_route_tier(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        router = _FakeRouter(text="ok")
        r = _responder(tmp_path, router=router, intelligent_routing=False)
        r.bind(_FakeApp())
        await r.dispatch(
            channel_id="ch1",
            sender_id="u",
            content="prove that the sum of the first n odd numbers is n squared",
            content_type="text/plain",
        )
        assert "route_tier" not in (router.last_reply_opts.metadata or {})

    async def test_enabled_tags_reply_with_scored_tier(self, tmp_path: Path) -> None:
        from meridiand._intelligent_router import classify_tier

        _seed(tmp_path)
        router = _FakeRouter(text="ok")
        r = _responder(tmp_path, router=router, intelligent_routing=True)
        r.bind(_FakeApp())
        content = "prove the theorem rigorously and derive each lemma step by step"
        await r.dispatch(
            channel_id="ch1", sender_id="u", content=content, content_type="text/plain"
        )
        tier = router.last_reply_opts.metadata.get("route_tier")
        assert tier == classify_tier(content) == "reasoning"  # no LLM call; deterministic

    async def test_simple_message_tagged_simple(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        router = _FakeRouter(text="hello")
        r = _responder(tmp_path, router=router, intelligent_routing=True)
        r.bind(_FakeApp())
        await r.dispatch(
            channel_id="ch1", sender_id="u", content="show me the status", content_type="text/plain"
        )
        assert router.last_reply_opts.metadata.get("route_tier") == "simple"


# ---------------------------------------------------------------------------
# run_prompt (system-triggered turn, e.g. cron)
# ---------------------------------------------------------------------------


class TestRunPrompt:
    async def test_returns_none_when_unbound(self, tmp_path: Path) -> None:
        r = _responder(tmp_path)  # never bound to an app
        assert await r.run_prompt("ch1", "do the thing") is None

    async def test_runs_turn_and_delivers_to_channel(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        router = _FakeRouter(text="cron reply")
        app = _FakeApp()
        r = _responder(tmp_path, router=router)
        r.bind(app)
        reply = await r.run_prompt("ch1", "summarize my day", session_id="sess_x")
        assert reply == "cron reply"
        # delivered via outbound with the supplied session id; no /inbound (no pairing)
        outs = [p for p in app.posts if p[0].endswith("/outbound")]
        assert outs and outs[0][1]["content"] == "cron reply"
        assert outs[0][1]["session_id"] == "sess_x"
        assert not any(p[0].endswith("/inbound") for p in app.posts)

    async def test_model_failure_delivers_fallback(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        router = _FakeRouter(raise_exc=RuntimeError("down"))
        r = _responder(tmp_path, router=router)
        r.bind(_FakeApp())
        assert await r.run_prompt("ch1", "do it") == _FALLBACK_REPLY

    async def test_uses_intelligent_routing_tier(self, tmp_path: Path) -> None:
        from meridiand._intelligent_router import classify_tier

        _seed(tmp_path)
        router = _FakeRouter(text="ok")
        r = _responder(tmp_path, router=router, intelligent_routing=True)
        r.bind(_FakeApp())
        content = "prove the theorem rigorously and derive each lemma step by step"
        await r.run_prompt("ch1", content)
        assert router.last_reply_opts.metadata.get("route_tier") == classify_tier(content)


class TestRunPromptCoverage:
    async def test_injects_skills_and_memory(self, tmp_path: Path) -> None:
        _seed(tmp_path, memory_store_id="memstore_x")
        _seed_skill(tmp_path, name="Changelog Writer", instructions="Write crisp changelogs.")
        router = _FakeRouter(text="ok")
        app = _FakeApp(memories=["likes brevity"])
        r = _responder(tmp_path, router=router)
        r.bind(app)
        await r.run_prompt("ch1", "go")
        system = router.last_reply_opts.system
        assert "Active skills" in system
        assert "Changelog Writer" in system
        assert "likes brevity" in system

    async def test_empty_reply_becomes_fallback(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        router = _FakeRouter(text="")  # yields nothing -> empty reply
        r = _responder(tmp_path, router=router)
        r.bind(_FakeApp())
        assert await r.run_prompt("ch1", "go") == _FALLBACK_REPLY
