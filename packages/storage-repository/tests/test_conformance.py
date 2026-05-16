"""
Repository conformance suite.

Covers RepositoryRuntime (via stub implementations) and SqliteRepositoryDriver
(via in-memory SQLite):

  RepositoryRuntime (stub driver):
    - save / get / delete / list success: span emitted, invocation event attached,
      no audit entries, correct data returned.
    - Store raises unexpected exception: wrapped in RepositoryFailure with correct
      code, cause preserved, audit entry written, span marked ERROR.
    - on_error callback invoked on every failure.
    - Span lifecycle: span ended on both success and failure paths.

  SqliteRepositoryDriver:
    - migrate() creates all tables.
    - save / get round-trip for all 12 resource types.
    - get missing id returns None.
    - delete removes the record (and is a no-op if already absent).
    - list with and without filters; pagination (limit/offset) honoured.
    - upsert (save twice with same id) updates the record.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from opentelemetry.trace import StatusCode
from storage_repository import (
    Agent,
    AgentFilter,
    AuditLogEntry,
    Channel,
    ChannelFilter,
    Environment,
    EnvironmentFilter,
    MemoryEntry,
    MemoryFilter,
    Message,
    MessageFilter,
    RepositoryFailure,
    RepositoryOptions,
    RepositoryRuntime,
    Session,
    SessionFilter,
    Skill,
    SkillFilter,
    SqliteRepositoryDriver,
    Thread,
    ThreadFilter,
    ToolCall,
    ToolCallFilter,
    UserProfile,
    UserProfileFilter,
    VaultEntry,
    VaultFilter,
    Webhook,
    WebhookFilter,
)
from storage_repository._contract import (
    AgentRepository,
    ChannelRepository,
    EnvironmentRepository,
    MemoryRepository,
    MessageRepository,
    SessionRepository,
    SkillRepository,
    ThreadRepository,
    ToolCallRepository,
    UserProfileRepository,
    VaultRepository,
    WebhookRepository,
)
from storage_repository._runtime import RepositoryDriver

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    return datetime.now(UTC).isoformat()


def make_agent(id: str = "a1") -> Agent:
    now = _ts()
    return Agent(
        id=id,
        kind="meridian.research",
        name="Test Agent",
        config="{}",
        capabilities="[]",
        created_at=now,
        updated_at=now,
    )


def make_session(id: str = "s1", agent_id: str = "a1") -> Session:
    now = _ts()
    return Session(
        id=id, agent_id=agent_id, status="active", metadata=None, created_at=now, updated_at=now
    )


def make_thread(id: str = "t1", session_id: str = "s1") -> Thread:
    now = _ts()
    return Thread(id=id, session_id=session_id, title="Hello", created_at=now, updated_at=now)


def make_message(id: str = "m1", thread_id: str = "t1", session_id: str = "s1") -> Message:
    return Message(
        id=id,
        thread_id=thread_id,
        session_id=session_id,
        role="user",
        content="[]",
        sequence=0,
        created_at=_ts(),
    )


def make_tool_call(id: str = "tc1", message_id: str = "m1", session_id: str = "s1") -> ToolCall:
    now = _ts()
    return ToolCall(
        id=id,
        message_id=message_id,
        session_id=session_id,
        tool_name="search",
        input="{}",
        output=None,
        status="pending",
        created_at=now,
        updated_at=now,
    )


def make_skill(id: str = "sk1") -> Skill:
    now = _ts()
    return Skill(
        id=id,
        name="web_search",
        description="Search the web",
        capabilities='["net.fetch"]',
        config="{}",
        created_at=now,
        updated_at=now,
    )


def make_environment(id: str = "e1") -> Environment:
    now = _ts()
    return Environment(
        id=id,
        kind="meridian.python",
        status="provisioned",
        config="{}",
        created_at=now,
        updated_at=now,
    )


def make_memory_entry(id: str = "me1", scope: str = "agent:a1", key: str = "k") -> MemoryEntry:
    now = _ts()
    return MemoryEntry(id=id, scope=scope, key=key, value="val", created_at=now, updated_at=now)


def make_vault_entry(id: str = "v1") -> VaultEntry:
    now = _ts()
    return VaultEntry(
        id=id, name="openai/api_key", description="OpenAI key", created_at=now, updated_at=now
    )


def make_user_profile(id: str = "u1") -> UserProfile:
    now = _ts()
    return UserProfile(
        id=id,
        username="alice",
        display_name="Alice",
        email="alice@example.com",
        metadata=None,
        created_at=now,
        updated_at=now,
    )


def make_channel(id: str = "ch1") -> Channel:
    now = _ts()
    return Channel(
        id=id,
        kind="meridian.slack",
        name="general",
        config="{}",
        status="active",
        created_at=now,
        updated_at=now,
    )


def make_webhook(id: str = "wh1") -> Webhook:
    now = _ts()
    return Webhook(
        id=id,
        url="https://example.com/hook",
        events='["message.created"]',
        secret_ref=None,
        status="active",
        created_at=now,
        updated_at=now,
    )


def make_options(
    audit: CapturingAuditLog,
    errors: list[RepositoryFailure] | None = None,
) -> RepositoryOptions:
    return RepositoryOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


# ---------------------------------------------------------------------------
# Stub repositories for runtime tests
# ---------------------------------------------------------------------------


class _StubAgentRepo(AgentRepository):
    def __init__(self, *, raises: Exception | None = None) -> None:
        self._data: dict[str, Agent] = {}
        self._raises = raises

    async def get(self, agent_id: str) -> Agent | None:
        if self._raises:
            raise self._raises
        return self._data.get(agent_id)

    async def save(self, agent: Agent) -> None:
        if self._raises:
            raise self._raises
        self._data[agent.id] = agent

    async def delete(self, agent_id: str) -> None:
        if self._raises:
            raise self._raises
        self._data.pop(agent_id, None)

    async def list(self, filter: AgentFilter) -> list[Agent]:
        if self._raises:
            raise self._raises
        return list(self._data.values())


class _StubDriver(RepositoryDriver):
    def __init__(self, *, raises: Exception | None = None) -> None:
        self._agents = _StubAgentRepo(raises=raises)

    @property
    def agents(self) -> AgentRepository:
        return self._agents

    # Unused repositories — return stubs that always return empty/None
    @property
    def sessions(self) -> SessionRepository:
        raise NotImplementedError

    @property
    def threads(self) -> ThreadRepository:
        raise NotImplementedError

    @property
    def messages(self) -> MessageRepository:
        raise NotImplementedError

    @property
    def tool_calls(self) -> ToolCallRepository:
        raise NotImplementedError

    @property
    def skills(self) -> SkillRepository:
        raise NotImplementedError

    @property
    def environments(self) -> EnvironmentRepository:
        raise NotImplementedError

    @property
    def memory(self) -> MemoryRepository:
        raise NotImplementedError

    @property
    def vault(self) -> VaultRepository:
        raise NotImplementedError

    @property
    def user_profiles(self) -> UserProfileRepository:
        raise NotImplementedError

    @property
    def channels(self) -> ChannelRepository:
        raise NotImplementedError

    @property
    def webhooks(self) -> WebhookRepository:
        raise NotImplementedError

    async def migrate(self) -> None:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# RepositoryRuntime — success paths
# ---------------------------------------------------------------------------


class TestRuntimeSaveSuccess:
    async def test_data_stored(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = _StubDriver()
        rt = RepositoryRuntime(driver, make_options(audit_log))
        agent = make_agent()
        await rt.agents.save(agent)
        assert driver.agents._data["a1"] == agent  # type: ignore[attr-defined]

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = RepositoryRuntime(_StubDriver(), make_options(audit_log))
        await rt.agents.save(make_agent())
        assert mock_span.name == "repo.agent.save"

    async def test_span_attributes(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = RepositoryRuntime(_StubDriver(), make_options(audit_log))
        await rt.agents.save(make_agent("x1"))
        assert mock_span.attributes["entity.type"] == "agent"
        assert mock_span.attributes["entity.id"] == "x1"
        assert mock_span.attributes["repo.operation"] == "save"

    async def test_invocation_event_attached(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(), make_options(audit_log))
        await rt.agents.save(make_agent())
        event_names = [e[0] for e in mock_span.events]
        assert "repo.invocation" in event_names

    async def test_no_audit_entries_on_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(), make_options(audit_log))
        await rt.agents.save(make_agent())
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = RepositoryRuntime(_StubDriver(), make_options(audit_log))
        await rt.agents.save(make_agent())
        assert mock_span.ended


class TestRuntimeGetSuccess:
    async def test_returns_entity(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = _StubDriver()
        agent = make_agent()
        driver.agents._data["a1"] = agent  # type: ignore[attr-defined]
        rt = RepositoryRuntime(driver, make_options(audit_log))
        result = await rt.agents.get("a1")
        assert result == agent

    async def test_returns_none_for_missing(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(), make_options(audit_log))
        assert await rt.agents.get("missing") is None

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = RepositoryRuntime(_StubDriver(), make_options(audit_log))
        await rt.agents.get("a1")
        assert mock_span.name == "repo.agent.get"


class TestRuntimeDeleteSuccess:
    async def test_key_removed(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = _StubDriver()
        driver.agents._data["a1"] = make_agent()  # type: ignore[attr-defined]
        rt = RepositoryRuntime(driver, make_options(audit_log))
        await rt.agents.delete("a1")
        assert "a1" not in driver.agents._data  # type: ignore[attr-defined]

    async def test_missing_is_noop(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        rt = RepositoryRuntime(_StubDriver(), make_options(audit_log))
        await rt.agents.delete("nonexistent")
        assert audit_log.entries == []


class TestRuntimeListSuccess:
    async def test_returns_list(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        driver = _StubDriver()
        driver.agents._data["a1"] = make_agent("a1")  # type: ignore[attr-defined]
        driver.agents._data["a2"] = make_agent("a2")  # type: ignore[attr-defined]
        rt = RepositoryRuntime(driver, make_options(audit_log))
        results = await rt.agents.list(AgentFilter())
        assert len(results) == 2


# ---------------------------------------------------------------------------
# RepositoryRuntime — failure paths
# ---------------------------------------------------------------------------


class TestRuntimeStoreRaises:
    async def test_wraps_as_repo_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(
            _StubDriver(raises=RuntimeError("db error")), make_options(audit_log)
        )
        with pytest.raises(RepositoryFailure) as exc_info:
            await rt.agents.save(make_agent())
        assert exc_info.value.code == "REPO_SAVE_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = RuntimeError("db error")
        rt = RepositoryRuntime(_StubDriver(raises=orig), make_options(audit_log))
        with pytest.raises(RepositoryFailure) as exc_info:
            await rt.agents.save(make_agent())
        assert exc_info.value.cause is orig

    async def test_entity_type_and_operation_set(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(raises=RuntimeError("boom")), make_options(audit_log))
        with pytest.raises(RepositoryFailure) as exc_info:
            await rt.agents.save(make_agent())
        assert exc_info.value.entity_type == "agent"
        assert exc_info.value.operation == "save"

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(raises=RuntimeError("boom")), make_options(audit_log))
        with pytest.raises(RepositoryFailure):
            await rt.agents.save(make_agent())
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "repo.agent.save.failed"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(raises=RuntimeError("boom")), make_options(audit_log))
        with pytest.raises(RepositoryFailure):
            await rt.agents.save(make_agent())
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_error_event_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(raises=RuntimeError("boom")), make_options(audit_log))
        with pytest.raises(RepositoryFailure):
            await rt.agents.save(make_agent())
        event_names = [e[0] for e in mock_span.events]
        assert "repo.error" in event_names

    async def test_exception_recorded_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = RuntimeError("boom")
        rt = RepositoryRuntime(_StubDriver(raises=orig), make_options(audit_log))
        with pytest.raises(RepositoryFailure):
            await rt.agents.save(make_agent())
        assert orig in mock_span.recorded_exceptions

    async def test_on_error_callback_invoked(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[RepositoryFailure] = []
        rt = RepositoryRuntime(
            _StubDriver(raises=RuntimeError("boom")), make_options(audit_log, errors)
        )
        with pytest.raises(RepositoryFailure):
            await rt.agents.save(make_agent())
        assert len(errors) == 1
        assert errors[0].code == "REPO_SAVE_FAILED"

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(raises=RuntimeError("boom")), make_options(audit_log))
        with pytest.raises(RepositoryFailure):
            await rt.agents.save(make_agent())
        assert mock_span.ended

    async def test_get_wraps_as_get_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(raises=OSError("read fail")), make_options(audit_log))
        with pytest.raises(RepositoryFailure) as exc_info:
            await rt.agents.get("a1")
        assert exc_info.value.code == "REPO_GET_FAILED"

    async def test_delete_wraps_as_delete_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(raises=OSError("locked")), make_options(audit_log))
        with pytest.raises(RepositoryFailure) as exc_info:
            await rt.agents.delete("a1")
        assert exc_info.value.code == "REPO_DELETE_FAILED"

    async def test_list_wraps_as_list_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = RepositoryRuntime(_StubDriver(raises=RuntimeError("boom")), make_options(audit_log))
        with pytest.raises(RepositoryFailure) as exc_info:
            await rt.agents.list(AgentFilter())
        assert exc_info.value.code == "REPO_LIST_FAILED"


# ---------------------------------------------------------------------------
# SqliteRepositoryDriver — integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
async def sqlite_driver():
    driver = await SqliteRepositoryDriver.open(":memory:")
    await driver.migrate()
    yield driver
    await driver.close()


class TestSqliteMigrate:
    async def test_migrate_is_idempotent(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        await sqlite_driver.migrate()  # second run must not raise
        await sqlite_driver.migrate()  # third run must not raise


class TestSqliteAgent:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        agent = make_agent()
        await sqlite_driver.agents.save(agent)
        result = await sqlite_driver.agents.get("a1")
        assert result == agent

    async def test_get_missing_returns_none(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        assert await sqlite_driver.agents.get("nonexistent") is None

    async def test_delete_removes_record(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        await sqlite_driver.agents.save(make_agent())
        await sqlite_driver.agents.delete("a1")
        assert await sqlite_driver.agents.get("a1") is None

    async def test_delete_missing_is_noop(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        await sqlite_driver.agents.delete("nonexistent")  # must not raise

    async def test_upsert_updates_record(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        agent = Agent(
            id="a1",
            kind="k1",
            name="Old",
            config="{}",
            capabilities="[]",
            created_at=now,
            updated_at=now,
        )
        await sqlite_driver.agents.save(agent)
        updated = Agent(
            id="a1",
            kind="k2",
            name="New",
            config="{}",
            capabilities="[]",
            created_at=now,
            updated_at=_ts(),
        )
        await sqlite_driver.agents.save(updated)
        result = await sqlite_driver.agents.get("a1")
        assert result is not None
        assert result.name == "New"
        assert result.kind == "k2"

    async def test_list_returns_all(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        await sqlite_driver.agents.save(make_agent("a1"))
        await sqlite_driver.agents.save(make_agent("a2"))
        results = await sqlite_driver.agents.list(AgentFilter())
        assert len(results) == 2

    async def test_list_filter_by_kind(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        await sqlite_driver.agents.save(
            Agent(
                id="a1",
                kind="research",
                name="R",
                config="{}",
                capabilities="[]",
                created_at=now,
                updated_at=now,
            )
        )
        await sqlite_driver.agents.save(
            Agent(
                id="a2",
                kind="coding",
                name="C",
                config="{}",
                capabilities="[]",
                created_at=now,
                updated_at=now,
            )
        )
        results = await sqlite_driver.agents.list(AgentFilter(kind="research"))
        assert len(results) == 1
        assert results[0].id == "a1"

    async def test_list_pagination(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        for i in range(5):
            await sqlite_driver.agents.save(make_agent(f"a{i}"))
        page1 = await sqlite_driver.agents.list(AgentFilter(limit=3, offset=0))
        page2 = await sqlite_driver.agents.list(AgentFilter(limit=3, offset=3))
        assert len(page1) == 3
        assert len(page2) == 2


class TestSqliteSession:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        session = make_session()
        await sqlite_driver.sessions.save(session)
        result = await sqlite_driver.sessions.get("s1")
        assert result == session

    async def test_get_missing_returns_none(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        assert await sqlite_driver.sessions.get("missing") is None

    async def test_delete_removes_record(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        await sqlite_driver.sessions.save(make_session())
        await sqlite_driver.sessions.delete("s1")
        assert await sqlite_driver.sessions.get("s1") is None

    async def test_list_filter_by_agent_id(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        await sqlite_driver.sessions.save(make_session("s1", agent_id="a1"))
        await sqlite_driver.sessions.save(make_session("s2", agent_id="a2"))
        results = await sqlite_driver.sessions.list(SessionFilter(agent_id="a1"))
        assert len(results) == 1
        assert results[0].id == "s1"

    async def test_list_filter_by_status(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        await sqlite_driver.sessions.save(
            Session(
                id="s1",
                agent_id="a1",
                status="active",
                metadata=None,
                created_at=now,
                updated_at=now,
            )
        )
        await sqlite_driver.sessions.save(
            Session(
                id="s2",
                agent_id="a1",
                status="closed",
                metadata=None,
                created_at=now,
                updated_at=now,
            )
        )
        active = await sqlite_driver.sessions.list(SessionFilter(status="active"))
        assert len(active) == 1
        assert active[0].id == "s1"


class TestSqliteThread:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        thread = make_thread()
        await sqlite_driver.threads.save(thread)
        assert await sqlite_driver.threads.get("t1") == thread

    async def test_list_filter_by_session_id(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        await sqlite_driver.threads.save(make_thread("t1", session_id="s1"))
        await sqlite_driver.threads.save(make_thread("t2", session_id="s2"))
        results = await sqlite_driver.threads.list(ThreadFilter(session_id="s1"))
        assert len(results) == 1 and results[0].id == "t1"


class TestSqliteMessage:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        msg = make_message()
        await sqlite_driver.messages.save(msg)
        assert await sqlite_driver.messages.get("m1") == msg

    async def test_list_ordered_by_sequence(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        for seq in (2, 0, 1):
            await sqlite_driver.messages.save(
                Message(
                    id=f"m{seq}",
                    thread_id="t1",
                    session_id="s1",
                    role="user",
                    content="[]",
                    sequence=seq,
                    created_at=now,
                )
            )
        results = await sqlite_driver.messages.list(MessageFilter(thread_id="t1"))
        assert [r.sequence for r in results] == [0, 1, 2]

    async def test_list_filter_by_role(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        await sqlite_driver.messages.save(
            Message(
                id="m1",
                thread_id="t1",
                session_id="s1",
                role="user",
                content="[]",
                sequence=0,
                created_at=now,
            )
        )
        await sqlite_driver.messages.save(
            Message(
                id="m2",
                thread_id="t1",
                session_id="s1",
                role="assistant",
                content="[]",
                sequence=1,
                created_at=now,
            )
        )
        results = await sqlite_driver.messages.list(MessageFilter(role="user"))
        assert len(results) == 1 and results[0].role == "user"


class TestSqliteToolCall:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        tc = make_tool_call()
        await sqlite_driver.tool_calls.save(tc)
        assert await sqlite_driver.tool_calls.get("tc1") == tc

    async def test_list_filter_by_status(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        pending = ToolCall(
            id="tc1",
            message_id="m1",
            session_id="s1",
            tool_name="search",
            input="{}",
            output=None,
            status="pending",
            created_at=now,
            updated_at=now,
        )
        done = ToolCall(
            id="tc2",
            message_id="m1",
            session_id="s1",
            tool_name="search",
            input="{}",
            output="{}",
            status="success",
            created_at=now,
            updated_at=now,
        )
        await sqlite_driver.tool_calls.save(pending)
        await sqlite_driver.tool_calls.save(done)
        results = await sqlite_driver.tool_calls.list(ToolCallFilter(status="pending"))
        assert len(results) == 1 and results[0].id == "tc1"


class TestSqliteSkill:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        skill = make_skill()
        await sqlite_driver.skills.save(skill)
        assert await sqlite_driver.skills.get("sk1") == skill

    async def test_list_ordered_by_name(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        for name in ("z_skill", "a_skill", "m_skill"):
            s_id = name
            await sqlite_driver.skills.save(
                Skill(
                    id=s_id,
                    name=name,
                    description="",
                    capabilities="[]",
                    config="{}",
                    created_at=now,
                    updated_at=now,
                )
            )
        results = await sqlite_driver.skills.list(SkillFilter())
        assert [r.name for r in results] == ["a_skill", "m_skill", "z_skill"]


class TestSqliteEnvironment:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        env = make_environment()
        await sqlite_driver.environments.save(env)
        assert await sqlite_driver.environments.get("e1") == env

    async def test_list_filter_by_kind(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        await sqlite_driver.environments.save(
            Environment(
                id="e1",
                kind="python",
                status="provisioned",
                config="{}",
                created_at=now,
                updated_at=now,
            )
        )
        await sqlite_driver.environments.save(
            Environment(
                id="e2",
                kind="node",
                status="provisioned",
                config="{}",
                created_at=now,
                updated_at=now,
            )
        )
        results = await sqlite_driver.environments.list(EnvironmentFilter(kind="python"))
        assert len(results) == 1 and results[0].id == "e1"


class TestSqliteMemory:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        entry = make_memory_entry()
        await sqlite_driver.memory.save(entry)
        assert await sqlite_driver.memory.get("me1") == entry

    async def test_list_filter_by_scope(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        await sqlite_driver.memory.save(
            MemoryEntry(
                id="me1", scope="agent:a1", key="k1", value="v1", created_at=now, updated_at=now
            )
        )
        await sqlite_driver.memory.save(
            MemoryEntry(
                id="me2", scope="agent:a2", key="k1", value="v2", created_at=now, updated_at=now
            )
        )
        results = await sqlite_driver.memory.list(MemoryFilter(scope="agent:a1"))
        assert len(results) == 1 and results[0].id == "me1"

    async def test_list_ordered_by_key(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        for k in ("z", "a", "m"):
            await sqlite_driver.memory.save(
                MemoryEntry(id=k, scope="s", key=k, value="v", created_at=now, updated_at=now)
            )
        results = await sqlite_driver.memory.list(MemoryFilter())
        assert [r.key for r in results] == ["a", "m", "z"]


class TestSqliteVault:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        entry = make_vault_entry()
        await sqlite_driver.vault.save(entry)
        assert await sqlite_driver.vault.get("v1") == entry

    async def test_list_ordered_by_name(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        for name in ("z_key", "a_key"):
            await sqlite_driver.vault.save(
                VaultEntry(id=name, name=name, description=None, created_at=now, updated_at=now)
            )
        results = await sqlite_driver.vault.list(VaultFilter())
        assert [r.name for r in results] == ["a_key", "z_key"]


class TestSqliteUserProfile:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        profile = make_user_profile()
        await sqlite_driver.user_profiles.save(profile)
        assert await sqlite_driver.user_profiles.get("u1") == profile

    async def test_list_ordered_by_username(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        for username in ("zara", "alice", "mike"):
            await sqlite_driver.user_profiles.save(
                UserProfile(
                    id=username,
                    username=username,
                    display_name=None,
                    email=None,
                    metadata=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        results = await sqlite_driver.user_profiles.list(UserProfileFilter())
        assert [r.username for r in results] == ["alice", "mike", "zara"]


class TestSqliteChannel:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        ch = make_channel()
        await sqlite_driver.channels.save(ch)
        assert await sqlite_driver.channels.get("ch1") == ch

    async def test_list_filter_by_status(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        await sqlite_driver.channels.save(
            Channel(
                id="c1",
                kind="slack",
                name="general",
                config="{}",
                status="active",
                created_at=now,
                updated_at=now,
            )
        )
        await sqlite_driver.channels.save(
            Channel(
                id="c2",
                kind="slack",
                name="alerts",
                config="{}",
                status="inactive",
                created_at=now,
                updated_at=now,
            )
        )
        results = await sqlite_driver.channels.list(ChannelFilter(status="active"))
        assert len(results) == 1 and results[0].id == "c1"


class TestSqliteWebhook:
    async def test_save_get_roundtrip(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        wh = make_webhook()
        await sqlite_driver.webhooks.save(wh)
        assert await sqlite_driver.webhooks.get("wh1") == wh

    async def test_list_filter_by_status(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        await sqlite_driver.webhooks.save(
            Webhook(
                id="wh1",
                url="https://a.com",
                events="[]",
                secret_ref=None,
                status="active",
                created_at=now,
                updated_at=now,
            )
        )
        await sqlite_driver.webhooks.save(
            Webhook(
                id="wh2",
                url="https://b.com",
                events="[]",
                secret_ref=None,
                status="inactive",
                created_at=now,
                updated_at=now,
            )
        )
        results = await sqlite_driver.webhooks.list(WebhookFilter(status="active"))
        assert len(results) == 1 and results[0].id == "wh1"

    async def test_secret_ref_nullable(self, sqlite_driver: SqliteRepositoryDriver) -> None:
        now = _ts()
        wh = Webhook(
            id="wh1",
            url="https://a.com",
            events="[]",
            secret_ref="secret_ref://vault/my_secret",
            status="active",
            created_at=now,
            updated_at=now,
        )
        await sqlite_driver.webhooks.save(wh)
        result = await sqlite_driver.webhooks.get("wh1")
        assert result is not None
        assert result.secret_ref == "secret_ref://vault/my_secret"
