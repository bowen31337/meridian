"""Full coverage for the Postgres backend driving every repository through a
fake asyncpg pool/connection — exercises get (found + missing), save, delete,
and list (with all filter conditions set and with none) for all 12 repos, plus
the driver open/migrate/close path. No live database required."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from storage_repository import (
    AgentFilter,
    AuditLogEntryFilter,
    AuditLogEntryRecord,
    ChannelFilter,
    EnvironmentFilter,
    MemoryFilter,
    MessageFilter,
    SessionFilter,
    SkillFilter,
    ThreadFilter,
    ToolCallFilter,
    UserProfileFilter,
    VaultFilter,
    WebhookFilter,
)
from storage_repository._postgres import PostgresRepositoryDriver

from .test_conformance import (
    make_agent,
    make_channel,
    make_environment,
    make_memory_entry,
    make_message,
    make_session,
    make_skill,
    make_thread,
    make_tool_call,
    make_user_profile,
    make_vault_entry,
    make_webhook,
)

# ---------------------------------------------------------------------------
# Fake asyncpg pool / connection
# ---------------------------------------------------------------------------


class FakeConn:
    def __init__(self) -> None:
        self.fetchrow_result: Any = None
        self.fetch_result: list[Any] = []
        self.executed: list[str] = []

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        return self.fetchrow_result

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        return self.fetch_result

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append(sql)
        return "OK"


class _Acquire:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeConn:
        return self._conn

    async def __aexit__(self, *exc: object) -> bool:
        return False


class FakePool:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn
        self.closed = False

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Row converters: (model -> positional tuple) in each _postgres converter order
# ---------------------------------------------------------------------------


def _agent_row(a: Any) -> tuple[Any, ...]:
    return (a.id, a.kind, a.name, a.config, a.capabilities, a.created_at, a.updated_at)


def _session_row(s: Any) -> tuple[Any, ...]:
    return (s.id, s.agent_id, s.status, s.metadata, s.created_at, s.updated_at)


def _thread_row(t: Any) -> tuple[Any, ...]:
    return (t.id, t.session_id, t.title, t.created_at, t.updated_at)


def _message_row(m: Any) -> tuple[Any, ...]:
    return (m.id, m.thread_id, m.session_id, m.role, m.content, m.sequence, m.created_at)


def _tool_call_row(tc: Any) -> tuple[Any, ...]:
    return (
        tc.id,
        tc.message_id,
        tc.session_id,
        tc.tool_name,
        tc.input,
        tc.output,
        tc.status,
        tc.created_at,
        tc.updated_at,
    )


def _skill_row(sk: Any) -> tuple[Any, ...]:
    return (
        sk.id,
        sk.name,
        sk.description,
        sk.capabilities,
        sk.config,
        sk.created_at,
        sk.updated_at,
    )


def _environment_row(e: Any) -> tuple[Any, ...]:
    return (e.id, e.kind, e.status, e.config, e.created_at, e.updated_at)


def _memory_row(me: Any) -> tuple[Any, ...]:
    return (me.id, me.scope, me.key, me.value, me.created_at, me.updated_at)


def _vault_row(v: Any) -> tuple[Any, ...]:
    return (v.id, v.name, v.description, v.created_at, v.updated_at)


def _user_profile_row(u: Any) -> tuple[Any, ...]:
    return (
        u.id,
        u.username,
        u.display_name,
        u.email,
        u.metadata,
        u.is_primary,
        u.created_at,
        u.updated_at,
    )


def _channel_row(c: Any) -> tuple[Any, ...]:
    return (c.id, c.kind, c.name, c.config, c.status, c.created_at, c.updated_at)


def _webhook_row(w: Any) -> tuple[Any, ...]:
    return (w.id, w.url, w.events, w.secret_ref, w.status, w.created_at, w.updated_at)


# repo_attr, make_fn, row_fn, full_filter (all conditions set), empty_filter
_CASES = [
    ("agents", make_agent, _agent_row, AgentFilter(kind="k"), AgentFilter()),
    (
        "sessions",
        make_session,
        _session_row,
        SessionFilter(agent_id="a", status="active"),
        SessionFilter(),
    ),
    ("threads", make_thread, _thread_row, ThreadFilter(session_id="s"), ThreadFilter()),
    (
        "messages",
        make_message,
        _message_row,
        MessageFilter(thread_id="t", session_id="s", role="user"),
        MessageFilter(),
    ),
    (
        "tool_calls",
        make_tool_call,
        _tool_call_row,
        ToolCallFilter(message_id="m", session_id="s", status="pending"),
        ToolCallFilter(),
    ),
    ("skills", make_skill, _skill_row, SkillFilter(), SkillFilter()),
    (
        "environments",
        make_environment,
        _environment_row,
        EnvironmentFilter(kind="k", status="provisioned"),
        EnvironmentFilter(),
    ),
    ("memory", make_memory_entry, _memory_row, MemoryFilter(scope="agent:a1"), MemoryFilter()),
    ("vault", make_vault_entry, _vault_row, VaultFilter(), VaultFilter()),
    (
        "user_profiles",
        make_user_profile,
        _user_profile_row,
        UserProfileFilter(),
        UserProfileFilter(),
    ),
    (
        "channels",
        make_channel,
        _channel_row,
        ChannelFilter(kind="k", status="active"),
        ChannelFilter(),
    ),
    ("webhooks", make_webhook, _webhook_row, WebhookFilter(status="active"), WebhookFilter()),
]


@pytest.fixture
def conn() -> FakeConn:
    return FakeConn()


@pytest.fixture
def driver(conn: FakeConn) -> PostgresRepositoryDriver:
    return PostgresRepositoryDriver(FakePool(conn))


@pytest.mark.parametrize(("attr", "make_fn", "row_fn", "full_filter", "empty_filter"), _CASES)
async def test_repo_crud_and_list(
    driver: PostgresRepositoryDriver,
    conn: FakeConn,
    attr: str,
    make_fn: Any,
    row_fn: Any,
    full_filter: Any,
    empty_filter: Any,
) -> None:
    repo = getattr(driver, attr)
    entity = make_fn()

    # get: found
    conn.fetchrow_result = row_fn(entity)
    fetched = await repo.get(entity.id)
    assert fetched == entity

    # get: missing
    conn.fetchrow_result = None
    assert await repo.get("missing") is None

    # save + delete
    await repo.save(entity)
    await repo.delete(entity.id)

    # list: with all conditions set, then with none
    conn.fetch_result = [row_fn(entity)]
    full = await repo.list(full_filter)
    assert full == [entity]
    empty = await repo.list(empty_filter)
    assert empty == [entity]


def _audit_record() -> AuditLogEntryRecord:
    return AuditLogEntryRecord(
        id="al1",
        level="info",
        event="agent.saved",
        entity_type="agent",
        entity_id="a1",
        operation="save",
        timestamp="2026-01-01T00:00:00Z",
        detail={"k": "v"},
        signature="sig",
    )


def _audit_row(r: AuditLogEntryRecord) -> tuple[Any, ...]:
    return (
        r.id,
        r.level,
        r.event,
        r.entity_type,
        r.entity_id,
        r.operation,
        r.timestamp,
        json.dumps(r.detail) if r.detail is not None else None,
        r.signature,
    )


async def test_audit_log_entries_append_and_list(
    driver: PostgresRepositoryDriver, conn: FakeConn
) -> None:
    repo = driver.audit_log_entries
    record = _audit_record()

    await repo.append(record)
    assert conn.executed  # INSERT issued

    conn.fetch_result = [_audit_row(record)]
    full_filter = AuditLogEntryFilter(
        level="info",
        event="agent.saved",
        entity_type="agent",
        entity_id="a1",
        since="2025-01-01T00:00:00Z",
        until="2027-01-01T00:00:00Z",
    )
    assert await repo.list(full_filter) == [record]
    assert await repo.list(AuditLogEntryFilter()) == [record]


async def test_audit_log_entries_append_null_detail(
    driver: PostgresRepositoryDriver, conn: FakeConn
) -> None:
    record = AuditLogEntryRecord(
        id="al2",
        level="warn",
        event="x",
        entity_type="t",
        entity_id="e",
        operation="op",
        timestamp="2026-01-01T00:00:00Z",
        detail=None,
        signature=None,
    )
    await driver.audit_log_entries.append(record)
    conn.fetch_result = [_audit_row(record)]
    assert await driver.audit_log_entries.list(AuditLogEntryFilter()) == [record]


async def test_memory_save_embedding_and_vec_search(
    driver: PostgresRepositoryDriver, conn: FakeConn
) -> None:
    from storage_repository import MemoryVecSearchFilter, MemoryVecSearchResult

    repo = driver.memory
    await repo.save_embedding("me1", b"\x00\x01")
    assert conn.executed  # UPDATE issued

    entry = make_memory_entry()
    conn.fetch_result = [(*_memory_row(entry), 0.25)]

    scoped = await repo.vec_search(
        MemoryVecSearchFilter(embedding=b"\x00\x01", scope="agent:a1", limit=5)
    )
    assert scoped == [MemoryVecSearchResult(entry=entry, distance=0.25)]

    unscoped = await repo.vec_search(MemoryVecSearchFilter(embedding=b"\x00\x01", limit=5))
    assert unscoped == [MemoryVecSearchResult(entry=entry, distance=0.25)]


async def test_driver_open_creates_pool() -> None:
    fake_pool = FakePool(FakeConn())
    with patch(
        "storage_repository._postgres.asyncpg.create_pool",
        new=AsyncMock(return_value=fake_pool),
    ) as create_pool:
        driver = await PostgresRepositoryDriver.open("postgresql://x/y", min_size=1)
    create_pool.assert_awaited_once()
    assert isinstance(driver, PostgresRepositoryDriver)


async def test_driver_migrate_and_close(driver: PostgresRepositoryDriver, conn: FakeConn) -> None:
    await driver.migrate()
    assert len(conn.executed) > 0
    await driver.close()
    assert driver._pool.closed is True  # type: ignore[attr-defined]
