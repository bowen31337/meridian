"""Closes the remaining _sqlite.py coverage gaps the conformance suite leaves:
every repo delete, the unfiltered/extra-condition list branches, the memory
embedding + vector-search success and not-found paths, the audit-log repo
append/list, the sqlite-vec loader fall-throughs, and the driver open/migrate
error branches not driven by a failing connect."""

from __future__ import annotations

import struct
from typing import Any

import pytest
from storage_repository import (
    AuditLogEntryFilter,
    AuditLogEntryRecord,
    ChannelFilter,
    EnvironmentFilter,
    MemoryVecSearchFilter,
    MessageFilter,
    RepositoryFailure,
    ThreadFilter,
    ToolCallFilter,
    WebhookFilter,
)
from storage_repository._sqlite import SqliteRepositoryDriver, _load_sqlite_vec

from .conftest import CapturingAuditLog
from .test_conformance import (
    make_channel,
    make_environment,
    make_memory_entry,
    make_message,
    make_skill,
    make_thread,
    make_tool_call,
    make_user_profile,
    make_vault_entry,
    make_webhook,
)


@pytest.fixture()
async def driver() -> Any:
    d = await SqliteRepositoryDriver.open(":memory:")
    await d.migrate()
    yield d
    await d.close()


def _embedding() -> bytes:
    return struct.pack("128f", *([0.1] * 128))


# ---------------------------------------------------------------------------
# _load_sqlite_vec fall-throughs
# ---------------------------------------------------------------------------


class _NoExtConn:
    """A connection object missing enable_load_extension (hasattr -> False)."""


class _RaisingExtConn:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def enable_load_extension(self, flag: bool) -> None:
        self.calls.append(flag)


def test_load_sqlite_vec_no_extension_support() -> None:
    # hasattr(...) is False -> returns immediately without raising.
    _load_sqlite_vec(_NoExtConn())  # type: ignore[arg-type]


def test_load_sqlite_vec_swallows_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import storage_repository._sqlite as sqlite_mod

    def _boom(_db: Any) -> None:
        raise RuntimeError("cannot load")

    monkeypatch.setattr(sqlite_mod.sqlite_vec, "load", _boom)
    conn = _RaisingExtConn()
    _load_sqlite_vec(conn)  # type: ignore[arg-type]
    # enable_load_extension toggled on then off even though load() raised.
    assert conn.calls == [True, False]


# ---------------------------------------------------------------------------
# Deletes + unfiltered/extra-condition list branches
# ---------------------------------------------------------------------------


async def test_thread_delete_and_unfiltered_list(driver: Any) -> None:
    thread = make_thread()
    await driver.threads.save(thread)
    await driver.threads.delete(thread.id)
    assert await driver.threads.get(thread.id) is None
    assert await driver.threads.list(ThreadFilter()) == []


async def test_message_delete_and_session_filter(driver: Any) -> None:
    msg = make_message()
    await driver.messages.save(msg)
    await driver.messages.delete(msg.id)
    assert await driver.messages.get(msg.id) is None
    await driver.messages.save(make_message())
    results = await driver.messages.list(MessageFilter(session_id=msg.session_id, role=msg.role))
    assert len(results) == 1


async def test_tool_call_delete_and_filters(driver: Any) -> None:
    tc = make_tool_call()
    await driver.tool_calls.save(tc)
    await driver.tool_calls.delete(tc.id)
    assert await driver.tool_calls.get(tc.id) is None
    await driver.tool_calls.save(make_tool_call())
    results = await driver.tool_calls.list(
        ToolCallFilter(message_id=tc.message_id, session_id=tc.session_id, status=tc.status)
    )
    assert len(results) == 1
    # empty filter exercises the all-conditions-None branches
    assert len(await driver.tool_calls.list(ToolCallFilter())) == 1


async def test_skill_delete(driver: Any) -> None:
    skill = make_skill()
    await driver.skills.save(skill)
    await driver.skills.delete(skill.id)
    assert await driver.skills.get(skill.id) is None


async def test_environment_delete_and_unfiltered_list(driver: Any) -> None:
    env = make_environment()
    await driver.environments.save(env)
    await driver.environments.delete(env.id)
    assert await driver.environments.get(env.id) is None
    await driver.environments.save(make_environment())
    assert len(await driver.environments.list(EnvironmentFilter(status="provisioned"))) == 1


async def test_vault_delete(driver: Any) -> None:
    entry = make_vault_entry()
    await driver.vault.save(entry)
    await driver.vault.delete(entry.id)
    assert await driver.vault.get(entry.id) is None


async def test_user_profile_delete(driver: Any) -> None:
    profile = make_user_profile()
    await driver.user_profiles.save(profile)
    await driver.user_profiles.delete(profile.id)
    assert await driver.user_profiles.get(profile.id) is None


async def test_channel_delete_and_kind_filter(driver: Any) -> None:
    ch = make_channel()
    await driver.channels.save(ch)
    await driver.channels.delete(ch.id)
    assert await driver.channels.get(ch.id) is None
    await driver.channels.save(make_channel())
    results = await driver.channels.list(ChannelFilter(kind=ch.kind, status=ch.status))
    assert len(results) == 1
    # kind-only filter leaves status None, exercising that skip branch
    assert len(await driver.channels.list(ChannelFilter(kind=ch.kind))) == 1


async def test_webhook_delete_and_unfiltered_list(driver: Any) -> None:
    wh = make_webhook()
    await driver.webhooks.save(wh)
    await driver.webhooks.delete(wh.id)
    assert await driver.webhooks.get(wh.id) is None
    await driver.webhooks.save(make_webhook())
    assert len(await driver.webhooks.list(WebhookFilter())) == 1


# ---------------------------------------------------------------------------
# Memory: delete (with vec rowid), save_embedding success + not-found, vec_search
# ---------------------------------------------------------------------------


async def test_memory_delete_clears_vector(driver: Any) -> None:
    entry = make_memory_entry()
    await driver.memory.save(entry)
    await driver.memory.save_embedding(entry.id, _embedding())
    await driver.memory.delete(entry.id)
    assert await driver.memory.get(entry.id) is None
    # deleting a missing entry skips the vec-row delete branch
    await driver.memory.delete("nonexistent")


async def test_memory_save_embedding_generic_error_wrapped(driver: Any) -> None:
    audit = CapturingAuditLog()
    d = await SqliteRepositoryDriver.open(":memory:", audit_log=audit)
    await d.migrate()
    try:
        entry = make_memory_entry()
        await d.memory.save(entry)
        # a malformed (wrong-length) embedding makes the vec INSERT fail,
        # exercising the generic-exception wrap in save_embedding.
        with pytest.raises(RepositoryFailure) as exc_info:
            await d.memory.save_embedding(entry.id, b"\x00\x01")
        assert exc_info.value.code == "MEMORY_SAVE_EMBEDDING_FAILED"
        assert any(e.event == "repo.memory.save_embedding.failed" for e in audit.entries)
    finally:
        await d.close()


async def test_memory_save_embedding_and_vec_search(driver: Any) -> None:
    entry = make_memory_entry()
    await driver.memory.save(entry)
    emb = _embedding()
    await driver.memory.save_embedding(entry.id, emb)
    # second call exercises the replace (delete-then-insert) path.
    await driver.memory.save_embedding(entry.id, emb)

    scoped = await driver.memory.vec_search(
        MemoryVecSearchFilter(embedding=emb, scope=entry.scope, limit=5)
    )
    assert len(scoped) == 1 and scoped[0].entry.id == entry.id
    unscoped = await driver.memory.vec_search(MemoryVecSearchFilter(embedding=emb, limit=5))
    assert len(unscoped) == 1


async def test_memory_save_embedding_missing_entry_raises(driver: Any) -> None:
    audit = CapturingAuditLog()
    d = await SqliteRepositoryDriver.open(":memory:", audit_log=audit)
    await d.migrate()
    try:
        with pytest.raises(RepositoryFailure) as exc_info:
            await d.memory.save_embedding("nonexistent", _embedding())
        assert exc_info.value.code == "MEMORY_ENTRY_NOT_FOUND"
        assert any(e.event == "repo.memory.save_embedding.failed" for e in audit.entries)
    finally:
        await d.close()


async def test_memory_vec_search_error_wrapped(driver: Any) -> None:
    audit = CapturingAuditLog()
    d = await SqliteRepositoryDriver.open(":memory:", audit_log=audit)
    await d.migrate()
    try:
        # A malformed embedding (wrong byte length) makes the vec MATCH fail,
        # exercising the vec_search generic-exception wrap + audit-log write.
        with pytest.raises(RepositoryFailure) as exc_info:
            await d.memory.vec_search(MemoryVecSearchFilter(embedding=b"\x00\x01", limit=5))
        assert exc_info.value.code == "MEMORY_VEC_SEARCH_FAILED"
        assert any(e.event == "repo.memory.vec_search.failed" for e in audit.entries)
    finally:
        await d.close()


# ---------------------------------------------------------------------------
# Audit log entries repo
# ---------------------------------------------------------------------------


def _audit_record(id: str = "al1", detail: dict[str, Any] | None = None) -> AuditLogEntryRecord:
    return AuditLogEntryRecord(
        id=id,
        level="info",
        event="agent.saved",
        entity_type="agent",
        entity_id="a1",
        operation="save",
        timestamp="2026-01-01T00:00:00Z",
        detail=detail,
        signature="sig",
    )


async def test_audit_log_append_and_list(driver: Any) -> None:
    assert driver.audit_log_entries is driver.audit_log_entries  # property accessor
    await driver.audit_log_entries.append(_audit_record("al1", {"k": "v"}))
    await driver.audit_log_entries.append(_audit_record("al2", None))

    full = await driver.audit_log_entries.list(
        AuditLogEntryFilter(
            level="info",
            event="agent.saved",
            entity_type="agent",
            entity_id="a1",
            since="2025-01-01T00:00:00Z",
            until="2027-01-01T00:00:00Z",
        )
    )
    assert {r.id for r in full} == {"al1", "al2"}
    assert any(r.detail == {"k": "v"} for r in full)
    assert any(r.detail is None for r in full)
    assert len(await driver.audit_log_entries.list(AuditLogEntryFilter())) == 2


# ---------------------------------------------------------------------------
# Driver open/migrate error branches
# ---------------------------------------------------------------------------


async def test_open_closes_conn_when_pragma_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import storage_repository._sqlite as sqlite_mod

    real_connect = sqlite_mod.aiosqlite.connect
    closed: list[bool] = []

    class _Wrapper:
        def __init__(self, conn: Any) -> None:
            self._conn = conn

        def __getattr__(self, name: str) -> Any:
            return getattr(self._conn, name)

        async def execute(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("pragma boom")

        async def close(self) -> None:
            closed.append(True)
            await self._conn.close()

    async def _fake_connect(*a: Any, **k: Any) -> Any:
        return _Wrapper(await real_connect(*a, **k))

    monkeypatch.setattr(sqlite_mod.aiosqlite, "connect", _fake_connect)
    with pytest.raises(RepositoryFailure) as exc_info:
        await SqliteRepositoryDriver.open(":memory:")
    assert exc_info.value.code == "SQLITE_OPEN_FAILED"
    assert closed == [True]


async def test_migrate_skips_no_such_module_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import storage_repository._sqlite as sqlite_mod

    # A migration referencing a missing vec module raises "no such module",
    # which migrate() swallows (continue) so the migration still records.
    monkeypatch.setattr(
        sqlite_mod,
        "load_migration_files",
        lambda: [(1, "0001_a.sql", "CREATE VIRTUAL TABLE t USING no_such_mod(x);")],
    )
    d = await SqliteRepositoryDriver.open(":memory:")
    try:
        await d.migrate()
        async with d._conn.execute("SELECT COUNT(*) FROM schema_migrations") as cur:
            row = await cur.fetchone()
        assert row[0] == 1
    finally:
        await d.close()


async def test_migrate_wraps_generic_statement_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import storage_repository._sqlite as sqlite_mod

    # A statement that fails with something other than "no such module" is
    # re-raised and wrapped as SQLITE_MIGRATE_FAILED.
    monkeypatch.setattr(
        sqlite_mod,
        "load_migration_files",
        lambda: [(1, "0001_a.sql", "THIS IS NOT VALID SQL;")],
    )
    audit = CapturingAuditLog()
    d = await SqliteRepositoryDriver.open(":memory:", audit_log=audit)
    try:
        with pytest.raises(RepositoryFailure) as exc_info:
            await d.migrate()
        assert exc_info.value.code == "SQLITE_MIGRATE_FAILED"
        assert any(e.event == "sqlite.driver.migrate.failed" for e in audit.entries)
    finally:
        await d.close()
