"""
SQLite backend for all 12 system repositories.

Requires the optional `aiosqlite` dependency (pip install meridian-storage-repository[sqlite]).

Usage:
    driver = await SqliteRepositoryDriver.open(":memory:")
    await driver.migrate()
    runtime = RepositoryRuntime(driver)
    await runtime.agents.save(Agent(...))
    await driver.close()

Upsert syntax uses the ANSI-standard ON CONFLICT DO UPDATE clause, available in
SQLite >= 3.24 (Python 3.11 ships with SQLite >= 3.39).  Placeholders are `?`.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

import aiosqlite
import sqlite_vec

from ._audit import AuditLog, NoopAuditLog
from ._contract import (
    AgentRepository,
    AuditLogEntryRepository,
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
from ._migrations import SCHEMA_VERSION, load_migration_files
from ._runtime import RepositoryDriver
from ._telemetry import get_tracer, record_invocation_event, record_repo_failure
from ._types import (
    Agent,
    AgentFilter,
    AuditLogEntry,
    AuditLogEntryFilter,
    AuditLogEntryRecord,
    Channel,
    ChannelFilter,
    Environment,
    EnvironmentFilter,
    MemoryEntry,
    MemoryFilter,
    MemoryVecSearchFilter,
    MemoryVecSearchResult,
    Message,
    MessageFilter,
    RepositoryFailure,
    Session,
    SessionFilter,
    Skill,
    SkillFilter,
    StructuredEvent,
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

Row = tuple[Any, ...]


# ---------------------------------------------------------------------------
# sqlite-vec extension loader
# ---------------------------------------------------------------------------


def _load_sqlite_vec(db: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension; no-op when extension loading is unsupported."""
    if not hasattr(db, "enable_load_extension"):
        return
    db.enable_load_extension(True)
    try:
        sqlite_vec.load(db)
    except Exception:
        pass
    finally:
        db.enable_load_extension(False)


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------


def _agent(row: Row) -> Agent:
    return Agent(
        id=row[0],
        kind=row[1],
        name=row[2],
        config=row[3],
        capabilities=row[4],
        created_at=row[5],
        updated_at=row[6],
    )


def _session(row: Row) -> Session:
    return Session(
        id=row[0],
        agent_id=row[1],
        status=row[2],
        metadata=row[3],
        created_at=row[4],
        updated_at=row[5],
    )


def _thread(row: Row) -> Thread:
    return Thread(id=row[0], session_id=row[1], title=row[2], created_at=row[3], updated_at=row[4])


def _message(row: Row) -> Message:
    return Message(
        id=row[0],
        thread_id=row[1],
        session_id=row[2],
        role=row[3],
        content=row[4],
        sequence=row[5],
        created_at=row[6],
    )


def _tool_call(row: Row) -> ToolCall:
    return ToolCall(
        id=row[0],
        message_id=row[1],
        session_id=row[2],
        tool_name=row[3],
        input=row[4],
        output=row[5],
        status=row[6],
        created_at=row[7],
        updated_at=row[8],
    )


def _skill(row: Row) -> Skill:
    return Skill(
        id=row[0],
        name=row[1],
        description=row[2],
        capabilities=row[3],
        config=row[4],
        created_at=row[5],
        updated_at=row[6],
    )


def _environment(row: Row) -> Environment:
    return Environment(
        id=row[0], kind=row[1], status=row[2], config=row[3], created_at=row[4], updated_at=row[5]
    )


def _memory_entry(row: Row) -> MemoryEntry:
    return MemoryEntry(
        id=row[0], scope=row[1], key=row[2], value=row[3], created_at=row[4], updated_at=row[5]
    )


def _vault_entry(row: Row) -> VaultEntry:
    return VaultEntry(
        id=row[0], name=row[1], description=row[2], created_at=row[3], updated_at=row[4]
    )


def _user_profile(row: Row) -> UserProfile:
    return UserProfile(
        id=row[0],
        username=row[1],
        display_name=row[2],
        email=row[3],
        metadata=row[4],
        is_primary=bool(row[5]),
        created_at=row[6],
        updated_at=row[7],
    )


def _channel(row: Row) -> Channel:
    return Channel(
        id=row[0],
        kind=row[1],
        name=row[2],
        config=row[3],
        status=row[4],
        created_at=row[5],
        updated_at=row[6],
    )


def _webhook(row: Row) -> Webhook:
    return Webhook(
        id=row[0],
        url=row[1],
        events=row[2],
        secret_ref=row[3],
        status=row[4],
        created_at=row[5],
        updated_at=row[6],
    )


def _audit_log_entry_record(row: Row) -> AuditLogEntryRecord:
    detail_raw = row[7]
    return AuditLogEntryRecord(
        id=row[0],
        level=row[1],
        event=row[2],
        entity_type=row[3],
        entity_id=row[4],
        operation=row[5],
        timestamp=row[6],
        detail=json.loads(detail_raw) if detail_raw is not None else None,
        signature=row[8],
    )


# ---------------------------------------------------------------------------
# Repository implementations
# ---------------------------------------------------------------------------


class _SqliteAgentRepo(AgentRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, agent_id: str) -> Agent | None:
        async with self._conn.execute(
            "SELECT id, kind, name, config, capabilities, created_at, updated_at"
            " FROM agents WHERE id = ?",
            (agent_id,),
        ) as cur:
            row = await cur.fetchone()
            return _agent(row) if row else None

    async def save(self, agent: Agent) -> None:
        await self._conn.execute(
            """
            INSERT INTO agents (id, kind, name, config, capabilities, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                kind         = excluded.kind,
                name         = excluded.name,
                config       = excluded.config,
                capabilities = excluded.capabilities,
                updated_at   = excluded.updated_at
            """,
            (
                agent.id,
                agent.kind,
                agent.name,
                agent.config,
                agent.capabilities,
                agent.created_at,
                agent.updated_at,
            ),
        )
        await self._conn.commit()

    async def delete(self, agent_id: str) -> None:
        await self._conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        await self._conn.commit()

    async def list(self, filter: AgentFilter) -> list[Agent]:
        conditions: list[str] = []
        params: list[Any] = []
        if filter.kind is not None:
            conditions.append("kind = ?")
            params.append(filter.kind)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._conn.execute(
            f"SELECT id, kind, name, config, capabilities, created_at, updated_at"
            f" FROM agents {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            return [_agent(row) for row in await cur.fetchall()]


class _SqliteSessionRepo(SessionRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, session_id: str) -> Session | None:
        async with self._conn.execute(
            "SELECT id, agent_id, status, metadata, created_at, updated_at"
            " FROM sessions WHERE id = ?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
            return _session(row) if row else None

    async def save(self, session: Session) -> None:
        await self._conn.execute(
            """
            INSERT INTO sessions (id, agent_id, status, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                agent_id   = excluded.agent_id,
                status     = excluded.status,
                metadata   = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (
                session.id,
                session.agent_id,
                session.status,
                session.metadata,
                session.created_at,
                session.updated_at,
            ),
        )
        await self._conn.commit()

    async def delete(self, session_id: str) -> None:
        await self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self._conn.commit()

    async def list(self, filter: SessionFilter) -> list[Session]:
        conditions: list[str] = []
        params: list[Any] = []
        if filter.agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(filter.agent_id)
        if filter.status is not None:
            conditions.append("status = ?")
            params.append(filter.status)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._conn.execute(
            f"SELECT id, agent_id, status, metadata, created_at, updated_at"
            f" FROM sessions {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            return [_session(row) for row in await cur.fetchall()]


class _SqliteThreadRepo(ThreadRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, thread_id: str) -> Thread | None:
        async with self._conn.execute(
            "SELECT id, session_id, title, created_at, updated_at FROM threads WHERE id = ?",
            (thread_id,),
        ) as cur:
            row = await cur.fetchone()
            return _thread(row) if row else None

    async def save(self, thread: Thread) -> None:
        await self._conn.execute(
            """
            INSERT INTO threads (id, session_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                session_id = excluded.session_id,
                title      = excluded.title,
                updated_at = excluded.updated_at
            """,
            (thread.id, thread.session_id, thread.title, thread.created_at, thread.updated_at),
        )
        await self._conn.commit()

    async def delete(self, thread_id: str) -> None:
        await self._conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        await self._conn.commit()

    async def list(self, filter: ThreadFilter) -> list[Thread]:
        conditions: list[str] = []
        params: list[Any] = []
        if filter.session_id is not None:
            conditions.append("session_id = ?")
            params.append(filter.session_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._conn.execute(
            f"SELECT id, session_id, title, created_at, updated_at"
            f" FROM threads {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            return [_thread(row) for row in await cur.fetchall()]


class _SqliteMessageRepo(MessageRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, message_id: str) -> Message | None:
        async with self._conn.execute(
            "SELECT id, thread_id, session_id, role, content, sequence, created_at"
            " FROM messages WHERE id = ?",
            (message_id,),
        ) as cur:
            row = await cur.fetchone()
            return _message(row) if row else None

    async def save(self, message: Message) -> None:
        await self._conn.execute(
            """
            INSERT INTO messages (id, thread_id, session_id, role, content, sequence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                thread_id  = excluded.thread_id,
                session_id = excluded.session_id,
                role       = excluded.role,
                content    = excluded.content,
                sequence   = excluded.sequence
            """,
            (
                message.id,
                message.thread_id,
                message.session_id,
                message.role,
                message.content,
                message.sequence,
                message.created_at,
            ),
        )
        await self._conn.commit()

    async def delete(self, message_id: str) -> None:
        await self._conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        await self._conn.commit()

    async def list(self, filter: MessageFilter) -> list[Message]:
        conditions: list[str] = []
        params: list[Any] = []
        if filter.thread_id is not None:
            conditions.append("thread_id = ?")
            params.append(filter.thread_id)
        if filter.session_id is not None:
            conditions.append("session_id = ?")
            params.append(filter.session_id)
        if filter.role is not None:
            conditions.append("role = ?")
            params.append(filter.role)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._conn.execute(
            f"SELECT id, thread_id, session_id, role, content, sequence, created_at"
            f" FROM messages {where} ORDER BY sequence ASC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            return [_message(row) for row in await cur.fetchall()]


class _SqliteToolCallRepo(ToolCallRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, tool_call_id: str) -> ToolCall | None:
        async with self._conn.execute(
            "SELECT id, message_id, session_id, tool_name, input, output,"
            " status, created_at, updated_at"
            " FROM tool_calls WHERE id = ?",
            (tool_call_id,),
        ) as cur:
            row = await cur.fetchone()
            return _tool_call(row) if row else None

    async def save(self, tool_call: ToolCall) -> None:
        await self._conn.execute(
            """
            INSERT INTO tool_calls
                (id, message_id, session_id, tool_name, input, output,
                 status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                message_id = excluded.message_id,
                session_id = excluded.session_id,
                tool_name  = excluded.tool_name,
                input      = excluded.input,
                output     = excluded.output,
                status     = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                tool_call.id,
                tool_call.message_id,
                tool_call.session_id,
                tool_call.tool_name,
                tool_call.input,
                tool_call.output,
                tool_call.status,
                tool_call.created_at,
                tool_call.updated_at,
            ),
        )
        await self._conn.commit()

    async def delete(self, tool_call_id: str) -> None:
        await self._conn.execute("DELETE FROM tool_calls WHERE id = ?", (tool_call_id,))
        await self._conn.commit()

    async def list(self, filter: ToolCallFilter) -> list[ToolCall]:
        conditions: list[str] = []
        params: list[Any] = []
        if filter.message_id is not None:
            conditions.append("message_id = ?")
            params.append(filter.message_id)
        if filter.session_id is not None:
            conditions.append("session_id = ?")
            params.append(filter.session_id)
        if filter.status is not None:
            conditions.append("status = ?")
            params.append(filter.status)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._conn.execute(
            f"SELECT id, message_id, session_id, tool_name, input, output,"
            f" status, created_at, updated_at"
            f" FROM tool_calls {where} ORDER BY created_at ASC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            return [_tool_call(row) for row in await cur.fetchall()]


class _SqliteSkillRepo(SkillRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, skill_id: str) -> Skill | None:
        async with self._conn.execute(
            "SELECT id, name, description, capabilities, config, created_at, updated_at"
            " FROM skills WHERE id = ?",
            (skill_id,),
        ) as cur:
            row = await cur.fetchone()
            return _skill(row) if row else None

    async def save(self, skill: Skill) -> None:
        await self._conn.execute(
            """
            INSERT INTO skills (id, name, description, capabilities, config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                name         = excluded.name,
                description  = excluded.description,
                capabilities = excluded.capabilities,
                config       = excluded.config,
                updated_at   = excluded.updated_at
            """,
            (
                skill.id,
                skill.name,
                skill.description,
                skill.capabilities,
                skill.config,
                skill.created_at,
                skill.updated_at,
            ),
        )
        await self._conn.commit()

    async def delete(self, skill_id: str) -> None:
        await self._conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        await self._conn.commit()

    async def list(self, filter: SkillFilter) -> list[Skill]:
        async with self._conn.execute(
            "SELECT id, name, description, capabilities, config, created_at, updated_at"
            " FROM skills ORDER BY name ASC LIMIT ? OFFSET ?",
            (filter.limit, filter.offset),
        ) as cur:
            return [_skill(row) for row in await cur.fetchall()]


class _SqliteEnvironmentRepo(EnvironmentRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, environment_id: str) -> Environment | None:
        async with self._conn.execute(
            "SELECT id, kind, status, config, created_at, updated_at"
            " FROM environments WHERE id = ?",
            (environment_id,),
        ) as cur:
            row = await cur.fetchone()
            return _environment(row) if row else None

    async def save(self, environment: Environment) -> None:
        await self._conn.execute(
            """
            INSERT INTO environments (id, kind, status, config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                kind       = excluded.kind,
                status     = excluded.status,
                config     = excluded.config,
                updated_at = excluded.updated_at
            """,
            (
                environment.id,
                environment.kind,
                environment.status,
                environment.config,
                environment.created_at,
                environment.updated_at,
            ),
        )
        await self._conn.commit()

    async def delete(self, environment_id: str) -> None:
        await self._conn.execute("DELETE FROM environments WHERE id = ?", (environment_id,))
        await self._conn.commit()

    async def list(self, filter: EnvironmentFilter) -> list[Environment]:
        conditions: list[str] = []
        params: list[Any] = []
        if filter.kind is not None:
            conditions.append("kind = ?")
            params.append(filter.kind)
        if filter.status is not None:
            conditions.append("status = ?")
            params.append(filter.status)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._conn.execute(
            f"SELECT id, kind, status, config, created_at, updated_at"
            f" FROM environments {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            return [_environment(row) for row in await cur.fetchall()]


class _SqliteMemoryRepo(MemoryRepository):
    def __init__(self, conn: aiosqlite.Connection, audit_log: AuditLog) -> None:
        self._conn = conn
        self._audit = audit_log

    async def get(self, entry_id: str) -> MemoryEntry | None:
        async with self._conn.execute(
            "SELECT id, scope, key, value, created_at, updated_at FROM memory_entries WHERE id = ?",
            (entry_id,),
        ) as cur:
            row = await cur.fetchone()
            return _memory_entry(row) if row else None

    async def save(self, entry: MemoryEntry) -> None:
        await self._conn.execute(
            """
            INSERT INTO memory_entries (id, scope, key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                scope      = excluded.scope,
                key        = excluded.key,
                value      = excluded.value,
                updated_at = excluded.updated_at
            """,
            (entry.id, entry.scope, entry.key, entry.value, entry.created_at, entry.updated_at),
        )
        await self._conn.commit()

    async def delete(self, entry_id: str) -> None:
        async with self._conn.execute(
            "SELECT rowid FROM memory_entries WHERE id = ?", (entry_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is not None:
            await self._conn.execute("DELETE FROM memory_entries_vec WHERE rowid = ?", (row[0],))
        await self._conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
        await self._conn.commit()

    async def list(self, filter: MemoryFilter) -> list[MemoryEntry]:
        conditions: list[str] = []
        params: list[Any] = []
        if filter.scope is not None:
            conditions.append("scope = ?")
            params.append(filter.scope)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._conn.execute(
            f"SELECT id, scope, key, value, created_at, updated_at"
            f" FROM memory_entries {where} ORDER BY key ASC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            return [_memory_entry(row) for row in await cur.fetchall()]

    async def save_embedding(self, entry_id: str, embedding: bytes) -> None:
        now = datetime.now(UTC).isoformat()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "repo.memory.save_embedding",
            attributes={
                "entity.type": "memory",
                "entity.id": entry_id,
                "repo.operation": "save_embedding",
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="repo.invocation",
                    entity_type="memory",
                    entity_id=entry_id,
                    operation="save_embedding",
                    timestamp=now,
                ),
            )
            try:
                async with self._conn.execute(
                    "SELECT rowid FROM memory_entries WHERE id = ?", (entry_id,)
                ) as cur:
                    row = await cur.fetchone()
                if row is None:
                    raise RepositoryFailure(
                        code="MEMORY_ENTRY_NOT_FOUND",
                        message=f"No memory entry with id {entry_id!r}",
                        entity_type="memory",
                        entity_id=entry_id,
                        operation="save_embedding",
                        timestamp=now,
                    )
                vec_rowid: int = row[0]
                # vec0 virtual tables do not support UPSERT, so replace any
                # existing vector for this rowid with a delete-then-insert.
                await self._conn.execute(
                    "DELETE FROM memory_entries_vec WHERE rowid = ?", (vec_rowid,)
                )
                await self._conn.execute(
                    "INSERT INTO memory_entries_vec(rowid, embedding) VALUES (?, ?)",
                    (vec_rowid, embedding),
                )
                await self._conn.commit()
            except RepositoryFailure as failure:
                record_repo_failure(span, failure)
                self._audit.write(
                    AuditLogEntry(
                        level="error",
                        event="repo.memory.save_embedding.failed",
                        entity_type=failure.entity_type,
                        entity_id=failure.entity_id,
                        operation=failure.operation,
                        timestamp=failure.timestamp,
                        detail={"code": failure.code, "message": failure.message},
                    )
                )
                raise
            except Exception as exc:
                failure = RepositoryFailure(
                    code="MEMORY_SAVE_EMBEDDING_FAILED",
                    message=str(exc),
                    entity_type="memory",
                    entity_id=entry_id,
                    operation="save_embedding",
                    timestamp=now,
                    cause=exc,
                )
                record_repo_failure(span, failure)
                self._audit.write(
                    AuditLogEntry(
                        level="error",
                        event="repo.memory.save_embedding.failed",
                        entity_type="memory",
                        entity_id=entry_id,
                        operation="save_embedding",
                        timestamp=now,
                        detail={"code": failure.code, "message": failure.message},
                    )
                )
                raise failure from exc

    async def vec_search(self, filter: MemoryVecSearchFilter) -> list[MemoryVecSearchResult]:
        now = datetime.now(UTC).isoformat()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "repo.memory.vec_search",
            attributes={
                "entity.type": "memory",
                "entity.id": "*",
                "repo.operation": "vec_search",
                "vec_search.scope": filter.scope or "",
                "vec_search.limit": filter.limit,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="repo.invocation",
                    entity_type="memory",
                    entity_id="*",
                    operation="vec_search",
                    timestamp=now,
                ),
            )
            try:
                candidates = filter.limit * 10
                if filter.scope is not None:
                    sql = """
                        SELECT m.id, m.scope, m.key, m.value, m.created_at, m.updated_at,
                               v.distance
                        FROM (
                            SELECT rowid, distance
                            FROM memory_entries_vec
                            WHERE embedding MATCH ? AND k = ?
                            ORDER BY distance
                        ) v
                        JOIN memory_entries m ON m.rowid = v.rowid
                        WHERE m.scope = ?
                        ORDER BY v.distance
                        LIMIT ?
                    """
                    params: list[Any] = [filter.embedding, candidates, filter.scope, filter.limit]
                else:
                    sql = """
                        SELECT m.id, m.scope, m.key, m.value, m.created_at, m.updated_at,
                               v.distance
                        FROM (
                            SELECT rowid, distance
                            FROM memory_entries_vec
                            WHERE embedding MATCH ? AND k = ?
                            ORDER BY distance
                        ) v
                        JOIN memory_entries m ON m.rowid = v.rowid
                        ORDER BY v.distance
                        LIMIT ?
                    """
                    params = [filter.embedding, candidates, filter.limit]

                async with self._conn.execute(sql, params) as cur:
                    rows = await cur.fetchall()

                results = [
                    MemoryVecSearchResult(
                        entry=_memory_entry(row),
                        distance=row[6],
                    )
                    for row in rows
                ]
                span.set_attribute("vec_search.result_count", len(results))
                return results

            except Exception as exc:
                failure = RepositoryFailure(
                    code="MEMORY_VEC_SEARCH_FAILED",
                    message=str(exc),
                    entity_type="memory",
                    entity_id="*",
                    operation="vec_search",
                    timestamp=now,
                    cause=exc,
                )
                record_repo_failure(span, failure)
                self._audit.write(
                    AuditLogEntry(
                        level="error",
                        event="repo.memory.vec_search.failed",
                        entity_type="memory",
                        entity_id="*",
                        operation="vec_search",
                        timestamp=now,
                        detail={"code": failure.code, "message": failure.message},
                    )
                )
                raise failure from exc


class _SqliteVaultRepo(VaultRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, entry_id: str) -> VaultEntry | None:
        async with self._conn.execute(
            "SELECT id, name, description, created_at, updated_at FROM vault_entries WHERE id = ?",
            (entry_id,),
        ) as cur:
            row = await cur.fetchone()
            return _vault_entry(row) if row else None

    async def save(self, entry: VaultEntry) -> None:
        await self._conn.execute(
            """
            INSERT INTO vault_entries (id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                name        = excluded.name,
                description = excluded.description,
                updated_at  = excluded.updated_at
            """,
            (entry.id, entry.name, entry.description, entry.created_at, entry.updated_at),
        )
        await self._conn.commit()

    async def delete(self, entry_id: str) -> None:
        await self._conn.execute("DELETE FROM vault_entries WHERE id = ?", (entry_id,))
        await self._conn.commit()

    async def list(self, filter: VaultFilter) -> list[VaultEntry]:
        async with self._conn.execute(
            "SELECT id, name, description, created_at, updated_at"
            " FROM vault_entries ORDER BY name ASC LIMIT ? OFFSET ?",
            (filter.limit, filter.offset),
        ) as cur:
            return [_vault_entry(row) for row in await cur.fetchall()]


class _SqliteUserProfileRepo(UserProfileRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, user_id: str) -> UserProfile | None:
        async with self._conn.execute(
            "SELECT id, username, display_name, email, metadata, is_primary, created_at, updated_at"
            " FROM user_profiles WHERE id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            return _user_profile(row) if row else None

    async def save(self, profile: UserProfile) -> None:
        await self._conn.execute(
            """
            INSERT INTO user_profiles
                (id, username, display_name, email, metadata, is_primary, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                username     = excluded.username,
                display_name = excluded.display_name,
                email        = excluded.email,
                metadata     = excluded.metadata,
                is_primary   = excluded.is_primary,
                updated_at   = excluded.updated_at
            """,
            (
                profile.id,
                profile.username,
                profile.display_name,
                profile.email,
                profile.metadata,
                1 if profile.is_primary else 0,
                profile.created_at,
                profile.updated_at,
            ),
        )
        await self._conn.commit()

    async def delete(self, user_id: str) -> None:
        await self._conn.execute("DELETE FROM user_profiles WHERE id = ?", (user_id,))
        await self._conn.commit()

    async def list(self, filter: UserProfileFilter) -> list[UserProfile]:
        async with self._conn.execute(
            "SELECT id, username, display_name, email, metadata, is_primary, created_at, updated_at"
            " FROM user_profiles ORDER BY username ASC LIMIT ? OFFSET ?",
            (filter.limit, filter.offset),
        ) as cur:
            return [_user_profile(row) for row in await cur.fetchall()]


class _SqliteChannelRepo(ChannelRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, channel_id: str) -> Channel | None:
        async with self._conn.execute(
            "SELECT id, kind, name, config, status, created_at, updated_at"
            " FROM channels WHERE id = ?",
            (channel_id,),
        ) as cur:
            row = await cur.fetchone()
            return _channel(row) if row else None

    async def save(self, channel: Channel) -> None:
        await self._conn.execute(
            """
            INSERT INTO channels (id, kind, name, config, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                kind       = excluded.kind,
                name       = excluded.name,
                config     = excluded.config,
                status     = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                channel.id,
                channel.kind,
                channel.name,
                channel.config,
                channel.status,
                channel.created_at,
                channel.updated_at,
            ),
        )
        await self._conn.commit()

    async def delete(self, channel_id: str) -> None:
        await self._conn.execute("DELETE FROM channels WHERE id = ?", (channel_id,))
        await self._conn.commit()

    async def list(self, filter: ChannelFilter) -> list[Channel]:
        conditions: list[str] = []
        params: list[Any] = []
        if filter.kind is not None:
            conditions.append("kind = ?")
            params.append(filter.kind)
        if filter.status is not None:
            conditions.append("status = ?")
            params.append(filter.status)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._conn.execute(
            f"SELECT id, kind, name, config, status, created_at, updated_at"
            f" FROM channels {where} ORDER BY name ASC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            return [_channel(row) for row in await cur.fetchall()]


class _SqliteWebhookRepo(WebhookRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, webhook_id: str) -> Webhook | None:
        async with self._conn.execute(
            "SELECT id, url, events, secret_ref, status, created_at, updated_at"
            " FROM webhooks WHERE id = ?",
            (webhook_id,),
        ) as cur:
            row = await cur.fetchone()
            return _webhook(row) if row else None

    async def save(self, webhook: Webhook) -> None:
        await self._conn.execute(
            """
            INSERT INTO webhooks (id, url, events, secret_ref, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                url        = excluded.url,
                events     = excluded.events,
                secret_ref = excluded.secret_ref,
                status     = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                webhook.id,
                webhook.url,
                webhook.events,
                webhook.secret_ref,
                webhook.status,
                webhook.created_at,
                webhook.updated_at,
            ),
        )
        await self._conn.commit()

    async def delete(self, webhook_id: str) -> None:
        await self._conn.execute("DELETE FROM webhooks WHERE id = ?", (webhook_id,))
        await self._conn.commit()

    async def list(self, filter: WebhookFilter) -> list[Webhook]:
        conditions: list[str] = []
        params: list[Any] = []
        if filter.status is not None:
            conditions.append("status = ?")
            params.append(filter.status)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._conn.execute(
            f"SELECT id, url, events, secret_ref, status, created_at, updated_at"
            f" FROM webhooks {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            return [_webhook(row) for row in await cur.fetchall()]


class _SqliteAuditLogEntryRepo(AuditLogEntryRepository):
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def append(self, entry: AuditLogEntryRecord) -> None:
        await self._conn.execute(
            """
            INSERT INTO audit_log_entries
                (id, level, event, entity_type, entity_id, operation, timestamp, detail, signature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.level,
                entry.event,
                entry.entity_type,
                entry.entity_id,
                entry.operation,
                entry.timestamp,
                json.dumps(entry.detail) if entry.detail is not None else None,
                entry.signature,
            ),
        )
        await self._conn.commit()

    async def list(self, filter: AuditLogEntryFilter) -> list[AuditLogEntryRecord]:
        conditions: list[str] = []
        params: list[Any] = []
        if filter.level is not None:
            conditions.append("level = ?")
            params.append(filter.level)
        if filter.event is not None:
            conditions.append("event = ?")
            params.append(filter.event)
        if filter.entity_type is not None:
            conditions.append("entity_type = ?")
            params.append(filter.entity_type)
        if filter.entity_id is not None:
            conditions.append("entity_id = ?")
            params.append(filter.entity_id)
        if filter.since is not None:
            conditions.append("timestamp >= ?")
            params.append(filter.since)
        if filter.until is not None:
            conditions.append("timestamp <= ?")
            params.append(filter.until)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._conn.execute(
            "SELECT id, level, event, entity_type, entity_id, operation,"
            " timestamp, detail, signature"
            f" FROM audit_log_entries {where} ORDER BY timestamp ASC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            return [_audit_log_entry_record(row) for row in await cur.fetchall()]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class SqliteRepositoryDriver(RepositoryDriver):
    """
    SQLite-backed RepositoryDriver.

    Obtain an instance via the async class method:
        driver = await SqliteRepositoryDriver.open(":memory:")
        await driver.migrate()

    Close when done:
        await driver.close()
    """

    def __init__(self, conn: aiosqlite.Connection, audit_log: AuditLog, db_path: str = "") -> None:
        self._conn = conn
        self._db_path = db_path
        self._audit = audit_log
        self._agents = _SqliteAgentRepo(conn)
        self._sessions = _SqliteSessionRepo(conn)
        self._threads = _SqliteThreadRepo(conn)
        self._messages = _SqliteMessageRepo(conn)
        self._tool_calls = _SqliteToolCallRepo(conn)
        self._skills = _SqliteSkillRepo(conn)
        self._environments = _SqliteEnvironmentRepo(conn)
        self._memory = _SqliteMemoryRepo(conn, audit_log)
        self._vault = _SqliteVaultRepo(conn)
        self._user_profiles = _SqliteUserProfileRepo(conn)
        self._channels = _SqliteChannelRepo(conn)
        self._webhooks = _SqliteWebhookRepo(conn)
        self._audit_log_entries = _SqliteAuditLogEntryRepo(conn)

    @classmethod
    async def open(
        cls,
        db_path: str | Path = ":memory:",
        *,
        audit_log: AuditLog | None = None,
    ) -> SqliteRepositoryDriver:
        """Open (or create) a WAL-mode SQLite database and return a driver instance."""
        _audit = audit_log if audit_log is not None else NoopAuditLog()
        path_str = str(db_path)
        now = datetime.now(UTC).isoformat()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "sqlite.open",
            attributes={"db.system": "sqlite", "db.path": path_str},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="sqlite.invocation",
                    entity_type="sqlite_driver",
                    entity_id=path_str,
                    operation="open",
                    timestamp=now,
                ),
            )
            conn: aiosqlite.Connection | None = None
            try:
                conn = await aiosqlite.connect(path_str)
                conn.row_factory = aiosqlite.Row
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA busy_timeout=5000")
                await conn.execute("PRAGMA synchronous=NORMAL")
                await conn.execute("PRAGMA foreign_keys=ON")
                await conn._execute(_load_sqlite_vec, conn._conn)
                return cls(conn, _audit, path_str)
            except Exception as exc:
                if conn is not None:
                    await conn.close()
                failure = RepositoryFailure(
                    code="SQLITE_OPEN_FAILED",
                    message=str(exc),
                    entity_type="sqlite_driver",
                    entity_id=path_str,
                    operation="open",
                    timestamp=now,
                    cause=exc,
                )
                record_repo_failure(span, failure)
                _audit.write(
                    AuditLogEntry(
                        level="error",
                        event="sqlite.driver.open.failed",
                        entity_type="sqlite_driver",
                        entity_id=path_str,
                        operation="open",
                        timestamp=now,
                        detail={"code": failure.code, "message": failure.message},
                    )
                )
                raise failure from exc

    @property
    def agents(self) -> AgentRepository:
        return self._agents

    @property
    def sessions(self) -> SessionRepository:
        return self._sessions

    @property
    def threads(self) -> ThreadRepository:
        return self._threads

    @property
    def messages(self) -> MessageRepository:
        return self._messages

    @property
    def tool_calls(self) -> ToolCallRepository:
        return self._tool_calls

    @property
    def skills(self) -> SkillRepository:
        return self._skills

    @property
    def environments(self) -> EnvironmentRepository:
        return self._environments

    @property
    def memory(self) -> MemoryRepository:
        return self._memory

    @property
    def vault(self) -> VaultRepository:
        return self._vault

    @property
    def user_profiles(self) -> UserProfileRepository:
        return self._user_profiles

    @property
    def channels(self) -> ChannelRepository:
        return self._channels

    @property
    def webhooks(self) -> WebhookRepository:
        return self._webhooks

    @property
    def audit_log_entries(self) -> AuditLogEntryRepository:
        return self._audit_log_entries

    async def migrate(self) -> None:
        """Apply pending SQL migrations from db/migrations/*.sql in order.

        Creates a schema_migrations tracking table on first run.  Raises
        RepositoryFailure with code SCHEMA_VERSION_AHEAD if the recorded DB
        version exceeds the binary's supported SCHEMA_VERSION.
        """
        now = datetime.now(UTC).isoformat()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "sqlite.migrate",
            attributes={
                "db.system": "sqlite",
                "db.path": self._db_path,
                "schema.version.supported": SCHEMA_VERSION,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="sqlite.invocation",
                    entity_type="sqlite_driver",
                    entity_id=self._db_path,
                    operation="migrate",
                    timestamp=now,
                ),
            )
            try:
                await self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version    INTEGER PRIMARY KEY,
                        filename   TEXT    NOT NULL,
                        applied_at TEXT    NOT NULL
                    )
                    """
                )
                await self._conn.commit()

                async with self._conn.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
                ) as cur:
                    row = await cur.fetchone()
                db_version: int = row[0] if row else 0

                if db_version > SCHEMA_VERSION:
                    raise RepositoryFailure(
                        code="SCHEMA_VERSION_AHEAD",
                        message=(
                            f"Schema version on disk ({db_version}) exceeds "
                            f"binary's supported version ({SCHEMA_VERSION}). "
                            "Upgrade the binary before starting the daemon."
                        ),
                        entity_type="sqlite_driver",
                        entity_id=self._db_path,
                        operation="migrate",
                        timestamp=now,
                    )

                pending = [
                    (v, fname, sql) for v, fname, sql in load_migration_files() if v > db_version
                ]
                for version, filename, sql in pending:
                    stmts = [s.strip() for s in sql.split(";") if s.strip()]
                    for stmt in stmts:
                        try:
                            await self._conn.execute(stmt)
                        except Exception as stmt_exc:
                            if "no such module" in str(stmt_exc).lower():
                                continue
                            raise
                    await self._conn.execute(
                        "INSERT INTO schema_migrations (version, filename, applied_at)"
                        " VALUES (?, ?, ?)",
                        (version, filename, datetime.now(UTC).isoformat()),
                    )
                    await self._conn.commit()

                span.set_attribute("schema.version.db", db_version)
                span.set_attribute("schema.version.applied_count", len(pending))

            except RepositoryFailure as failure:
                record_repo_failure(span, failure)
                self._audit.write(
                    AuditLogEntry(
                        level="error",
                        event="sqlite.driver.migrate.failed",
                        entity_type=failure.entity_type,
                        entity_id=failure.entity_id,
                        operation=failure.operation,
                        timestamp=failure.timestamp,
                        detail={"code": failure.code, "message": failure.message},
                    )
                )
                raise
            except Exception as exc:
                failure = RepositoryFailure(
                    code="SQLITE_MIGRATE_FAILED",
                    message=str(exc),
                    entity_type="sqlite_driver",
                    entity_id=self._db_path,
                    operation="migrate",
                    timestamp=now,
                    cause=exc,
                )
                record_repo_failure(span, failure)
                self._audit.write(
                    AuditLogEntry(
                        level="error",
                        event="sqlite.driver.migrate.failed",
                        entity_type="sqlite_driver",
                        entity_id=self._db_path,
                        operation="migrate",
                        timestamp=now,
                        detail={"code": failure.code, "message": failure.message},
                    )
                )
                raise failure from exc

    async def close(self) -> None:
        """Close the underlying aiosqlite connection."""
        await self._conn.close()
