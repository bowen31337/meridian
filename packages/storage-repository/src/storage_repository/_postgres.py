"""
Postgres backend for all 12 system repositories.

Requires the optional `asyncpg` dependency (pip install meridian-storage-repository[postgres]).

Usage:
    driver = await PostgresRepositoryDriver.open("postgresql://user:pass@localhost/db")
    await driver.migrate()
    runtime = RepositoryRuntime(driver)
    await runtime.agents.save(Agent(...))
    await driver.close()

Upsert syntax uses the ANSI-standard ON CONFLICT DO UPDATE clause.
Placeholders are $1, $2, … (asyncpg style).
"""

from __future__ import annotations

from typing import Any

import asyncpg

from ._contract import (
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
from ._migrations import MIGRATIONS
from ._runtime import RepositoryDriver
from ._types import (
    Agent,
    AgentFilter,
    Channel,
    ChannelFilter,
    Environment,
    EnvironmentFilter,
    MemoryEntry,
    MemoryFilter,
    Message,
    MessageFilter,
    Session,
    SessionFilter,
    Skill,
    SkillFilter,
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

Record = asyncpg.Record


# ---------------------------------------------------------------------------
# Row converters  (asyncpg.Record supports index access like a tuple)
# ---------------------------------------------------------------------------


def _agent(r: Record) -> Agent:
    return Agent(
        id=r[0],
        kind=r[1],
        name=r[2],
        config=r[3],
        capabilities=r[4],
        created_at=r[5],
        updated_at=r[6],
    )


def _session(r: Record) -> Session:
    return Session(
        id=r[0], agent_id=r[1], status=r[2], metadata=r[3], created_at=r[4], updated_at=r[5]
    )


def _thread(r: Record) -> Thread:
    return Thread(id=r[0], session_id=r[1], title=r[2], created_at=r[3], updated_at=r[4])


def _message(r: Record) -> Message:
    return Message(
        id=r[0],
        thread_id=r[1],
        session_id=r[2],
        role=r[3],
        content=r[4],
        sequence=r[5],
        created_at=r[6],
    )


def _tool_call(r: Record) -> ToolCall:
    return ToolCall(
        id=r[0],
        message_id=r[1],
        session_id=r[2],
        tool_name=r[3],
        input=r[4],
        output=r[5],
        status=r[6],
        created_at=r[7],
        updated_at=r[8],
    )


def _skill(r: Record) -> Skill:
    return Skill(
        id=r[0],
        name=r[1],
        description=r[2],
        capabilities=r[3],
        config=r[4],
        created_at=r[5],
        updated_at=r[6],
    )


def _environment(r: Record) -> Environment:
    return Environment(
        id=r[0], kind=r[1], status=r[2], config=r[3], created_at=r[4], updated_at=r[5]
    )


def _memory_entry(r: Record) -> MemoryEntry:
    return MemoryEntry(id=r[0], scope=r[1], key=r[2], value=r[3], created_at=r[4], updated_at=r[5])


def _vault_entry(r: Record) -> VaultEntry:
    return VaultEntry(id=r[0], name=r[1], description=r[2], created_at=r[3], updated_at=r[4])


def _user_profile(r: Record) -> UserProfile:
    return UserProfile(
        id=r[0],
        username=r[1],
        display_name=r[2],
        email=r[3],
        metadata=r[4],
        is_primary=bool(r[5]),
        created_at=r[6],
        updated_at=r[7],
    )


def _channel(r: Record) -> Channel:
    return Channel(
        id=r[0], kind=r[1], name=r[2], config=r[3], status=r[4], created_at=r[5], updated_at=r[6]
    )


def _webhook(r: Record) -> Webhook:
    return Webhook(
        id=r[0],
        url=r[1],
        events=r[2],
        secret_ref=r[3],
        status=r[4],
        created_at=r[5],
        updated_at=r[6],
    )


# ---------------------------------------------------------------------------
# Repository implementations
# ---------------------------------------------------------------------------


class _PgAgentRepo(AgentRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, agent_id: str) -> Agent | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, kind, name, config, capabilities, created_at, updated_at"
                " FROM agents WHERE id = $1",
                agent_id,
            )
            return _agent(row) if row else None

    async def save(self, agent: Agent) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agents (id, kind, name, config, capabilities, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    kind         = EXCLUDED.kind,
                    name         = EXCLUDED.name,
                    config       = EXCLUDED.config,
                    capabilities = EXCLUDED.capabilities,
                    updated_at   = EXCLUDED.updated_at
                """,
                agent.id,
                agent.kind,
                agent.name,
                agent.config,
                agent.capabilities,
                agent.created_at,
                agent.updated_at,
            )

    async def delete(self, agent_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM agents WHERE id = $1", agent_id)

    async def list(self, filter: AgentFilter) -> list[Agent]:
        conditions: list[str] = []
        params: list[Any] = []
        n = 1
        if filter.kind is not None:
            conditions.append(f"kind = ${n}")
            params.append(filter.kind)
            n += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, kind, name, config, capabilities, created_at, updated_at"
                f" FROM agents {where} ORDER BY created_at DESC LIMIT ${n} OFFSET ${n + 1}",
                *params,
            )
            return [_agent(r) for r in rows]


class _PgSessionRepo(SessionRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, session_id: str) -> Session | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, agent_id, status, metadata, created_at, updated_at"
                " FROM sessions WHERE id = $1",
                session_id,
            )
            return _session(row) if row else None

    async def save(self, session: Session) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (id, agent_id, status, metadata, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    agent_id   = EXCLUDED.agent_id,
                    status     = EXCLUDED.status,
                    metadata   = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                """,
                session.id,
                session.agent_id,
                session.status,
                session.metadata,
                session.created_at,
                session.updated_at,
            )

    async def delete(self, session_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)

    async def list(self, filter: SessionFilter) -> list[Session]:
        conditions: list[str] = []
        params: list[Any] = []
        n = 1
        if filter.agent_id is not None:
            conditions.append(f"agent_id = ${n}")
            params.append(filter.agent_id)
            n += 1
        if filter.status is not None:
            conditions.append(f"status = ${n}")
            params.append(filter.status)
            n += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, agent_id, status, metadata, created_at, updated_at"
                f" FROM sessions {where} ORDER BY created_at DESC LIMIT ${n} OFFSET ${n + 1}",
                *params,
            )
            return [_session(r) for r in rows]


class _PgThreadRepo(ThreadRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, thread_id: str) -> Thread | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, session_id, title, created_at, updated_at FROM threads WHERE id = $1",
                thread_id,
            )
            return _thread(row) if row else None

    async def save(self, thread: Thread) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO threads (id, session_id, title, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    title      = EXCLUDED.title,
                    updated_at = EXCLUDED.updated_at
                """,
                thread.id,
                thread.session_id,
                thread.title,
                thread.created_at,
                thread.updated_at,
            )

    async def delete(self, thread_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM threads WHERE id = $1", thread_id)

    async def list(self, filter: ThreadFilter) -> list[Thread]:
        conditions: list[str] = []
        params: list[Any] = []
        n = 1
        if filter.session_id is not None:
            conditions.append(f"session_id = ${n}")
            params.append(filter.session_id)
            n += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, session_id, title, created_at, updated_at"
                f" FROM threads {where} ORDER BY created_at DESC LIMIT ${n} OFFSET ${n + 1}",
                *params,
            )
            return [_thread(r) for r in rows]


class _PgMessageRepo(MessageRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, message_id: str) -> Message | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, thread_id, session_id, role, content, sequence, created_at"
                " FROM messages WHERE id = $1",
                message_id,
            )
            return _message(row) if row else None

    async def save(self, message: Message) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages
                    (id, thread_id, session_id, role, content, sequence, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    thread_id  = EXCLUDED.thread_id,
                    session_id = EXCLUDED.session_id,
                    role       = EXCLUDED.role,
                    content    = EXCLUDED.content,
                    sequence   = EXCLUDED.sequence
                """,
                message.id,
                message.thread_id,
                message.session_id,
                message.role,
                message.content,
                message.sequence,
                message.created_at,
            )

    async def delete(self, message_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM messages WHERE id = $1", message_id)

    async def list(self, filter: MessageFilter) -> list[Message]:
        conditions: list[str] = []
        params: list[Any] = []
        n = 1
        if filter.thread_id is not None:
            conditions.append(f"thread_id = ${n}")
            params.append(filter.thread_id)
            n += 1
        if filter.session_id is not None:
            conditions.append(f"session_id = ${n}")
            params.append(filter.session_id)
            n += 1
        if filter.role is not None:
            conditions.append(f"role = ${n}")
            params.append(filter.role)
            n += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, thread_id, session_id, role, content, sequence, created_at"
                f" FROM messages {where} ORDER BY sequence ASC LIMIT ${n} OFFSET ${n + 1}",
                *params,
            )
            return [_message(r) for r in rows]


class _PgToolCallRepo(ToolCallRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, tool_call_id: str) -> ToolCall | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, message_id, session_id, tool_name, input, output,"
                " status, created_at, updated_at"
                " FROM tool_calls WHERE id = $1",
                tool_call_id,
            )
            return _tool_call(row) if row else None

    async def save(self, tool_call: ToolCall) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tool_calls
                    (id, message_id, session_id, tool_name, input, output,
                     status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    message_id = EXCLUDED.message_id,
                    session_id = EXCLUDED.session_id,
                    tool_name  = EXCLUDED.tool_name,
                    input      = EXCLUDED.input,
                    output     = EXCLUDED.output,
                    status     = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
                """,
                tool_call.id,
                tool_call.message_id,
                tool_call.session_id,
                tool_call.tool_name,
                tool_call.input,
                tool_call.output,
                tool_call.status,
                tool_call.created_at,
                tool_call.updated_at,
            )

    async def delete(self, tool_call_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM tool_calls WHERE id = $1", tool_call_id)

    async def list(self, filter: ToolCallFilter) -> list[ToolCall]:
        conditions: list[str] = []
        params: list[Any] = []
        n = 1
        if filter.message_id is not None:
            conditions.append(f"message_id = ${n}")
            params.append(filter.message_id)
            n += 1
        if filter.session_id is not None:
            conditions.append(f"session_id = ${n}")
            params.append(filter.session_id)
            n += 1
        if filter.status is not None:
            conditions.append(f"status = ${n}")
            params.append(filter.status)
            n += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, message_id, session_id, tool_name, input, output,"
                f" status, created_at, updated_at"
                f" FROM tool_calls {where} ORDER BY created_at ASC LIMIT ${n} OFFSET ${n + 1}",
                *params,
            )
            return [_tool_call(r) for r in rows]


class _PgSkillRepo(SkillRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, skill_id: str) -> Skill | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, name, description, capabilities, config, created_at, updated_at"
                " FROM skills WHERE id = $1",
                skill_id,
            )
            return _skill(row) if row else None

    async def save(self, skill: Skill) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO skills
                    (id, name, description, capabilities, config, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    name         = EXCLUDED.name,
                    description  = EXCLUDED.description,
                    capabilities = EXCLUDED.capabilities,
                    config       = EXCLUDED.config,
                    updated_at   = EXCLUDED.updated_at
                """,
                skill.id,
                skill.name,
                skill.description,
                skill.capabilities,
                skill.config,
                skill.created_at,
                skill.updated_at,
            )

    async def delete(self, skill_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM skills WHERE id = $1", skill_id)

    async def list(self, filter: SkillFilter) -> list[Skill]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, description, capabilities, config, created_at, updated_at"
                " FROM skills ORDER BY name ASC LIMIT $1 OFFSET $2",
                filter.limit,
                filter.offset,
            )
            return [_skill(r) for r in rows]


class _PgEnvironmentRepo(EnvironmentRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, environment_id: str) -> Environment | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, kind, status, config, created_at, updated_at"
                " FROM environments WHERE id = $1",
                environment_id,
            )
            return _environment(row) if row else None

    async def save(self, environment: Environment) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO environments (id, kind, status, config, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    kind       = EXCLUDED.kind,
                    status     = EXCLUDED.status,
                    config     = EXCLUDED.config,
                    updated_at = EXCLUDED.updated_at
                """,
                environment.id,
                environment.kind,
                environment.status,
                environment.config,
                environment.created_at,
                environment.updated_at,
            )

    async def delete(self, environment_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM environments WHERE id = $1", environment_id)

    async def list(self, filter: EnvironmentFilter) -> list[Environment]:
        conditions: list[str] = []
        params: list[Any] = []
        n = 1
        if filter.kind is not None:
            conditions.append(f"kind = ${n}")
            params.append(filter.kind)
            n += 1
        if filter.status is not None:
            conditions.append(f"status = ${n}")
            params.append(filter.status)
            n += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, kind, status, config, created_at, updated_at"
                f" FROM environments {where} ORDER BY created_at DESC LIMIT ${n} OFFSET ${n + 1}",
                *params,
            )
            return [_environment(r) for r in rows]


class _PgMemoryRepo(MemoryRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, entry_id: str) -> MemoryEntry | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, scope, key, value, created_at, updated_at"
                " FROM memory_entries WHERE id = $1",
                entry_id,
            )
            return _memory_entry(row) if row else None

    async def save(self, entry: MemoryEntry) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_entries (id, scope, key, value, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    scope      = EXCLUDED.scope,
                    key        = EXCLUDED.key,
                    value      = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
                """,
                entry.id,
                entry.scope,
                entry.key,
                entry.value,
                entry.created_at,
                entry.updated_at,
            )

    async def delete(self, entry_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_entries WHERE id = $1", entry_id)

    async def list(self, filter: MemoryFilter) -> list[MemoryEntry]:
        conditions: list[str] = []
        params: list[Any] = []
        n = 1
        if filter.scope is not None:
            conditions.append(f"scope = ${n}")
            params.append(filter.scope)
            n += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, scope, key, value, created_at, updated_at"
                f" FROM memory_entries {where} ORDER BY key ASC LIMIT ${n} OFFSET ${n + 1}",
                *params,
            )
            return [_memory_entry(r) for r in rows]


class _PgVaultRepo(VaultRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, entry_id: str) -> VaultEntry | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, name, description, created_at, updated_at"
                " FROM vault_entries WHERE id = $1",
                entry_id,
            )
            return _vault_entry(row) if row else None

    async def save(self, entry: VaultEntry) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO vault_entries (id, name, description, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    name        = EXCLUDED.name,
                    description = EXCLUDED.description,
                    updated_at  = EXCLUDED.updated_at
                """,
                entry.id,
                entry.name,
                entry.description,
                entry.created_at,
                entry.updated_at,
            )

    async def delete(self, entry_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM vault_entries WHERE id = $1", entry_id)

    async def list(self, filter: VaultFilter) -> list[VaultEntry]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, description, created_at, updated_at"
                " FROM vault_entries ORDER BY name ASC LIMIT $1 OFFSET $2",
                filter.limit,
                filter.offset,
            )
            return [_vault_entry(r) for r in rows]


class _PgUserProfileRepo(UserProfileRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, user_id: str) -> UserProfile | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, username, display_name, email, metadata, is_primary,"
                " created_at, updated_at"
                " FROM user_profiles WHERE id = $1",
                user_id,
            )
            return _user_profile(row) if row else None

    async def save(self, profile: UserProfile) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_profiles
                    (id, username, display_name, email, metadata, is_primary,
                     created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    username     = EXCLUDED.username,
                    display_name = EXCLUDED.display_name,
                    email        = EXCLUDED.email,
                    metadata     = EXCLUDED.metadata,
                    is_primary   = EXCLUDED.is_primary,
                    updated_at   = EXCLUDED.updated_at
                """,
                profile.id,
                profile.username,
                profile.display_name,
                profile.email,
                profile.metadata,
                profile.is_primary,
                profile.created_at,
                profile.updated_at,
            )

    async def delete(self, user_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM user_profiles WHERE id = $1", user_id)

    async def list(self, filter: UserProfileFilter) -> list[UserProfile]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, username, display_name, email, metadata, is_primary,"
                " created_at, updated_at"
                " FROM user_profiles ORDER BY username ASC LIMIT $1 OFFSET $2",
                filter.limit,
                filter.offset,
            )
            return [_user_profile(r) for r in rows]


class _PgChannelRepo(ChannelRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, channel_id: str) -> Channel | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, kind, name, config, status, created_at, updated_at"
                " FROM channels WHERE id = $1",
                channel_id,
            )
            return _channel(row) if row else None

    async def save(self, channel: Channel) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO channels (id, kind, name, config, status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    kind       = EXCLUDED.kind,
                    name       = EXCLUDED.name,
                    config     = EXCLUDED.config,
                    status     = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
                """,
                channel.id,
                channel.kind,
                channel.name,
                channel.config,
                channel.status,
                channel.created_at,
                channel.updated_at,
            )

    async def delete(self, channel_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM channels WHERE id = $1", channel_id)

    async def list(self, filter: ChannelFilter) -> list[Channel]:
        conditions: list[str] = []
        params: list[Any] = []
        n = 1
        if filter.kind is not None:
            conditions.append(f"kind = ${n}")
            params.append(filter.kind)
            n += 1
        if filter.status is not None:
            conditions.append(f"status = ${n}")
            params.append(filter.status)
            n += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, kind, name, config, status, created_at, updated_at"
                f" FROM channels {where} ORDER BY name ASC LIMIT ${n} OFFSET ${n + 1}",
                *params,
            )
            return [_channel(r) for r in rows]


class _PgWebhookRepo(WebhookRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, webhook_id: str) -> Webhook | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, url, events, secret_ref, status, created_at, updated_at"
                " FROM webhooks WHERE id = $1",
                webhook_id,
            )
            return _webhook(row) if row else None

    async def save(self, webhook: Webhook) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO webhooks (id, url, events, secret_ref, status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    url        = EXCLUDED.url,
                    events     = EXCLUDED.events,
                    secret_ref = EXCLUDED.secret_ref,
                    status     = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
                """,
                webhook.id,
                webhook.url,
                webhook.events,
                webhook.secret_ref,
                webhook.status,
                webhook.created_at,
                webhook.updated_at,
            )

    async def delete(self, webhook_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM webhooks WHERE id = $1", webhook_id)

    async def list(self, filter: WebhookFilter) -> list[Webhook]:
        conditions: list[str] = []
        params: list[Any] = []
        n = 1
        if filter.status is not None:
            conditions.append(f"status = ${n}")
            params.append(filter.status)
            n += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([filter.limit, filter.offset])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, url, events, secret_ref, status, created_at, updated_at"
                f" FROM webhooks {where} ORDER BY created_at DESC LIMIT ${n} OFFSET ${n + 1}",
                *params,
            )
            return [_webhook(r) for r in rows]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class PostgresRepositoryDriver(RepositoryDriver):
    """
    Postgres-backed RepositoryDriver backed by an asyncpg connection pool.

    Obtain an instance via the async class method:
        driver = await PostgresRepositoryDriver.open("postgresql://user:pass@host/db")
        await driver.migrate()

    Close when done:
        await driver.close()
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._agents = _PgAgentRepo(pool)
        self._sessions = _PgSessionRepo(pool)
        self._threads = _PgThreadRepo(pool)
        self._messages = _PgMessageRepo(pool)
        self._tool_calls = _PgToolCallRepo(pool)
        self._skills = _PgSkillRepo(pool)
        self._environments = _PgEnvironmentRepo(pool)
        self._memory = _PgMemoryRepo(pool)
        self._vault = _PgVaultRepo(pool)
        self._user_profiles = _PgUserProfileRepo(pool)
        self._channels = _PgChannelRepo(pool)
        self._webhooks = _PgWebhookRepo(pool)

    @classmethod
    async def open(cls, dsn: str, **pool_kwargs: Any) -> PostgresRepositoryDriver:
        """Create an asyncpg connection pool and return a driver instance."""
        pool = await asyncpg.create_pool(dsn, **pool_kwargs)
        return cls(pool)

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

    async def migrate(self) -> None:
        """Execute all DDL migration statements idempotently."""
        async with self._pool.acquire() as conn:
            for stmt in MIGRATIONS:
                await conn.execute(stmt)

    async def close(self) -> None:
        """Close the asyncpg connection pool."""
        await self._pool.close()
