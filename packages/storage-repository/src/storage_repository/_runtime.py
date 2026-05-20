from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from typing import TypeVar

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

T = TypeVar("T")


@dataclass
class RepositoryOptions:
    """Options supplied by the host application to RepositoryRuntime."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[RepositoryFailure], None] | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# RepositoryDriver — bundles all 12 repository contracts
# ---------------------------------------------------------------------------


class RepositoryDriver(ABC):
    """
    Container for all 12 concrete repository implementations.

    Instantiate a backend-specific subclass (SqliteRepositoryDriver or
    PostgresRepositoryDriver), call migrate() once to apply DDL, then pass it
    to RepositoryRuntime which wraps every operation with OTel tracing and
    audit-log writes on failure.
    """

    @property
    @abstractmethod
    def agents(self) -> AgentRepository: ...

    @property
    @abstractmethod
    def sessions(self) -> SessionRepository: ...

    @property
    @abstractmethod
    def threads(self) -> ThreadRepository: ...

    @property
    @abstractmethod
    def messages(self) -> MessageRepository: ...

    @property
    @abstractmethod
    def tool_calls(self) -> ToolCallRepository: ...

    @property
    @abstractmethod
    def skills(self) -> SkillRepository: ...

    @property
    @abstractmethod
    def environments(self) -> EnvironmentRepository: ...

    @property
    @abstractmethod
    def memory(self) -> MemoryRepository: ...

    @property
    @abstractmethod
    def vault(self) -> VaultRepository: ...

    @property
    @abstractmethod
    def user_profiles(self) -> UserProfileRepository: ...

    @property
    @abstractmethod
    def channels(self) -> ChannelRepository: ...

    @property
    @abstractmethod
    def webhooks(self) -> WebhookRepository: ...

    @property
    @abstractmethod
    def audit_log_entries(self) -> AuditLogEntryRepository: ...

    @abstractmethod
    async def migrate(self) -> None:
        """Apply all pending DDL migrations. Call once before any repository use."""

    @abstractmethod
    async def close(self) -> None:
        """Release underlying database connections."""


# ---------------------------------------------------------------------------
# Generic tracing helper
# ---------------------------------------------------------------------------


async def _trace_op(
    entity_type: str,
    entity_id: str,
    operation: str,
    opts: RepositoryOptions,
    fn: Callable[[], Awaitable[T]],
) -> T:
    now = _now()
    tracer = get_tracer()
    span_name = f"repo.{entity_type}.{operation}"

    with tracer.start_as_current_span(
        span_name,
        attributes={
            "entity.type": entity_type,
            "entity.id": entity_id,
            "repo.operation": operation,
        },
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="repo.invocation",
                entity_type=entity_type,
                entity_id=entity_id,
                operation=operation,
                timestamp=now,
            ),
        )
        try:
            return await fn()
        except RepositoryFailure as failure:
            record_repo_failure(span, failure)
            opts.audit_log.write(
                AuditLogEntry(
                    level="error",
                    event=f"repo.{entity_type}.{operation}.failed",
                    entity_type=failure.entity_type,
                    entity_id=failure.entity_id,
                    operation=failure.operation,
                    timestamp=failure.timestamp,
                    detail={"code": failure.code, "message": failure.message},
                )
            )
            if opts.on_error is not None:
                opts.on_error(failure)
            raise
        except Exception as exc:
            failure = RepositoryFailure(
                code=f"REPO_{operation.upper()}_FAILED",
                message=str(exc),
                entity_type=entity_type,
                entity_id=entity_id,
                operation=operation,
                timestamp=now,
                cause=exc,
            )
            record_repo_failure(span, failure)
            opts.audit_log.write(
                AuditLogEntry(
                    level="error",
                    event=f"repo.{entity_type}.{operation}.failed",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    operation=operation,
                    timestamp=now,
                    detail={"code": failure.code, "message": failure.message},
                )
            )
            if opts.on_error is not None:
                opts.on_error(failure)
            raise failure from exc


# ---------------------------------------------------------------------------
# Traced repository wrappers
# ---------------------------------------------------------------------------


class _TracedAgentRepo:
    def __init__(self, repo: AgentRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, agent_id: str) -> Agent | None:
        return await _trace_op(
            "agent", agent_id, "get", self._opts, lambda: self._repo.get(agent_id)
        )

    async def save(self, agent: Agent) -> None:
        await _trace_op("agent", agent.id, "save", self._opts, lambda: self._repo.save(agent))

    async def delete(self, agent_id: str) -> None:
        await _trace_op(
            "agent", agent_id, "delete", self._opts, lambda: self._repo.delete(agent_id)
        )

    async def list(self, filter: AgentFilter) -> list[Agent]:
        return await _trace_op("agent", "*", "list", self._opts, lambda: self._repo.list(filter))


class _TracedSessionRepo:
    def __init__(self, repo: SessionRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, session_id: str) -> Session | None:
        return await _trace_op(
            "session", session_id, "get", self._opts, lambda: self._repo.get(session_id)
        )

    async def save(self, session: Session) -> None:
        await _trace_op("session", session.id, "save", self._opts, lambda: self._repo.save(session))

    async def delete(self, session_id: str) -> None:
        await _trace_op(
            "session", session_id, "delete", self._opts, lambda: self._repo.delete(session_id)
        )

    async def list(self, filter: SessionFilter) -> list[Session]:
        return await _trace_op("session", "*", "list", self._opts, lambda: self._repo.list(filter))


class _TracedThreadRepo:
    def __init__(self, repo: ThreadRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, thread_id: str) -> Thread | None:
        return await _trace_op(
            "thread", thread_id, "get", self._opts, lambda: self._repo.get(thread_id)
        )

    async def save(self, thread: Thread) -> None:
        await _trace_op("thread", thread.id, "save", self._opts, lambda: self._repo.save(thread))

    async def delete(self, thread_id: str) -> None:
        await _trace_op(
            "thread", thread_id, "delete", self._opts, lambda: self._repo.delete(thread_id)
        )

    async def list(self, filter: ThreadFilter) -> list[Thread]:
        return await _trace_op("thread", "*", "list", self._opts, lambda: self._repo.list(filter))


class _TracedMessageRepo:
    def __init__(self, repo: MessageRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, message_id: str) -> Message | None:
        return await _trace_op(
            "message", message_id, "get", self._opts, lambda: self._repo.get(message_id)
        )

    async def save(self, message: Message) -> None:
        await _trace_op("message", message.id, "save", self._opts, lambda: self._repo.save(message))

    async def delete(self, message_id: str) -> None:
        await _trace_op(
            "message", message_id, "delete", self._opts, lambda: self._repo.delete(message_id)
        )

    async def list(self, filter: MessageFilter) -> list[Message]:
        return await _trace_op("message", "*", "list", self._opts, lambda: self._repo.list(filter))


class _TracedToolCallRepo:
    def __init__(self, repo: ToolCallRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, tool_call_id: str) -> ToolCall | None:
        return await _trace_op(
            "tool_call", tool_call_id, "get", self._opts, lambda: self._repo.get(tool_call_id)
        )

    async def save(self, tool_call: ToolCall) -> None:
        await _trace_op(
            "tool_call", tool_call.id, "save", self._opts, lambda: self._repo.save(tool_call)
        )

    async def delete(self, tool_call_id: str) -> None:
        await _trace_op(
            "tool_call", tool_call_id, "delete", self._opts, lambda: self._repo.delete(tool_call_id)
        )

    async def list(self, filter: ToolCallFilter) -> list[ToolCall]:
        return await _trace_op(
            "tool_call", "*", "list", self._opts, lambda: self._repo.list(filter)
        )


class _TracedSkillRepo:
    def __init__(self, repo: SkillRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, skill_id: str) -> Skill | None:
        return await _trace_op(
            "skill", skill_id, "get", self._opts, lambda: self._repo.get(skill_id)
        )

    async def save(self, skill: Skill) -> None:
        await _trace_op("skill", skill.id, "save", self._opts, lambda: self._repo.save(skill))

    async def delete(self, skill_id: str) -> None:
        await _trace_op(
            "skill", skill_id, "delete", self._opts, lambda: self._repo.delete(skill_id)
        )

    async def list(self, filter: SkillFilter) -> list[Skill]:
        return await _trace_op("skill", "*", "list", self._opts, lambda: self._repo.list(filter))


class _TracedEnvironmentRepo:
    def __init__(self, repo: EnvironmentRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, environment_id: str) -> Environment | None:
        return await _trace_op(
            "environment", environment_id, "get", self._opts, lambda: self._repo.get(environment_id)
        )

    async def save(self, environment: Environment) -> None:
        await _trace_op(
            "environment", environment.id, "save", self._opts, lambda: self._repo.save(environment)
        )

    async def delete(self, environment_id: str) -> None:
        await _trace_op(
            "environment",
            environment_id,
            "delete",
            self._opts,
            lambda: self._repo.delete(environment_id),
        )

    async def list(self, filter: EnvironmentFilter) -> list[Environment]:
        return await _trace_op(
            "environment", "*", "list", self._opts, lambda: self._repo.list(filter)
        )


class _TracedMemoryRepo:
    def __init__(self, repo: MemoryRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, entry_id: str) -> MemoryEntry | None:
        return await _trace_op(
            "memory", entry_id, "get", self._opts, lambda: self._repo.get(entry_id)
        )

    async def save(self, entry: MemoryEntry) -> None:
        await _trace_op("memory", entry.id, "save", self._opts, lambda: self._repo.save(entry))

    async def delete(self, entry_id: str) -> None:
        await _trace_op(
            "memory", entry_id, "delete", self._opts, lambda: self._repo.delete(entry_id)
        )

    async def list(self, filter: MemoryFilter) -> list[MemoryEntry]:
        return await _trace_op("memory", "*", "list", self._opts, lambda: self._repo.list(filter))

    async def save_embedding(self, entry_id: str, embedding: bytes) -> None:
        await _trace_op(
            "memory",
            entry_id,
            "save_embedding",
            self._opts,
            lambda: self._repo.save_embedding(entry_id, embedding),
        )

    async def vec_search(self, filter: MemoryVecSearchFilter) -> list[MemoryVecSearchResult]:
        return await _trace_op(
            "memory",
            "*",
            "vec_search",
            self._opts,
            lambda: self._repo.vec_search(filter),
        )


class _TracedVaultRepo:
    def __init__(self, repo: VaultRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, entry_id: str) -> VaultEntry | None:
        return await _trace_op(
            "vault", entry_id, "get", self._opts, lambda: self._repo.get(entry_id)
        )

    async def save(self, entry: VaultEntry) -> None:
        await _trace_op("vault", entry.id, "save", self._opts, lambda: self._repo.save(entry))

    async def delete(self, entry_id: str) -> None:
        await _trace_op(
            "vault", entry_id, "delete", self._opts, lambda: self._repo.delete(entry_id)
        )

    async def list(self, filter: VaultFilter) -> list[VaultEntry]:
        return await _trace_op("vault", "*", "list", self._opts, lambda: self._repo.list(filter))


class _TracedUserProfileRepo:
    def __init__(self, repo: UserProfileRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, user_id: str) -> UserProfile | None:
        return await _trace_op(
            "user_profile", user_id, "get", self._opts, lambda: self._repo.get(user_id)
        )

    async def save(self, profile: UserProfile) -> None:
        await _trace_op(
            "user_profile", profile.id, "save", self._opts, lambda: self._repo.save(profile)
        )

    async def delete(self, user_id: str) -> None:
        await _trace_op(
            "user_profile", user_id, "delete", self._opts, lambda: self._repo.delete(user_id)
        )

    async def list(self, filter: UserProfileFilter) -> list[UserProfile]:
        return await _trace_op(
            "user_profile", "*", "list", self._opts, lambda: self._repo.list(filter)
        )


class _TracedChannelRepo:
    def __init__(self, repo: ChannelRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, channel_id: str) -> Channel | None:
        return await _trace_op(
            "channel", channel_id, "get", self._opts, lambda: self._repo.get(channel_id)
        )

    async def save(self, channel: Channel) -> None:
        await _trace_op("channel", channel.id, "save", self._opts, lambda: self._repo.save(channel))

    async def delete(self, channel_id: str) -> None:
        await _trace_op(
            "channel", channel_id, "delete", self._opts, lambda: self._repo.delete(channel_id)
        )

    async def list(self, filter: ChannelFilter) -> list[Channel]:
        return await _trace_op("channel", "*", "list", self._opts, lambda: self._repo.list(filter))


class _TracedWebhookRepo:
    def __init__(self, repo: WebhookRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def get(self, webhook_id: str) -> Webhook | None:
        return await _trace_op(
            "webhook", webhook_id, "get", self._opts, lambda: self._repo.get(webhook_id)
        )

    async def save(self, webhook: Webhook) -> None:
        await _trace_op("webhook", webhook.id, "save", self._opts, lambda: self._repo.save(webhook))

    async def delete(self, webhook_id: str) -> None:
        await _trace_op(
            "webhook", webhook_id, "delete", self._opts, lambda: self._repo.delete(webhook_id)
        )

    async def list(self, filter: WebhookFilter) -> list[Webhook]:
        return await _trace_op("webhook", "*", "list", self._opts, lambda: self._repo.list(filter))


class _TracedAuditLogEntryRepo:
    def __init__(self, repo: AuditLogEntryRepository, opts: RepositoryOptions) -> None:
        self._repo = repo
        self._opts = opts

    async def append(self, entry: AuditLogEntryRecord) -> None:
        await _trace_op(
            "audit_log_entry", entry.id, "append", self._opts, lambda: self._repo.append(entry)
        )

    async def list(self, filter: AuditLogEntryFilter) -> list[AuditLogEntryRecord]:
        return await _trace_op(
            "audit_log_entry", "*", "list", self._opts, lambda: self._repo.list(filter)
        )


# ---------------------------------------------------------------------------
# RepositoryRuntime — public entry point
# ---------------------------------------------------------------------------


class RepositoryRuntime:
    """
    Thin wrapper around a RepositoryDriver that adds OTel spans, structured
    invocation events, and audit-log writes on failure to every operation
    across all 12 resource repositories.

    Instantiate once with a concrete RepositoryDriver (e.g. SqliteRepositoryDriver),
    then access resources through the typed properties (runtime.agents, runtime.sessions, …).
    """

    def __init__(self, driver: RepositoryDriver, options: RepositoryOptions | None = None) -> None:
        self._driver = driver
        self._opts = options or RepositoryOptions()

    @cached_property
    def agents(self) -> _TracedAgentRepo:
        return _TracedAgentRepo(self._driver.agents, self._opts)

    @cached_property
    def sessions(self) -> _TracedSessionRepo:
        return _TracedSessionRepo(self._driver.sessions, self._opts)

    @cached_property
    def threads(self) -> _TracedThreadRepo:
        return _TracedThreadRepo(self._driver.threads, self._opts)

    @cached_property
    def messages(self) -> _TracedMessageRepo:
        return _TracedMessageRepo(self._driver.messages, self._opts)

    @cached_property
    def tool_calls(self) -> _TracedToolCallRepo:
        return _TracedToolCallRepo(self._driver.tool_calls, self._opts)

    @cached_property
    def skills(self) -> _TracedSkillRepo:
        return _TracedSkillRepo(self._driver.skills, self._opts)

    @cached_property
    def environments(self) -> _TracedEnvironmentRepo:
        return _TracedEnvironmentRepo(self._driver.environments, self._opts)

    @cached_property
    def memory(self) -> _TracedMemoryRepo:
        return _TracedMemoryRepo(self._driver.memory, self._opts)

    @cached_property
    def vault(self) -> _TracedVaultRepo:
        return _TracedVaultRepo(self._driver.vault, self._opts)

    @cached_property
    def user_profiles(self) -> _TracedUserProfileRepo:
        return _TracedUserProfileRepo(self._driver.user_profiles, self._opts)

    @cached_property
    def channels(self) -> _TracedChannelRepo:
        return _TracedChannelRepo(self._driver.channels, self._opts)

    @cached_property
    def webhooks(self) -> _TracedWebhookRepo:
        return _TracedWebhookRepo(self._driver.webhooks, self._opts)

    @cached_property
    def audit_log_entries(self) -> _TracedAuditLogEntryRepo:
        return _TracedAuditLogEntryRepo(self._driver.audit_log_entries, self._opts)

    async def migrate(self) -> None:
        """Apply all pending DDL migrations via the underlying driver."""
        await self._driver.migrate()

    async def close(self) -> None:
        """Release underlying database connections via the driver."""
        await self._driver.close()
