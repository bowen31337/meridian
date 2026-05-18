from __future__ import annotations

from abc import ABC, abstractmethod

from ._types import (
    Agent,
    AgentFilter,
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


class AgentRepository(ABC):
    """
    Persistence contract for Agent records.

    Pass a concrete implementation to RepositoryRuntime, which wraps each call
    with an OTel span, a structured invocation event, and audit-log writes on failure.
    """

    @abstractmethod
    async def get(self, agent_id: str) -> Agent | None:
        """Return the Agent with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, agent: Agent) -> None:
        """Upsert the Agent record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, agent_id: str) -> None:
        """Remove the Agent with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: AgentFilter) -> list[Agent]:
        """Return Agents matching the filter, ordered by created_at descending."""


class SessionRepository(ABC):
    """Persistence contract for Session records."""

    @abstractmethod
    async def get(self, session_id: str) -> Session | None:
        """Return the Session with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, session: Session) -> None:
        """Upsert the Session record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Remove the Session with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: SessionFilter) -> list[Session]:
        """Return Sessions matching the filter, ordered by created_at descending."""


class ThreadRepository(ABC):
    """Persistence contract for Thread records."""

    @abstractmethod
    async def get(self, thread_id: str) -> Thread | None:
        """Return the Thread with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, thread: Thread) -> None:
        """Upsert the Thread record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, thread_id: str) -> None:
        """Remove the Thread with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: ThreadFilter) -> list[Thread]:
        """Return Threads matching the filter, ordered by created_at descending."""


class MessageRepository(ABC):
    """Persistence contract for Message records."""

    @abstractmethod
    async def get(self, message_id: str) -> Message | None:
        """Return the Message with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, message: Message) -> None:
        """Upsert the Message record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, message_id: str) -> None:
        """Remove the Message with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: MessageFilter) -> list[Message]:
        """Return Messages matching the filter, ordered by sequence ascending."""


class ToolCallRepository(ABC):
    """Persistence contract for ToolCall records."""

    @abstractmethod
    async def get(self, tool_call_id: str) -> ToolCall | None:
        """Return the ToolCall with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, tool_call: ToolCall) -> None:
        """Upsert the ToolCall record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, tool_call_id: str) -> None:
        """Remove the ToolCall with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: ToolCallFilter) -> list[ToolCall]:
        """Return ToolCalls matching the filter, ordered by created_at ascending."""


class SkillRepository(ABC):
    """Persistence contract for Skill records."""

    @abstractmethod
    async def get(self, skill_id: str) -> Skill | None:
        """Return the Skill with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, skill: Skill) -> None:
        """Upsert the Skill record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, skill_id: str) -> None:
        """Remove the Skill with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: SkillFilter) -> list[Skill]:
        """Return Skills matching the filter, ordered by name ascending."""


class EnvironmentRepository(ABC):
    """Persistence contract for Environment records."""

    @abstractmethod
    async def get(self, environment_id: str) -> Environment | None:
        """Return the Environment with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, environment: Environment) -> None:
        """Upsert the Environment record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, environment_id: str) -> None:
        """Remove the Environment with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: EnvironmentFilter) -> list[Environment]:
        """Return Environments matching the filter, ordered by created_at descending."""


class MemoryRepository(ABC):
    """Persistence contract for MemoryEntry records."""

    @abstractmethod
    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Return the MemoryEntry with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, entry: MemoryEntry) -> None:
        """Upsert the MemoryEntry record (insert or replace by id; (scope, key) must be unique)."""

    @abstractmethod
    async def delete(self, entry_id: str) -> None:
        """Remove the MemoryEntry with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: MemoryFilter) -> list[MemoryEntry]:
        """Return MemoryEntries matching the filter, ordered by key ascending."""

    @abstractmethod
    async def save_embedding(self, entry_id: str, embedding: bytes) -> None:
        """Store or replace the float32 embedding for an existing MemoryEntry."""

    @abstractmethod
    async def vec_search(self, filter: MemoryVecSearchFilter) -> list[MemoryVecSearchResult]:
        """Cosine-distance ANN query via vec_search() over the memory_entries_vec table."""


class VaultRepository(ABC):
    """Persistence contract for VaultEntry metadata records.

    Secret values are never stored; this repository tracks only name/description
    metadata so other subsystems can reference secrets via secret_ref:// URIs.
    """

    @abstractmethod
    async def get(self, entry_id: str) -> VaultEntry | None:
        """Return the VaultEntry with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, entry: VaultEntry) -> None:
        """Upsert the VaultEntry record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, entry_id: str) -> None:
        """Remove the VaultEntry with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: VaultFilter) -> list[VaultEntry]:
        """Return VaultEntries matching the filter, ordered by name ascending."""


class UserProfileRepository(ABC):
    """Persistence contract for UserProfile records."""

    @abstractmethod
    async def get(self, user_id: str) -> UserProfile | None:
        """Return the UserProfile with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, profile: UserProfile) -> None:
        """Upsert the UserProfile record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, user_id: str) -> None:
        """Remove the UserProfile with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: UserProfileFilter) -> list[UserProfile]:
        """Return UserProfiles matching the filter, ordered by username ascending."""


class ChannelRepository(ABC):
    """Persistence contract for Channel records."""

    @abstractmethod
    async def get(self, channel_id: str) -> Channel | None:
        """Return the Channel with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, channel: Channel) -> None:
        """Upsert the Channel record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, channel_id: str) -> None:
        """Remove the Channel with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: ChannelFilter) -> list[Channel]:
        """Return Channels matching the filter, ordered by name ascending."""


class WebhookRepository(ABC):
    """Persistence contract for Webhook records."""

    @abstractmethod
    async def get(self, webhook_id: str) -> Webhook | None:
        """Return the Webhook with the given id, or None if it does not exist."""

    @abstractmethod
    async def save(self, webhook: Webhook) -> None:
        """Upsert the Webhook record (insert or replace by id)."""

    @abstractmethod
    async def delete(self, webhook_id: str) -> None:
        """Remove the Webhook with the given id. No-op if it does not exist."""

    @abstractmethod
    async def list(self, filter: WebhookFilter) -> list[Webhook]:
        """Return Webhooks matching the filter, ordered by created_at descending."""
