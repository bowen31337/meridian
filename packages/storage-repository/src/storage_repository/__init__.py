# Types — domain models
# Audit log
from ._audit import AuditLog, NoopAuditLog

# Contracts — abstract repository interfaces
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

# Migrations
from ._migrations import SCHEMA_VERSION, load_migration_files

# Runtime
from ._runtime import RepositoryDriver, RepositoryOptions, RepositoryRuntime

# SQLite backend (requires aiosqlite)
from ._sqlite import SqliteRepositoryDriver

# Telemetry
from ._telemetry import get_tracer, record_invocation_event, record_repo_failure
from ._types import (
    Agent,
    AgentFilter,
    AuditLogEntry,
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

# Version
from ._version import REPOSITORY_SDK_VERSION

__all__ = [
    # Types
    "Agent",
    "AgentFilter",
    "AuditLogEntry",
    "Channel",
    "ChannelFilter",
    "Environment",
    "EnvironmentFilter",
    "MemoryEntry",
    "MemoryFilter",
    "MemoryVecSearchFilter",
    "MemoryVecSearchResult",
    "Message",
    "MessageFilter",
    "RepositoryFailure",
    "Session",
    "SessionFilter",
    "Skill",
    "SkillFilter",
    "StructuredEvent",
    "Thread",
    "ThreadFilter",
    "ToolCall",
    "ToolCallFilter",
    "UserProfile",
    "UserProfileFilter",
    "VaultEntry",
    "VaultFilter",
    "Webhook",
    "WebhookFilter",
    # Contracts
    "AgentRepository",
    "ChannelRepository",
    "EnvironmentRepository",
    "MemoryRepository",
    "MessageRepository",
    "SessionRepository",
    "SkillRepository",
    "ThreadRepository",
    "ToolCallRepository",
    "UserProfileRepository",
    "VaultRepository",
    "WebhookRepository",
    # Audit
    "AuditLog",
    "NoopAuditLog",
    # Telemetry
    "get_tracer",
    "record_invocation_event",
    "record_repo_failure",
    # Migrations
    "SCHEMA_VERSION",
    "load_migration_files",
    # Runtime
    "RepositoryDriver",
    "RepositoryOptions",
    "RepositoryRuntime",
    # SQLite backend
    "SqliteRepositoryDriver",
    # Version
    "REPOSITORY_SDK_VERSION",
]
