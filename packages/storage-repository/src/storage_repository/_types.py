from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


class RepositoryFailure(Exception):
    """
    Structured failure raised by RepositoryRuntime on any repository operation error.
    Recorded on the OTel span and written to the audit log before being raised.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        entity_type: str,
        entity_id: str,
        operation: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.operation = operation
        self.timestamp = timestamp
        self.cause = cause


# ---------------------------------------------------------------------------
# Audit / telemetry helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every repository failure."""

    level: Literal["info", "warn", "error"]
    event: str
    entity_type: str
    entity_id: str
    operation: str
    timestamp: str
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class AuditLogEntryRecord:
    """Stored audit log entry retrieved from the audit_log_entries table."""

    id: str
    level: Literal["info", "warn", "error"]
    event: str
    entity_type: str
    entity_id: str
    operation: str
    timestamp: str
    detail: dict[str, Any] | None = None
    signature: str | None = None


@dataclass(frozen=True)
class AuditLogEntryFilter:
    level: str | None = None
    event: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    since: str | None = None  # ISO 8601 lower bound (inclusive)
    until: str | None = None  # ISO 8601 upper bound (inclusive)
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class StructuredEvent:
    """Structured event attached to every OTel span, one per operation invocation."""

    name: str
    entity_type: str
    entity_id: str
    operation: str
    timestamp: str


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Agent:
    id: str
    kind: str
    name: str
    config: str  # JSON object
    capabilities: str  # JSON array of capability strings
    created_at: str  # ISO 8601
    updated_at: str  # ISO 8601


@dataclass(frozen=True)
class Session:
    id: str
    agent_id: str
    status: str  # "active" | "closed" | "error"
    metadata: str | None  # JSON object or None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Thread:
    id: str
    session_id: str
    title: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    id: str
    thread_id: str
    session_id: str
    role: str  # "user" | "assistant" | "tool"
    content: str  # JSON array of content blocks
    sequence: int
    created_at: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    message_id: str
    session_id: str
    tool_name: str
    input: str  # JSON object
    output: str | None  # JSON object or None
    status: str  # "pending" | "success" | "error"
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str
    capabilities: str  # JSON array of capability strings
    config: str  # JSON object
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Environment:
    id: str
    kind: str
    status: str  # "provisioned" | "running" | "reclaimed" | "error"
    config: str  # JSON object
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    scope: str  # e.g., agent_id or session_id
    key: str
    value: str  # JSON or plain text
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class VaultEntry:
    """Metadata record for a named secret. Secret values are never stored here."""

    id: str
    name: str  # e.g., "openai/api_key"
    description: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class UserProfile:
    id: str
    username: str
    display_name: str | None
    email: str | None
    metadata: str | None  # JSON object or None
    is_primary: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Channel:
    id: str
    kind: str  # e.g., "meridian.slack"
    name: str
    config: str  # JSON object (no secrets)
    status: str  # "active" | "inactive" | "error"
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Webhook:
    id: str
    url: str
    events: str  # JSON array of event type strings
    secret_ref: str | None  # secret_ref://vault/name or None
    status: str  # "active" | "inactive"
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentFilter:
    kind: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class SessionFilter:
    agent_id: str | None = None
    status: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class ThreadFilter:
    session_id: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class MessageFilter:
    thread_id: str | None = None
    session_id: str | None = None
    role: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class ToolCallFilter:
    message_id: str | None = None
    session_id: str | None = None
    status: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class SkillFilter:
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class EnvironmentFilter:
    kind: str | None = None
    status: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class MemoryFilter:
    scope: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class VaultFilter:
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class UserProfileFilter:
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class ChannelFilter:
    kind: str | None = None
    status: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class WebhookFilter:
    status: str | None = None
    limit: int = 100
    offset: int = 0


# ---------------------------------------------------------------------------
# Vector search types (sqlite-vec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryVecSearchFilter:
    """ANN query parameters for cosine-distance search over memory_entries_vec."""

    embedding: bytes  # sqlite_vec.serialize_float32(vector)
    scope: str | None = None
    limit: int = 10


@dataclass(frozen=True)
class MemoryVecSearchResult:
    """A single result from a vec_search() ANN query."""

    entry: MemoryEntry
    distance: float
