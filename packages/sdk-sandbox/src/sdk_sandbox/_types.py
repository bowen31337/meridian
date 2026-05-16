from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ExecutionContext:
    """Ambient context forwarded to every tool invocation."""

    session_id: str
    workspace: str = ""
    scratch_dir: str | None = None
    granted_capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class InProcessHandler:
    """Tool implemented directly in the host process."""

    module: str = ""
    kind: Literal["in_process"] = field(default="in_process", init=False)


@dataclass(frozen=True)
class SubprocessHandler:
    """Tool implemented as a subprocess (JSON stdin/stdout protocol)."""

    path: str = ""
    kind: Literal["subprocess"] = field(default="subprocess", init=False)


@dataclass(frozen=True)
class McpHandler:
    """Tool hosted on an MCP server."""

    server_url: str = ""
    tool_name: str = ""
    kind: Literal["mcp"] = field(default="mcp", init=False)


@dataclass(frozen=True)
class HttpHandler:
    """Tool exposed over HTTP."""

    url: str = ""
    kind: Literal["http"] = field(default="http", init=False)


@dataclass(frozen=True)
class ContainerHandler:
    """Tool running inside a container environment."""

    environment_id: str = ""
    entrypoint: str = ""
    kind: Literal["container"] = field(default="container", init=False)


ToolHandler = InProcessHandler | SubprocessHandler | McpHandler | HttpHandler | ContainerHandler


@dataclass(frozen=True)
class ToolDefinition:
    """Full specification of a tool registered with the Sandbox."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Any  # ToolHandler union — kept as Any to avoid runtime union issues
    output_schema: dict[str, Any] | None = None
    required_capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class SandboxResult:
    """Successful outcome of a sandbox execute operation."""

    content: Any
    duration_ms: float = 0.0

    def to_mcp_content_blocks(self) -> list[dict[str, Any]]:
        """Convert result content to MCP content blocks."""
        if isinstance(self.content, str):
            return [{"type": "text", "text": self.content}]
        return [{"type": "text", "text": str(self.content)}]


class SandboxFailure(Exception):
    """
    Structured failure raised by the Sandbox on any execute error.
    Written to the audit log and recorded on the OTel span before being raised.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        tool_name: str,
        session_id: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.tool_name = tool_name
        self.session_id = session_id
        self.timestamp = timestamp
        self.cause = cause


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every sandbox failure."""

    level: Literal["info", "warn", "error"]
    event: str
    tool_name: str
    session_id: str
    timestamp: str
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class StructuredEvent:
    """Structured event attached to every OTel span, one per execute invocation."""

    name: str
    tool_name: str
    session_id: str
    timestamp: str
    operation: str
