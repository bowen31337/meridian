from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

# Capability string: dotted name with optional bracket parameter.
# Examples: "fs.read[/workspace/**]", "net.fetch[api.example.com]", "exec.shell"
Capability = str


class ToolContext(BaseModel):
    """Runtime context passed into every tool invocation."""

    workspace: str
    session_id: str
    thread_id: str | None = None
    scratch_dir: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ToolError(BaseModel):
    """Structured error surfaced to the model as is_error=true."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Return value from any tool execution."""

    result: Any = None
    is_error: bool = False
    error: ToolError | None = None

    @classmethod
    def ok(cls, result: Any) -> ToolResult:
        return cls(result=result, is_error=False)

    @classmethod
    def err(cls, code: str, message: str, **details: Any) -> ToolResult:
        return cls(
            result=None,
            is_error=True,
            error=ToolError(code=code, message=message, details=details),
        )


# ---------------------------------------------------------------------------
# Handler kinds (Architecture §11.1)
# ---------------------------------------------------------------------------


class InProcessHandler(BaseModel):
    """The callable is a Python function in the same process."""

    kind: Literal["in_process"] = "in_process"
    module: str = ""  # populated by decorator; informational only


class SubprocessHandler(BaseModel):
    """Tool runs as a child process; args/result flow over stdin/stdout JSON."""

    kind: Literal["subprocess"] = "subprocess"
    path: str


class McpHandler(BaseModel):
    """Tool is served by an MCP server; the Sandbox proxy routes the call."""

    kind: Literal["mcp"] = "mcp"
    server_url: str
    tool_name: str


class HttpHandler(BaseModel):
    """Tool is served by an HTTP endpoint that accepts POST with JSON body."""

    kind: Literal["http"] = "http"
    url: str
    auth: dict[str, Any] | None = None


class ContainerHandler(BaseModel):
    """Tool runs inside a container managed by the Environment Manager."""

    kind: Literal["container"] = "container"
    environment_id: str
    entrypoint: str


ToolHandler = Annotated[
    Union[
        InProcessHandler,
        SubprocessHandler,
        McpHandler,
        HttpHandler,
        ContainerHandler,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Tool definition (Architecture §11 / PRD F-SB-2)
# ---------------------------------------------------------------------------


class ToolDefinition(BaseModel):
    """Complete declarative description of a Meridian tool.

    Carries everything the harness / Sandbox needs to dispatch, validate,
    and enforce capability intersection for a single tool.
    """

    name: str
    description: str
    input_schema: dict[str, Any]   # JSON Schema — validated pre-dispatch
    output_schema: dict[str, Any] | None = None  # JSON Schema — validated post-dispatch
    capabilities: list[Capability] = Field(default_factory=list)
    required_environment: str | None = None   # e.g. "docker"; None = any
    timeout_ms: int = 30_000
    memory_cap_mb: int | None = None
    handler: ToolHandler
