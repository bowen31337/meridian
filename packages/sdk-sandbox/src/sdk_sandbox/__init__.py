# Types
# Audit log
from ._audit import AuditLog, NoopAuditLog

# Contract
from ._contract import ToolDispatcher

# Dispatchers
from ._dispatchers import (
    ContainerDispatcher,
    HttpDispatcher,
    InProcessDispatcher,
    McpDispatcher,
    SubprocessDispatcher,
)

# MCP client — tool discovery and stdio transport utilities
from ._mcp_client import (
    McpToolSpec,
    discover_mcp_tools,
    discover_mcp_tools_http,
    discover_mcp_tools_stdio,
)

# Runtime
from ._runtime import RuntimeOptions, Sandbox, default_sandbox

# Secret-ref substitution
from ._secret_refs import (
    SecretRefNotFoundError,
    SecretRefResolveError,
    SecretRefVaultNotFoundError,
    substitute_secret_refs,
)

# Telemetry
from ._telemetry import (
    get_tracer,
    record_capability_denial,
    record_env_mismatch,
    record_env_routing,
    record_input_schema_failure,
    record_invocation_event,
    record_output_schema_failure,
    record_sandbox_failure,
    record_tool_timeout,
)
from ._types import (
    AuditLogEntry,
    ContainerHandler,
    ExecutionContext,
    HttpHandler,
    InProcessHandler,
    McpHandler,
    SandboxFailure,
    SandboxResult,
    StructuredEvent,
    SubprocessHandler,
    ToolDefinition,
    ToolHandler,
)

# Version
from ._version import SANDBOX_SDK_VERSION
from .fake import FakeSandboxAdapter, write_sandbox_fixture

__all__ = [
    # Types
    "AuditLogEntry",
    "ContainerHandler",
    "ExecutionContext",
    "HttpHandler",
    "InProcessHandler",
    "McpHandler",
    "SandboxFailure",
    "SandboxResult",
    "StructuredEvent",
    "SubprocessHandler",
    "ToolDefinition",
    "ToolHandler",
    # Contract
    "ToolDispatcher",
    # Dispatchers
    "ContainerDispatcher",
    "HttpDispatcher",
    "InProcessDispatcher",
    "McpDispatcher",
    "SubprocessDispatcher",
    # Audit
    "AuditLog",
    "NoopAuditLog",
    # Secret-ref substitution
    "SecretRefNotFoundError",
    "SecretRefResolveError",
    "SecretRefVaultNotFoundError",
    "substitute_secret_refs",
    # MCP client
    "McpToolSpec",
    "discover_mcp_tools",
    "discover_mcp_tools_http",
    "discover_mcp_tools_stdio",
    # Telemetry
    "get_tracer",
    "record_capability_denial",
    "record_env_mismatch",
    "record_env_routing",
    "record_input_schema_failure",
    "record_invocation_event",
    "record_output_schema_failure",
    "record_sandbox_failure",
    "record_tool_timeout",
    # Runtime
    "RuntimeOptions",
    "Sandbox",
    "default_sandbox",
    # Fake / testing
    "FakeSandboxAdapter",
    "write_sandbox_fixture",
    # Version
    "SANDBOX_SDK_VERSION",
]
