"""Meridian SystemOAuthProvider — Claude Code CLI subprocess-backed OAuth provider.

Manages the full subprocess lifecycle (spawn, health-check, restart on hang,
kill on cancel) for a long-running ``claude --server`` process.  The pinned
CLI version is read from ``meridian.lock``.

Also exposes :func:`create_sdk_mcp_server` which builds the Sandbox-proxy MCP
bridge that forwards Claude Code inner-loop tool calls into Meridian's Sandbox.
"""

from ._mcp_server import SdkMcpServer, create_sdk_mcp_server
from ._version import CLAUDE_CODE_OAUTH_PROVIDER_VERSION
from .provider import SystemOAuthProvider

__version__ = CLAUDE_CODE_OAUTH_PROVIDER_VERSION

__all__ = [
    "SystemOAuthProvider",
    "SdkMcpServer",
    "create_sdk_mcp_server",
    "CLAUDE_CODE_OAUTH_PROVIDER_VERSION",
    "__version__",
]
