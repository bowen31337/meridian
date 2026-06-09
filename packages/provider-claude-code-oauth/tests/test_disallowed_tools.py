"""
Tests for Contract 1 — built-in tools stay disabled (Architecture §13.4).

In headless one-shot mode the CLI's own Read/Write/Bash/Edit are passed to
``--disallowed-tools`` on every call, so the inner loop can only reach the
Meridian tools bridged via MCP.

Coverage:
  ALL_CLAUDE_CODE_BUILTIN_TOOLS constant: contains the built-ins, excludes the
    MCP proxy tool, is a frozenset.
  _build_args: --disallowed-tools always present with every built-in, both
    without and with the MCP tool bridge.
  DisallowedToolError: subclass of CliSubprocessError and ProviderCallError.
  Package public API: both symbols exported from the package __init__.
"""

from __future__ import annotations

from typing import Any

from meridian_sdk_provider.errors import ProviderCallError
from meridian_sdk_provider.types import Message, ModelCallOpts

from meridian_provider_claude_code_oauth import (
    ALL_CLAUDE_CODE_BUILTIN_TOOLS as PKG_BUILTINS,
)
from meridian_provider_claude_code_oauth import (
    DisallowedToolError as PKG_DISALLOWED,
)
from meridian_provider_claude_code_oauth._disallowed_tools import ALL_CLAUDE_CODE_BUILTIN_TOOLS
from meridian_provider_claude_code_oauth._subprocess import (
    CliSubprocessError,
    CliSubprocessManager,
    DisallowedToolError,
)


def _opts(metadata: dict[str, Any] | None = None) -> ModelCallOpts:
    return ModelCallOpts(
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="hi")],
        metadata=metadata or {},
    )


class TestBuiltinConstant:
    def test_contains_builtins(self) -> None:
        assert {"Read", "Write", "Bash", "Edit"} <= set(ALL_CLAUDE_CODE_BUILTIN_TOOLS)

    def test_excludes_mcp_proxy(self) -> None:
        assert "meridian_tool_proxy" not in ALL_CLAUDE_CODE_BUILTIN_TOOLS

    def test_is_frozenset(self) -> None:
        assert isinstance(ALL_CLAUDE_CODE_BUILTIN_TOOLS, frozenset)


class TestDisallowedInArgs:
    def test_disallowed_present_without_bridge(self) -> None:
        args, _ = CliSubprocessManager("claude", "v")._build_args(_opts())
        assert "--disallowed-tools" in args
        assert set(ALL_CLAUDE_CODE_BUILTIN_TOOLS) <= set(args)

    def test_disallowed_present_with_bridge(self) -> None:
        meta = {
            "meridian_tools": {
                "agent_id": "a",
                "storage_root": "/r",
                "tools": ["exec"],
                "workspace": "/ws",
            }
        }
        args, _ = CliSubprocessManager("claude", "v")._build_args(_opts(meta))
        # Built-ins disabled even while MCP tools are allowed.
        assert "--disallowed-tools" in args
        assert set(ALL_CLAUDE_CODE_BUILTIN_TOOLS) <= set(args)
        assert "mcp__meridian__exec" in args


class TestDisallowedToolError:
    def test_subclass_hierarchy(self) -> None:
        assert issubclass(DisallowedToolError, CliSubprocessError)
        assert issubclass(DisallowedToolError, ProviderCallError)


class TestPackageExports:
    def test_constant_exported(self) -> None:
        assert PKG_BUILTINS is ALL_CLAUDE_CODE_BUILTIN_TOOLS

    def test_error_exported(self) -> None:
        assert PKG_DISALLOWED is DisallowedToolError
