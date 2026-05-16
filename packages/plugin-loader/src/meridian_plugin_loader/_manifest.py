from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

PluginKind = Literal["tool", "provider", "environment", "channel"]
SandboxMode = Literal["in_daemon", "out_of_process"]


class PluginManifest(BaseModel):
    """Declarative description of an installed Meridian plugin.

    Attributes:
        name: Unique plugin identifier.
        kind: Plugin category (tool, provider, environment, or channel).
        sandbox_mode: Execution isolation model — "in_daemon" runs trusted
            code in the daemon process; "out_of_process" isolates the plugin
            as a subprocess, HTTP server, or MCP endpoint.
        capabilities: Capability strings this plugin requires at runtime.
        entry_point: Dotted import path (``module:attribute``) to the plugin
            implementation.
        metadata: Arbitrary key-value pairs for plugin-specific configuration.
    """

    name: str
    kind: PluginKind
    sandbox_mode: SandboxMode
    capabilities: list[str] = Field(default_factory=list)
    entry_point: str
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class PluginLoadError:
    """Describes a single plugin that failed to load."""

    plugin_name: str
    message: str
    code: str


@dataclass(frozen=True)
class PluginLoadResult:
    """Result of a plugin discovery pass."""

    manifests: list[PluginManifest] = field(default_factory=list)
    errors: list[PluginLoadError] = field(default_factory=list)
