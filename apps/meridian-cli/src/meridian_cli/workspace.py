"""uv workspace initializer with OTel instrumentation and audit logging."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from opentelemetry.trace import Span, Status, StatusCode

from ._telemetry import get_tracer, record_failure, record_invocation_event

AUDIT_DIR = Path(".meridian")
AUDIT_LOG = AUDIT_DIR / "workspace-audit.ndjson"

_REQUIRED_MEMBERS = [
    "apps/meridiand",
    "apps/meridian-cli",
    "packages/core-errors",
    "packages/knowledge-base-indexer",
    "packages/sdk-capabilities",
    "packages/sdk-channel",
    "packages/sdk-environment",
    "packages/sdk-provider",
    "packages/sdk-sandbox",
    "packages/sdk-tool",
    "packages/storage-blob",
    "packages/storage-event-log",
    "packages/storage-reposit",
    "packages/storage-repository",
    "packages/system-ulid",
]


@dataclass
class WorkspaceError(Exception):
    code: str
    message: str
    detail: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _audit(level: str, event: str, detail: dict[str, object] | None = None) -> None:
    AUDIT_DIR.mkdir(exist_ok=True)
    entry: dict[str, object] = {"ts": _now(), "level": level, "event": event}
    if detail:
        entry["detail"] = detail
    with AUDIT_LOG.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


class UvWorkspaceInitializer:
    """Initializes the uv workspace and emits OTel spans + audit events per invocation."""

    def __init__(
        self,
        repo_root: Path | None = None,
        on_error: Callable[[WorkspaceError], None] | None = None,
    ) -> None:
        self._root = repo_root or Path.cwd()
        self._on_error = on_error

    def init(self) -> None:
        """Verify workspace members exist and run uv sync.

        Emits "workspace.init" OTel span. On failure writes to audit log,
        calls on_error, prints to stderr, and raises WorkspaceError.
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("workspace.init") as span:
            span.set_attribute("workspace.root", str(self._root))
            record_invocation_event(
                span,
                {
                    "event.name": "workspace.invocation",
                    "workspace.operation": "init",
                    "workspace.root": str(self._root),
                },
            )
            _audit("info", "workspace.init.invoked", {"root": str(self._root)})

            try:
                self._verify_members(span)
                self._run_uv_sync(span)
            except WorkspaceError as exc:
                record_failure(span, exc.code, exc.message)
                _audit("error", "workspace.init.failed", {"code": exc.code, "message": exc.message, **exc.detail})
                if self._on_error:
                    self._on_error(exc)
                print(f"error: {exc}", file=sys.stderr)
                raise

            _audit("info", "workspace.init.ok", {"root": str(self._root)})
            span.add_event("workspace.init.completed")

    def _verify_members(self, span: Span) -> None:
        missing = [m for m in _REQUIRED_MEMBERS if not (self._root / m / "pyproject.toml").exists()]
        if missing:
            raise WorkspaceError(
                code="WORKSPACE_MISSING_MEMBERS",
                message=f"workspace members not found: {', '.join(missing)}",
                detail={"missing": missing},
            )
        span.add_event("workspace.members.verified", {"member.count": len(_REQUIRED_MEMBERS)})

    def _run_uv_sync(self, span: Span) -> None:
        result = subprocess.run(
            ["uv", "sync", "--frozen"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise WorkspaceError(
                code="WORKSPACE_SYNC_FAILED",
                message=f"uv sync failed (exit {result.returncode}): {result.stderr.strip()}",
                detail={"exit_code": result.returncode},
            )
        span.add_event("workspace.sync.completed")
