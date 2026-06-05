#!/usr/bin/env python
"""OTel-instrumented uv workspace initializer.

Usage: uv run python scripts/init_uv_workspace.py [--check]
  --check   Verify members and lockfile only; do not run uv sync.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

AUDIT_DIR = Path(".meridian")
AUDIT_LOG = AUDIT_DIR / "workspace-audit.ndjson"

_TRACER_NAME = "meridian.workspace-init"
_tracer = trace.get_tracer(_TRACER_NAME)

_WORKSPACE_MEMBERS = [
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


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _audit(level: str, event: str, detail: dict | None = None) -> None:
    AUDIT_DIR.mkdir(exist_ok=True)
    entry: dict = {"ts": _now(), "level": level, "event": event}
    if detail:
        entry["detail"] = detail
    with AUDIT_LOG.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _fail(msg: str, detail: dict | None = None) -> int:
    print(f"error: {msg}", file=sys.stderr)
    _audit("error", "workspace.init.failed", {"message": msg, **(detail or {})})
    return 1


def _verify_members(root: Path) -> list[str]:
    return [m for m in _WORKSPACE_MEMBERS if not (root / m / "pyproject.toml").exists()]


def _check_lockfile(root: Path) -> bool:
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _run_sync(root: Path) -> tuple[int, str]:
    result = subprocess.run(
        ["uv", "sync", "--frozen"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stderr.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize the uv workspace at repo root.")
    parser.add_argument("--check", action="store_true", help="Verify only; do not sync")
    args = parser.parse_args()

    root = Path.cwd()
    _audit("info", "workspace.init.invoked", {"root": str(root), "check_only": args.check})

    with _tracer.start_as_current_span("workspace.init") as span:
        span.set_attribute("workspace.root", str(root))
        span.add_event(
            "workspace.invocation",
            {
                "workspace.operation": "init",
                "workspace.root": str(root),
                "check_only": args.check,
            },
        )

        # Step 1 — verify all workspace members are present.
        missing = _verify_members(root)
        if missing:
            msg = (
                f"workspace members not found: {', '.join(missing)}\n"
                "  Create the missing pyproject.toml files or add the directories."
            )
            span.set_status(Status(StatusCode.ERROR, msg))
            span.add_event("workspace.members.missing", {"missing": str(missing)})
            return _fail(msg, {"missing": missing})

        span.add_event("workspace.members.verified", {"member.count": len(_WORKSPACE_MEMBERS)})
        print(f"workspace: {len(_WORKSPACE_MEMBERS)} members verified")

        # Step 2 — verify lockfile is up-to-date.
        if not _check_lockfile(root):
            msg = "uv.lock is out of date — run 'uv lock' and commit the result"
            span.set_status(Status(StatusCode.ERROR, msg))
            span.add_event("workspace.lockfile.stale")
            return _fail(msg)

        span.add_event("workspace.lockfile.ok")
        print("workspace: uv.lock is up-to-date")

        if args.check:
            _audit("info", "workspace.init.ok", {"mode": "check"})
            span.add_event("workspace.init.completed", {"mode": "check"})
            print("workspace: check passed")
            return 0

        # Step 3 — sync.
        rc, stderr = _run_sync(root)
        if rc != 0:
            msg = f"uv sync failed (exit {rc}): {stderr}"
            span.set_status(Status(StatusCode.ERROR, msg))
            span.add_event("workspace.sync.failed", {"exit_code": rc})
            return _fail(msg, {"exit_code": rc})

        span.add_event("workspace.sync.completed")
        _audit("info", "workspace.init.ok", {"mode": "sync"})
        span.add_event("workspace.init.completed", {"mode": "sync"})
        print("workspace: initialized successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())
