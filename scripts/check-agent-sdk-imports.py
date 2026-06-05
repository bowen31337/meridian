#!/usr/bin/env python
"""Lint Rule L1: claude_agent_sdk imports confined to the OAuth provider subtree.

The claude_agent_sdk package may only be imported from inside
apps/meridiand/src/meridiand/providers/anthropic_oauth/.

Any import of claude_agent_sdk outside that directory is a violation of the
provider-isolation boundary (ARCHITECTURE.md §13.4 and §13.5).

Exit code 0 = clean.  Exit code 1 = violations found.
On failure writes a structured error to stderr and appends an NDJSON audit
entry to ${AUDIT_LOG_PATH:-ci-audit.ndjson}.
"""

from __future__ import annotations

import ast
import datetime
import json
import os
from pathlib import Path
import sys

# The only directory (relative to repo root) from which claude_agent_sdk may
# be imported.
_ALLOWED_DIR = Path("apps/meridiand/src/meridiand/providers/anthropic_oauth")

_BANNED_IMPORT = "claude_agent_sdk"

# Directory names that are always skipped during traversal.
_SKIP_DIRS: frozenset[str] = frozenset({".git", "__pycache__", ".venv", ".eggs", "node_modules"})


def _is_test_file(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name == "conftest.py"


def _in_allowed_dir(path: Path, repo_root: Path) -> bool:
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        return False
    return rel.is_relative_to(_ALLOWED_DIR)


def _imports_agent_sdk(path: Path) -> bool:
    """Return True if *path* imports claude_agent_sdk at any level."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == _BANNED_IMPORT:
                    return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == _BANNED_IMPORT
        ):
            return True
    return False


def _should_skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or name.endswith(".egg-info")


def _iter_python_files(repo_root: Path):
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for filename in filenames:
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


def main() -> int:
    repo_root = Path(__file__).parent.parent.resolve()
    audit_log = Path(os.environ.get("AUDIT_LOG_PATH", "ci-audit.ndjson"))
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()

    violations: list[str] = []  # relative paths of offending files

    for py_file in _iter_python_files(repo_root):
        if _is_test_file(py_file):
            continue
        if _in_allowed_dir(py_file, repo_root):
            continue
        if _imports_agent_sdk(py_file):
            violations.append(str(py_file.relative_to(repo_root)))

    if not violations:
        return 0

    print("", file=sys.stderr)
    print(
        "ERROR: Lint Rule L1 — claude_agent_sdk import(s) outside the designated subtree.",
        file=sys.stderr,
    )
    print(
        "  claude_agent_sdk may only be imported inside:",
        file=sys.stderr,
    )
    print(f"    {_ALLOWED_DIR}/", file=sys.stderr)
    print("", file=sys.stderr)
    for rel in violations:
        print(f"  {rel}: imports 'claude_agent_sdk'", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "  See docs/ARCHITECTURE.md §13.4 and §13.5 for the provider-isolation rationale.",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    entry = {
        "level": "error",
        "event": "ci.lint_l1.agent_sdk_imports.failed",
        "timestamp": timestamp,
        "detail": {
            "message": "claude_agent_sdk import(s) found outside the designated subtree",
            "violations": [{"file": rel, "import": _BANNED_IMPORT} for rel in violations],
        },
    }
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
