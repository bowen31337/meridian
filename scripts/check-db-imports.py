#!/usr/bin/env python
"""Lint Rule L3: DB driver imports confined to the storage layer.

sqlite3, aiosqlite, psycopg, asyncpg, and sqlalchemy may only be imported
inside approved storage packages.  Any import of these modules outside the
approved directories is a repository-pattern violation.

Exit code 0 = clean.  Exit code 1 = violations found.
On failure writes a structured error to stderr and appends an NDJSON audit
entry to ${AUDIT_LOG_PATH:-ci-audit.ndjson}.
"""

from __future__ import annotations

import ast
import datetime
import json
import os
import sys
from pathlib import Path

# Imports that may only appear inside the approved storage directories.
_BANNED_IMPORTS: frozenset[str] = frozenset(
    {"sqlite3", "aiosqlite", "psycopg", "asyncpg", "sqlalchemy"}
)

# Directories (relative to repo root) where these imports are allowed.
# These are the repository-pattern storage packages and the KB storage layer.
_ALLOWED_DIRS: tuple[Path, ...] = (
    Path("packages/storage-repository"),
    Path("packages/storage-reposit"),
    Path("packages/storage-blob"),
    Path("packages/storage-event-log"),
    Path("packages/knowledge-base-indexer"),
)

# Directory names that are always skipped during traversal.
_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", ".eggs", "node_modules"}
)


def _is_test_file(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name == "conftest.py"


def _in_allowed_dir(path: Path, repo_root: Path) -> bool:
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        return False
    return any(rel.is_relative_to(allowed) for allowed in _ALLOWED_DIRS)


def _extract_db_imports(path: Path) -> list[str]:
    """Return banned top-level module names imported in *path*, deduplicated."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    seen: dict[str, None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BANNED_IMPORTS:
                    seen[top] = None
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _BANNED_IMPORTS:
                    seen[top] = None
    return list(seen)


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

    violations: list[tuple[str, str]] = []  # (rel_path, import_name)

    for py_file in _iter_python_files(repo_root):
        if _is_test_file(py_file):
            continue
        if _in_allowed_dir(py_file, repo_root):
            continue
        for imp in _extract_db_imports(py_file):
            rel = str(py_file.relative_to(repo_root))
            violations.append((rel, imp))

    if not violations:
        return 0

    print("", file=sys.stderr)
    print(
        "ERROR: Lint Rule L3 — DB driver import(s) outside the storage layer.",
        file=sys.stderr,
    )
    print(
        "  Only packages/storage-*/ and packages/knowledge-base-indexer/ may",
        file=sys.stderr,
    )
    print(
        "  import sqlite3, aiosqlite, psycopg, asyncpg, or sqlalchemy.",
        file=sys.stderr,
    )
    print("", file=sys.stderr)
    for rel, imp in violations:
        print(f"  {rel}: imports '{imp}'", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "  Move DB access behind a repository or storage-layer abstraction.",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    entry = {
        "level": "error",
        "event": "ci.lint_l3.db_imports.failed",
        "timestamp": timestamp,
        "detail": {
            "message": "DB driver import(s) found outside the storage layer",
            "violations": [{"file": rel, "import": imp} for rel, imp in violations],
        },
    }
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
