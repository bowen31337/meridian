#!/usr/bin/env python
"""Lint Rule L2: Provider SDK imports confined to their own subtree.

Each third-party provider SDK may only be imported inside its designated
packages/providers/<name>/ directory:

  anthropic  → packages/providers/anthropic/
  openai     → packages/providers/openai/
  openrouter → packages/providers/openrouter/
  ollama     → packages/providers/ollama/

Any import of these packages outside its designated directory is a
provider-isolation violation.

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

# Map each provider SDK top-level import name to its allowed directory
# (relative to repo root).
_PROVIDER_ALLOWED: dict[str, Path] = {
    "anthropic": Path("packages/providers/anthropic"),
    "openai": Path("packages/providers/openai"),
    "openrouter": Path("packages/providers/openrouter"),
    "ollama": Path("packages/providers/ollama"),
}

_BANNED_IMPORTS: frozenset[str] = frozenset(_PROVIDER_ALLOWED)

# Directory names that are always skipped during traversal.
_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", ".eggs", "node_modules"}
)


def _is_test_file(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name == "conftest.py"


def _in_allowed_dir(path: Path, import_name: str, repo_root: Path) -> bool:
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        return False
    allowed = _PROVIDER_ALLOWED[import_name]
    return rel.is_relative_to(allowed)


def _extract_provider_imports(path: Path) -> list[str]:
    """Return banned top-level provider SDK names imported in *path*, deduplicated."""
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
        for imp in _extract_provider_imports(py_file):
            if not _in_allowed_dir(py_file, imp, repo_root):
                rel = str(py_file.relative_to(repo_root))
                violations.append((rel, imp))

    if not violations:
        return 0

    print("", file=sys.stderr)
    print(
        "ERROR: Lint Rule L2 — provider SDK import(s) outside the designated subtree.",
        file=sys.stderr,
    )
    print("  Each provider SDK must only be imported inside its own directory:", file=sys.stderr)
    for sdk, allowed in _PROVIDER_ALLOWED.items():
        print(f"    {sdk!r:12s} → {allowed}/", file=sys.stderr)
    print("", file=sys.stderr)
    for rel, imp in violations:
        print(f"  {rel}: imports '{imp}'", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "  Move provider-specific calls behind the sdk-provider abstraction layer.",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    entry = {
        "level": "error",
        "event": "ci.lint_l2.provider_imports.failed",
        "timestamp": timestamp,
        "detail": {
            "message": "Provider SDK import(s) found outside the designated subtree",
            "violations": [{"file": rel, "import": imp} for rel, imp in violations],
        },
    }
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
