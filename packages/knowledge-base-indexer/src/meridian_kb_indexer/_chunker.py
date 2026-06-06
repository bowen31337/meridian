from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ._types import Chunk

# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".rst": "text",
}

_CODE_LANGS = {"python", "javascript", "typescript", "tsx"}

_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".ruff_cache",
}
_IGNORE_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dylib", ".exe", ".bin", ".lock"}


def should_index_path(path: str) -> bool:
    """Return True if this path should be indexed (not a build artifact or binary)."""
    p = Path(path)
    for part in p.parts:
        if part in _IGNORE_DIRS:
            return False
    if p.suffix in _IGNORE_EXTS:
        return False
    return True


def detect_language(file_path: str) -> str | None:
    ext = Path(file_path).suffix.lower()
    return _EXT_TO_LANG.get(ext)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chunk_file(file_path: str, content: str) -> list[Chunk]:
    """Parse *content* (read from *file_path*) into a list of Chunks.

    Dispatch:
    - Code files (.py, .js, .ts, .tsx): tree-sitter symbol extraction.
      Falls back to plain-text chunking when tree-sitter is unavailable.
    - Markdown (.md, .markdown): heading-boundary sections.
    - Everything else: paragraph-boundary text chunks.
    """
    language = detect_language(file_path)

    if language in _CODE_LANGS:
        chunks = _chunk_code(file_path, content, language)
        if chunks:
            return chunks
        # tree-sitter unavailable or no symbols found — fall through

    if language == "markdown":
        return _chunk_markdown(file_path, content)

    return _chunk_text(file_path, content)


# ---------------------------------------------------------------------------
# Tree-sitter code chunking
# ---------------------------------------------------------------------------

_PYTHON_QUERY = """
(decorated_definition) @symbol
(function_definition) @symbol
(class_definition) @symbol
"""

_JAVASCRIPT_QUERY = """
(function_declaration) @symbol
(class_declaration) @symbol
(generator_function_declaration) @symbol
(export_statement declaration: (function_declaration)) @symbol
(export_statement declaration: (class_declaration)) @symbol
"""

_TYPESCRIPT_QUERY = """
(function_declaration) @symbol
(class_declaration) @symbol
(interface_declaration) @symbol
(type_alias_declaration) @symbol
(export_statement declaration: (function_declaration)) @symbol
(export_statement declaration: (class_declaration)) @symbol
(export_statement declaration: (interface_declaration)) @symbol
(export_statement declaration: (type_alias_declaration)) @symbol
"""

_LANG_QUERIES: dict[str, str] = {
    "python": _PYTHON_QUERY,
    "javascript": _JAVASCRIPT_QUERY,
    "typescript": _TYPESCRIPT_QUERY,
    "tsx": _TYPESCRIPT_QUERY,
}


def _get_ts_parser_and_lang(language: str) -> tuple[Any, Any]:
    """Return (Parser, Language) for *language*. Raises ImportError if unavailable."""
    from tree_sitter import Language, Parser  # type: ignore[import-untyped]

    if language == "python":
        import tree_sitter_python as _m  # type: ignore[import-untyped]

        ts_lang = Language(_m.language())
    elif language in ("javascript", "jsx"):
        import tree_sitter_javascript as _m  # type: ignore[import-untyped]

        ts_lang = Language(_m.language())
    elif language == "typescript":
        import tree_sitter_typescript as _m  # type: ignore[import-untyped]

        ts_lang = Language(_m.language_typescript())
    elif language == "tsx":
        import tree_sitter_typescript as _m  # type: ignore[import-untyped]

        ts_lang = Language(_m.language_tsx())
    else:
        raise ValueError(f"Unsupported language: {language}")

    return Parser(ts_lang), ts_lang


def _symbol_name_and_kind(node: Any, language: str) -> tuple[str | None, str]:
    """Return (name, kind) for a tree-sitter definition node."""
    t = node.type

    if t == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _symbol_name_and_kind(child, language)
        return None, "decorated"

    if t in ("function_definition", "function_declaration", "generator_function_declaration"):
        n = node.child_by_field_name("name")
        return (n.text.decode() if n else None), "function"

    if t in ("class_definition", "class_declaration"):
        n = node.child_by_field_name("name")
        return (n.text.decode() if n else None), "class"

    if t == "interface_declaration":
        n = node.child_by_field_name("name")
        return (n.text.decode() if n else None), "interface"

    if t == "type_alias_declaration":
        n = node.child_by_field_name("name")
        return (n.text.decode() if n else None), "type"

    if t == "export_statement":
        decl = node.child_by_field_name("declaration")
        if decl:
            return _symbol_name_and_kind(decl, language)
        return None, "export"

    return None, t


def _chunk_code(file_path: str, content: str, language: str) -> list[Chunk]:
    try:
        parser, ts_lang = _get_ts_parser_and_lang(language)
    except (ImportError, Exception):
        return []

    query_pattern = _LANG_QUERIES.get(language, "")
    if not query_pattern:
        return []

    source = content.encode()
    tree = parser.parse(source)

    try:
        from tree_sitter import Query, QueryCursor  # type: ignore[import-untyped]

        query = Query(ts_lang, query_pattern)
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)
    except Exception:
        return []

    seen_ranges: set[tuple[int, int]] = set()
    chunks: list[Chunk] = []

    for _pattern_idx, captures_dict in matches:
        for node in captures_dict.get("symbol", []):
            key = (node.start_byte, node.end_byte)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)

            chunk_content = source[node.start_byte : node.end_byte].decode(errors="replace")
            symbol_name, symbol_kind = _symbol_name_and_kind(node, language)

            chunks.append(
                Chunk(
                    file_path=file_path,
                    kind="symbol",
                    content=chunk_content,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    symbol_name=symbol_name,
                    symbol_kind=symbol_kind,
                    language=language,
                )
            )

    return sorted(chunks, key=lambda c: c.start_line)


# ---------------------------------------------------------------------------
# Markdown chunking — split by ATX heading boundaries
# ---------------------------------------------------------------------------

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def _chunk_markdown(file_path: str, content: str) -> list[Chunk]:
    lines = content.splitlines(keepends=True)

    # Collect (line_index, level, heading_text) for every ATX heading
    headings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = _ATX_HEADING_RE.match(line.rstrip())
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    if not headings:
        return [
            Chunk(
                file_path=file_path,
                kind="text",
                content=content,
                start_line=1,
                end_line=max(len(lines), 1),
                language="markdown",
            )
        ]

    chunks: list[Chunk] = []
    for idx, (line_idx, level, heading_text) in enumerate(headings):
        end_idx = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        section = "".join(lines[line_idx:end_idx])
        chunks.append(
            Chunk(
                file_path=file_path,
                kind="heading",
                content=section,
                start_line=line_idx + 1,
                end_line=end_idx,
                heading_level=level,
                heading_text=heading_text,
                language="markdown",
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Plain-text chunking — split by blank-line paragraph boundaries
# ---------------------------------------------------------------------------


def _chunk_text(file_path: str, content: str) -> list[Chunk]:
    lines = content.splitlines(keepends=True)
    chunks: list[Chunk] = []
    para_start = 0
    in_para = False

    for i, line in enumerate(lines):
        if line.strip():
            if not in_para:
                para_start = i
                in_para = True
        else:
            if in_para:
                body = "".join(lines[para_start:i]).strip()
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        kind="text",
                        content=body,
                        start_line=para_start + 1,
                        end_line=i,
                    )
                )
                in_para = False

    if in_para:
        body = "".join(lines[para_start:]).strip()
        chunks.append(
            Chunk(
                file_path=file_path,
                kind="text",
                content=body,
                start_line=para_start + 1,
                end_line=len(lines),
            )
        )

    return chunks
