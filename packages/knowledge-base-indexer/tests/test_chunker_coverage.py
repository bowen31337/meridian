"""Branch-coverage completion for _chunker.py.

Exercises the remaining paths the happy-path suite in test_chunker.py misses:
the code -> text fall-through, every _get_ts_parser_and_lang language branch,
all _symbol_name_and_kind symbol kinds (including the defensive fall-throughs),
each graceful-degradation branch in _chunk_code, the seen_ranges dedup, and the
plain-text loop's skip branches.
"""

from __future__ import annotations

from typing import Any

import pytest

from meridian_kb_indexer import _chunker
from meridian_kb_indexer._chunker import (
    _chunk_code,
    _get_ts_parser_and_lang,
    _symbol_name_and_kind,
    chunk_file,
)


def _ts_available(*mods: str) -> bool:
    try:
        for m in mods:
            __import__(m)
        return True
    except ImportError:
        return False


_HAS_PY = _ts_available("tree_sitter", "tree_sitter_python")
_HAS_TS = _ts_available("tree_sitter", "tree_sitter_typescript")
_HAS_JS = _ts_available("tree_sitter", "tree_sitter_javascript")

_needs_py = pytest.mark.skipif(not _HAS_PY, reason="tree-sitter-python not installed")
_needs_ts = pytest.mark.skipif(not _HAS_TS, reason="tree-sitter-typescript not installed")


# ---------------------------------------------------------------------------
# chunk_file: code language that yields no symbols falls through to text (73->77)
# ---------------------------------------------------------------------------


@_needs_py
def test_python_file_without_symbols_falls_through_to_text() -> None:
    chunks = chunk_file("config.py", "x = 1\ny = 2\n")
    assert chunks
    assert all(c.kind == "text" for c in chunks)


# ---------------------------------------------------------------------------
# _get_ts_parser_and_lang: typescript / tsx / unsupported branches (132-141)
# ---------------------------------------------------------------------------


@_needs_ts
def test_get_parser_typescript() -> None:
    parser, lang = _get_ts_parser_and_lang("typescript")
    assert parser is not None and lang is not None


@_needs_ts
def test_get_parser_tsx() -> None:
    parser, lang = _get_ts_parser_and_lang("tsx")
    assert parser is not None and lang is not None


def test_get_parser_unsupported_language_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        _get_ts_parser_and_lang("ruby")


# ---------------------------------------------------------------------------
# _symbol_name_and_kind: TS symbol kinds via real parsing (164-175)
# ---------------------------------------------------------------------------


_TS_SRC = """\
export interface Widget {
    id: string;
}

export type Id = string;

export function build(): Widget {
    return { id: "x" };
}

interface Bare {
    n: number;
}

type Plain = number;
"""


@_needs_ts
def test_typescript_symbol_kinds() -> None:
    chunks = chunk_file("widget.ts", _TS_SRC)
    kinds = {c.symbol_kind for c in chunks}
    names = {c.symbol_name for c in chunks}
    assert "interface" in kinds
    assert "type" in kinds
    assert "function" in kinds
    assert {"Widget", "Id", "build", "Bare", "Plain"} <= names


_PY_DECORATED = """\
import functools


@functools.cache
def memoized() -> int:
    return 1
"""


@_needs_py
def test_python_decorated_definition_resolves_inner_name() -> None:
    chunks = chunk_file("dec.py", _PY_DECORATED)
    decorated = [c for c in chunks if c.symbol_name == "memoized" or c.symbol_kind == "function"]
    assert any(c.symbol_name == "memoized" for c in chunks)
    assert decorated


# ---------------------------------------------------------------------------
# _symbol_name_and_kind: defensive fall-through branches via fake nodes
# (154, 176, 178) — unreachable through the live queries but structurally
# required, so exercised directly.
# ---------------------------------------------------------------------------


class _FakeNode:
    def __init__(
        self,
        node_type: str,
        children: tuple[Any, ...] = (),
        fields: dict[str, Any] | None = None,
    ) -> None:
        self.type = node_type
        self.children = children
        self._fields = fields or {}

    def child_by_field_name(self, name: str) -> Any:
        return self._fields.get(name)


def test_symbol_decorated_without_inner_definition() -> None:
    node = _FakeNode("decorated_definition", children=(_FakeNode("decorator"),))
    assert _symbol_name_and_kind(node, "python") == (None, "decorated")


def test_symbol_export_without_declaration() -> None:
    node = _FakeNode("export_statement", fields={"declaration": None})
    assert _symbol_name_and_kind(node, "typescript") == (None, "export")


def test_symbol_unknown_node_type() -> None:
    node = _FakeNode("mystery_node")
    assert _symbol_name_and_kind(node, "python") == (None, "mystery_node")


# ---------------------------------------------------------------------------
# _chunk_code graceful-degradation branches
# ---------------------------------------------------------------------------


def test_chunk_code_parser_unavailable_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_language: str) -> tuple[Any, Any]:
        raise ImportError("tree-sitter missing")

    monkeypatch.setattr(_chunker, "_get_ts_parser_and_lang", _boom)
    assert _chunk_code("m.py", "def f(): pass\n", "python") == []


@_needs_py
def test_chunk_code_empty_query_pattern_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_chunker, "_LANG_QUERIES", {})
    assert _chunk_code("m.py", "def f(): pass\n", "python") == []


@_needs_py
def test_chunk_code_invalid_query_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_chunker, "_LANG_QUERIES", {"python": "((("})
    assert _chunk_code("m.py", "def f(): pass\n", "python") == []


@_needs_py
def test_chunk_code_dedupes_duplicate_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    dup_query = "(function_definition) @symbol\n(function_definition) @symbol\n"
    monkeypatch.setattr(_chunker, "_LANG_QUERIES", {"python": dup_query})
    chunks = _chunk_code("m.py", "def only(): return 1\n", "python")
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "only"


# ---------------------------------------------------------------------------
# _chunk_text loop skip branches (294->292, 298->292)
# ---------------------------------------------------------------------------


def test_text_leading_blanks_and_consecutive_lines() -> None:
    content = "\n\nfirst\nsecond\n\nthird\n"
    chunks = chunk_file("notes.txt", content)
    contents = [c.content for c in chunks]
    assert contents == ["first\nsecond", "third"]


def test_text_trailing_paragraph_without_blank_line() -> None:
    chunks = chunk_file("notes.txt", "alpha\nbeta")
    assert len(chunks) == 1
    assert chunks[0].content == "alpha\nbeta"
