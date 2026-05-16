"""Tests for _chunker: markdown, plain-text, code (if tree-sitter available), and path filter."""

from __future__ import annotations

import pytest

from meridian_kb_indexer._chunker import chunk_file, detect_language, should_index_path

# ---------------------------------------------------------------------------
# Path filter
# ---------------------------------------------------------------------------


def test_should_index_python_file() -> None:
    assert should_index_path("/workspace/src/foo.py") is True


def test_should_index_markdown() -> None:
    assert should_index_path("/workspace/README.md") is True


def test_ignores_git_dir() -> None:
    assert should_index_path("/workspace/.git/COMMIT_EDITMSG") is False


def test_ignores_pycache() -> None:
    assert should_index_path("/workspace/src/__pycache__/foo.cpython-312.pyc") is False


def test_ignores_node_modules() -> None:
    assert should_index_path("/workspace/frontend/node_modules/lodash/lodash.js") is False


def test_ignores_pyc_extension() -> None:
    assert should_index_path("/workspace/src/foo.pyc") is False


def test_ignores_lock_files() -> None:
    assert should_index_path("/workspace/uv.lock") is False


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def test_detect_language_python() -> None:
    assert detect_language("foo.py") == "python"


def test_detect_language_ts() -> None:
    assert detect_language("bar.ts") == "typescript"


def test_detect_language_markdown() -> None:
    assert detect_language("README.md") == "markdown"


def test_detect_language_unknown() -> None:
    assert detect_language("binary.bin") is None


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------


_MD_BASIC = """\
# Title

Some intro text.

## Section One

Content of section one.

## Section Two

Content of section two.
"""

_MD_NO_HEADINGS = """\
Just a plain paragraph.

Another paragraph.
"""


def test_markdown_three_sections() -> None:
    chunks = chunk_file("doc.md", _MD_BASIC)
    assert len(chunks) == 3
    assert chunks[0].kind == "heading"
    assert chunks[0].heading_level == 1
    assert chunks[0].heading_text == "Title"
    assert chunks[1].heading_text == "Section One"
    assert chunks[2].heading_text == "Section Two"


def test_markdown_no_headings_fallback_to_text() -> None:
    chunks = chunk_file("doc.md", _MD_NO_HEADINGS)
    assert len(chunks) == 1
    assert chunks[0].kind == "text"


def test_markdown_start_lines_are_correct() -> None:
    chunks = chunk_file("doc.md", _MD_BASIC)
    assert chunks[0].start_line == 1
    assert chunks[1].start_line == 5
    assert chunks[2].start_line == 9


def test_markdown_heading_levels() -> None:
    content = "# H1\n## H2\n### H3\n"
    chunks = chunk_file("doc.md", content)
    assert [c.heading_level for c in chunks] == [1, 2, 3]


def test_markdown_language_tag() -> None:
    chunks = chunk_file("doc.md", _MD_BASIC)
    assert all(c.language == "markdown" for c in chunks)


# ---------------------------------------------------------------------------
# Plain-text chunking
# ---------------------------------------------------------------------------


_TXT_TWO_PARAS = "First paragraph.\n\nSecond paragraph.\n"


def test_text_two_paragraphs() -> None:
    chunks = chunk_file("notes.txt", _TXT_TWO_PARAS)
    assert len(chunks) == 2
    assert chunks[0].content == "First paragraph."
    assert chunks[1].content == "Second paragraph."
    assert chunks[0].kind == "text"


def test_text_single_paragraph_no_blank_lines() -> None:
    chunks = chunk_file("notes.txt", "Hello world.")
    assert len(chunks) == 1
    assert chunks[0].kind == "text"


def test_text_empty_content() -> None:
    chunks = chunk_file("notes.txt", "")
    # No paragraphs → empty list
    assert chunks == []


# ---------------------------------------------------------------------------
# Code chunking (requires tree-sitter; skipped when unavailable)
# ---------------------------------------------------------------------------

_PYTHON_SRC = """\
def hello(name: str) -> str:
    return f"Hello, {name}"


class Greeter:
    def greet(self, name: str) -> str:
        return hello(name)
"""

_JS_SRC = """\
function add(a, b) {
    return a + b;
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}
"""


def _tree_sitter_available() -> bool:
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_python  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _tree_sitter_available(), reason="tree-sitter not installed")
def test_python_code_extracts_function_and_class() -> None:
    chunks = chunk_file("module.py", _PYTHON_SRC)
    kinds = {c.symbol_kind for c in chunks}
    names = {c.symbol_name for c in chunks}
    assert "function" in kinds
    assert "class" in kinds
    assert "hello" in names
    assert "Greeter" in names


@pytest.mark.skipif(not _tree_sitter_available(), reason="tree-sitter not installed")
def test_python_chunks_have_correct_kind() -> None:
    chunks = chunk_file("module.py", _PYTHON_SRC)
    assert all(c.kind == "symbol" for c in chunks)


@pytest.mark.skipif(not _tree_sitter_available(), reason="tree-sitter not installed")
def test_python_chunks_ordered_by_line() -> None:
    chunks = chunk_file("module.py", _PYTHON_SRC)
    lines = [c.start_line for c in chunks]
    assert lines == sorted(lines)


@pytest.mark.skipif(not _tree_sitter_available(), reason="tree-sitter not installed")
def test_python_chunk_content_matches_source() -> None:
    chunks = chunk_file("module.py", _PYTHON_SRC)
    func_chunk = next(c for c in chunks if c.symbol_name == "hello")
    assert "def hello" in func_chunk.content


def _js_tree_sitter_available() -> bool:
    try:
        import tree_sitter_javascript  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _js_tree_sitter_available(), reason="tree-sitter-javascript not installed")
def test_javascript_extracts_function_and_class() -> None:
    chunks = chunk_file("app.js", _JS_SRC)
    kinds = {c.symbol_kind for c in chunks}
    names = {c.symbol_name for c in chunks}
    assert "function" in kinds
    assert "class" in kinds
    assert "add" in names
    assert "Calculator" in names


def test_unknown_extension_uses_text_chunker() -> None:
    chunks = chunk_file("file.xyz", "Line one.\n\nLine two.\n")
    assert all(c.kind == "text" for c in chunks)
