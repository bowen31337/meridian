"""Tests for WorkspaceIndexer: scan, OTel spans, audit log, and error handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from meridian_kb_indexer import IndexEvent, WorkspaceIndexer
from meridian_kb_indexer._types import IndexerError

# ---------------------------------------------------------------------------
# OTel setup fixture — provider set once per session; exporter cleared per test
# ---------------------------------------------------------------------------

_SESSION_EXPORTER: InMemorySpanExporter | None = None


@pytest.fixture(scope="session", autouse=True)
def _otel_session_provider():
    global _SESSION_EXPORTER
    _SESSION_EXPORTER = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_SESSION_EXPORTER))
    trace.set_tracer_provider(provider)
    yield


@pytest.fixture()
def otel_exporter(_otel_session_provider):
    assert _SESSION_EXPORTER is not None
    _SESSION_EXPORTER.clear()
    yield _SESSION_EXPORTER
    _SESSION_EXPORTER.clear()


# ---------------------------------------------------------------------------
# index_file
# ---------------------------------------------------------------------------


async def test_index_file_returns_chunks(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nBody text.\n## Section\n\nMore text.\n")
    indexer = WorkspaceIndexer(workspace=str(tmp_path))
    chunks = await indexer.index_file(str(f))
    assert len(chunks) == 2
    assert chunks[0].heading_text == "Title"


async def test_index_file_plain_text(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("Hello world.\n\nSecond paragraph.\n")
    indexer = WorkspaceIndexer(workspace=str(tmp_path))
    chunks = await indexer.index_file(str(f))
    assert len(chunks) == 2
    assert all(c.kind == "text" for c in chunks)


async def test_index_file_emits_otel_span(tmp_path: Path, otel_exporter) -> None:
    f = tmp_path / "sample.md"
    f.write_text("# H\n\ntext\n")
    indexer = WorkspaceIndexer(workspace=str(tmp_path))
    await indexer.index_file(str(f))

    spans = otel_exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "indexer.index_file" in span_names


async def test_index_file_raises_on_missing_file(tmp_path: Path) -> None:
    indexer = WorkspaceIndexer(workspace=str(tmp_path))
    with pytest.raises(IndexerError) as exc_info:
        await indexer.index_file(str(tmp_path / "missing.md"))
    assert exc_info.value.code == "INDEXER_FILE_FAILED"


async def test_index_file_writes_audit_log_on_failure(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.ndjson"
    indexer = WorkspaceIndexer(
        workspace=str(tmp_path),
        audit_log_path=str(audit_path),
    )
    with pytest.raises(IndexerError):
        await indexer.index_file(str(tmp_path / "nonexistent.md"))

    assert audit_path.exists()
    record = json.loads(audit_path.read_text().strip())
    assert record["type"] == "indexer.index_file.failed"
    assert record["component"] == "kb_indexer"


# ---------------------------------------------------------------------------
# scan_workspace (via start)
# ---------------------------------------------------------------------------


async def test_scan_emits_initial_scan_events(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nContent.\n")
    (tmp_path / "b.txt").write_text("Hello.\n")

    received: list[IndexEvent] = []

    async def handler(event: IndexEvent) -> None:
        received.append(event)

    indexer = WorkspaceIndexer(workspace=str(tmp_path), on_index_event=handler)
    await indexer.start()
    indexer.stop()

    event_kinds = {e.event_kind for e in received}
    assert "initial_scan" in event_kinds

    file_paths = {e.file_path for e in received}
    assert str(tmp_path / "a.md") in file_paths
    assert str(tmp_path / "b.txt") in file_paths


async def test_scan_skips_ignored_dirs(tmp_path: Path) -> None:
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "foo.pyc").write_bytes(b"bytecode")
    (tmp_path / "real.txt").write_text("content")

    received: list[IndexEvent] = []

    async def handler(event: IndexEvent) -> None:
        received.append(event)

    indexer = WorkspaceIndexer(workspace=str(tmp_path), on_index_event=handler)
    await indexer.start()
    indexer.stop()

    cache_events = [e for e in received if "__pycache__" in e.file_path]
    assert len(cache_events) == 0


async def test_scan_emits_otel_span_for_start(tmp_path: Path, otel_exporter) -> None:
    (tmp_path / "note.txt").write_text("text")
    indexer = WorkspaceIndexer(workspace=str(tmp_path))
    await indexer.start()
    indexer.stop()

    span_names = [s.name for s in otel_exporter.get_finished_spans()]
    assert "indexer.start" in span_names
    assert "indexer.scan_workspace" in span_names


# ---------------------------------------------------------------------------
# stop() safety
# ---------------------------------------------------------------------------


async def test_stop_before_start_is_safe(tmp_path: Path) -> None:
    indexer = WorkspaceIndexer(workspace=str(tmp_path))
    indexer.stop()  # should not raise


async def test_stop_twice_is_safe(tmp_path: Path) -> None:
    indexer = WorkspaceIndexer(workspace=str(tmp_path))
    await indexer.start()
    indexer.stop()
    indexer.stop()  # should not raise
