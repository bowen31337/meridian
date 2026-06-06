"""Branch-coverage completion for indexer.py, _audit.py, _telemetry.py, and the
_watcher bridge handler — the failure and dispatch paths the happy-path suites
do not reach."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from meridian_kb_indexer import _audit
from meridian_kb_indexer import indexer as _indexer_mod
from meridian_kb_indexer._audit import write_audit_event
from meridian_kb_indexer._telemetry import record_indexer_failure
from meridian_kb_indexer._types import IndexerError, IndexEvent
from meridian_kb_indexer._watcher import _BridgeHandler
from meridian_kb_indexer.indexer import WorkspaceIndexer

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSpan:
    def __init__(self) -> None:
        self.status: Any = None
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.exceptions: list[BaseException] = []

    def set_status(self, status: Any) -> None:
        self.status = status

    def add_event(self, name: str, attrs: dict[str, Any]) -> None:
        self.events.append((name, attrs))

    def record_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)


class _DirEvent:
    """Minimal watchdog-style event that is a directory."""

    is_directory = True
    src_path = "/workspace/some/dir"


class _FileEvent:
    """Minimal watchdog-style event for a regular file."""

    is_directory = False
    src_path = "/workspace/src/gone.py"


# ---------------------------------------------------------------------------
# indexer.start failure paths (78-87)
# ---------------------------------------------------------------------------


async def test_start_reraises_indexer_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    indexer = WorkspaceIndexer(workspace=str(tmp_path))

    async def _boom() -> None:
        raise IndexerError("INNER", "scan blew up")

    monkeypatch.setattr(indexer, "_scan_workspace", _boom)
    with pytest.raises(IndexerError) as exc_info:
        await indexer.start()
    assert exc_info.value.code == "INNER"


async def test_start_wraps_generic_error_and_audits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit_path = tmp_path / "audit.ndjson"
    indexer = WorkspaceIndexer(workspace=str(tmp_path), audit_log_path=str(audit_path))

    async def _boom() -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(indexer, "_scan_workspace", _boom)
    with pytest.raises(IndexerError) as exc_info:
        await indexer.start()
    assert exc_info.value.code == "INDEXER_START_FAILED"
    record = json.loads(audit_path.read_text().strip())
    assert record["type"] == "indexer.start.failed"
    assert record["error"]["type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# index_file re-raises an IndexerError unchanged (118)
# ---------------------------------------------------------------------------


async def test_index_file_reraises_indexer_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hello")
    indexer = WorkspaceIndexer(workspace=str(tmp_path))

    def _boom(_path: str, _content: str) -> Any:
        raise IndexerError("PRECOMPUTED", "already structured")

    monkeypatch.setattr(_indexer_mod, "chunk_file", _boom)
    with pytest.raises(IndexerError) as exc_info:
        await indexer.index_file(str(f))
    assert exc_info.value.code == "PRECOMPUTED"


# ---------------------------------------------------------------------------
# _scan_workspace swallows per-file IndexerError (159-160)
# ---------------------------------------------------------------------------


async def test_scan_swallows_per_file_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a.txt").write_text("content")
    received: list[IndexEvent] = []

    async def handler(event: IndexEvent) -> None:
        received.append(event)

    indexer = WorkspaceIndexer(workspace=str(tmp_path), on_index_event=handler)

    async def _boom(_path: str) -> Any:
        raise IndexerError("INDEXER_FILE_FAILED", "unreadable")

    monkeypatch.setattr(indexer, "index_file", _boom)
    await indexer._scan_workspace()
    assert received == []  # failure swallowed, no event emitted


# ---------------------------------------------------------------------------
# _handle_change: deleted / success / failure / no-callback (169-191)
# ---------------------------------------------------------------------------


def _evt(kind: str, file_path: str) -> IndexEvent:
    return IndexEvent(event_kind=kind, file_path=file_path, timestamp="2026-06-06T00:00:00Z")


async def test_handle_change_deleted_forwards_event(tmp_path: Path) -> None:
    received: list[IndexEvent] = []

    async def handler(event: IndexEvent) -> None:
        received.append(event)

    indexer = WorkspaceIndexer(workspace=str(tmp_path), on_index_event=handler)
    await indexer._handle_change(_evt("deleted", str(tmp_path / "gone.txt")))
    assert len(received) == 1 and received[0].event_kind == "deleted"


async def test_handle_change_deleted_without_callback_is_safe(tmp_path: Path) -> None:
    indexer = WorkspaceIndexer(workspace=str(tmp_path))
    await indexer._handle_change(_evt("deleted", str(tmp_path / "gone.txt")))


async def test_handle_change_success_enriches_with_chunks(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("Paragraph one.\n\nParagraph two.\n")
    received: list[IndexEvent] = []

    async def handler(event: IndexEvent) -> None:
        received.append(event)

    indexer = WorkspaceIndexer(workspace=str(tmp_path), on_index_event=handler)
    await indexer._handle_change(_evt("modified", str(f)))
    assert len(received) == 1
    assert received[0].chunks
    assert received[0].error is None


async def test_handle_change_failure_sets_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: list[IndexEvent] = []

    async def handler(event: IndexEvent) -> None:
        received.append(event)

    indexer = WorkspaceIndexer(workspace=str(tmp_path), on_index_event=handler)

    async def _boom(_path: str) -> Any:
        raise IndexerError("INDEXER_FILE_FAILED", "boom")

    monkeypatch.setattr(indexer, "index_file", _boom)
    await indexer._handle_change(_evt("created", str(tmp_path / "x.txt")))
    assert len(received) == 1
    assert received[0].error == "boom"


async def test_handle_change_success_without_callback_is_safe(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("text\n")
    indexer = WorkspaceIndexer(workspace=str(tmp_path))
    await indexer._handle_change(_evt("modified", str(f)))


# ---------------------------------------------------------------------------
# _audit: optional fields skipped + OSError swallowed (32->34, 34->37, 43-44)
# ---------------------------------------------------------------------------


def test_write_audit_event_minimal_record(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.ndjson"
    write_audit_event("indexer.ping", audit_log_path=str(audit_path))
    record = json.loads(audit_path.read_text().strip())
    assert record["type"] == "indexer.ping"
    assert "file_path" not in record
    assert "error" not in record


def test_write_audit_event_swallows_os_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    # Parent path traverses through a regular file -> mkdir raises NotADirectoryError.
    bad_path = blocker / "sub" / "audit.ndjson"
    write_audit_event("indexer.fail", file_path="x", audit_log_path=str(bad_path))  # no raise


def test_default_audit_log_constant_is_under_meridian() -> None:
    assert _audit._DEFAULT_AUDIT_LOG.endswith(".meridian/audit.ndjson")


# ---------------------------------------------------------------------------
# _telemetry: record_indexer_failure without file_path (45->47)
# ---------------------------------------------------------------------------


def test_record_indexer_failure_without_file_path() -> None:
    span = _FakeSpan()
    record_indexer_failure(span, RuntimeError("nope"), operation="start")  # type: ignore[arg-type]
    name, attrs = span.events[0]
    assert name == "indexer.error"
    assert "indexer.file_path" not in attrs
    assert span.exceptions


def test_record_indexer_failure_with_file_path() -> None:
    span = _FakeSpan()
    record_indexer_failure(
        span,  # type: ignore[arg-type]
        ValueError("bad"),
        operation="index_file",
        file_path="src/a.py",
    )
    _name, attrs = span.events[0]
    assert attrs["indexer.file_path"] == "src/a.py"


# ---------------------------------------------------------------------------
# _watcher: on_deleted ignores directory events (59->exit)
# ---------------------------------------------------------------------------


def test_bridge_handler_on_deleted_ignores_directory() -> None:
    fired: list[IndexEvent] = []

    handler = _BridgeHandler.__new__(_BridgeHandler)
    handler._fire = fired.append  # type: ignore[method-assign,assignment]
    handler.on_deleted(_DirEvent())  # type: ignore[arg-type]
    assert fired == []


def test_bridge_handler_on_deleted_fires_for_file() -> None:
    fired: list[IndexEvent] = []

    handler = _BridgeHandler.__new__(_BridgeHandler)
    handler._fire = fired.append  # type: ignore[method-assign,assignment]
    handler.on_deleted(_FileEvent())  # type: ignore[arg-type]
    assert len(fired) == 1
    assert fired[0].event_kind == "deleted"
    assert fired[0].file_path == "/workspace/src/gone.py"
