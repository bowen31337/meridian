"""
Blob-store conformance suite.

Covers BlobRuntime (via a StubBlobStore) and LocalBlobStore (via tmp_path):

  BlobRuntime:
    - put / get / delete success: span emitted, invocation event attached,
      no audit entries, correct data returned.
    - get key-not-found (BLOB_KEY_NOT_FOUND): BlobFailure raised, audit entry
      written at "error", span set to ERROR.
    - Store raises unexpected exception: wrapped in BlobFailure with correct
      code, cause preserved, audit entry written, span marked ERROR.
    - on_error callback invoked on every failure.
    - Span lifecycle: span ended on both success and failure paths.

  LocalBlobStore:
    - put / get / delete round-trip with real filesystem.
    - get missing key raises BlobFailure(BLOB_KEY_NOT_FOUND).
    - delete of missing key is a no-op (no exception).
    - Hierarchical keys (e.g. "a/b/c") create parent directories.
    - Path-traversal keys raise BlobFailure(BLOB_KEY_INVALID).
"""

from __future__ import annotations

from pathlib import Path

from opentelemetry.trace import StatusCode
import pytest
from storage_blob import (
    AuditLogEntry,
    BlobFailure,
    BlobOptions,
    BlobRuntime,
    BlobStore,
    LocalBlobStore,
)

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Stub store
# ---------------------------------------------------------------------------


class StubBlobStore(BlobStore):
    """In-memory store with configurable failure injection."""

    def __init__(
        self,
        *,
        put_raises: Exception | None = None,
        get_raises: Exception | None = None,
        delete_raises: Exception | None = None,
    ) -> None:
        self._put_raises = put_raises
        self._get_raises = get_raises
        self._delete_raises = delete_raises
        self._data: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        if self._put_raises:
            raise self._put_raises
        self._data[key] = data

    async def get(self, key: str) -> bytes:
        if self._get_raises:
            raise self._get_raises
        if key not in self._data:
            from storage_blob._local import _now

            raise BlobFailure(
                code="BLOB_KEY_NOT_FOUND",
                message=f"Key not found: {key!r}",
                key=key,
                timestamp=_now(),
            )
        return self._data[key]

    async def delete(self, key: str) -> None:
        if self._delete_raises:
            raise self._delete_raises
        self._data.pop(key, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_options(audit: CapturingAuditLog, errors: list[BlobFailure] | None = None) -> BlobOptions:
    return BlobOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


def make_runtime(store: BlobStore | None = None) -> BlobRuntime:
    return BlobRuntime(store or StubBlobStore())


# ---------------------------------------------------------------------------
# put — success
# ---------------------------------------------------------------------------


class TestPutSuccess:
    async def test_data_stored(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        store = StubBlobStore()
        rt = BlobRuntime(store)
        await rt.put("k1", b"hello", make_options(audit_log))
        assert store._data["k1"] == b"hello"

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().put("k1", b"x", make_options(audit_log))
        assert mock_span.name == "blob.put"

    async def test_span_key_attribute(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().put("k1", b"x", make_options(audit_log))
        assert mock_span.attributes["blob.key"] == "k1"

    async def test_invocation_event_attached(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().put("k1", b"x", make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "blob.invocation" in event_names

    async def test_invocation_event_operation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().put("k1", b"x", make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "blob.invocation")
        assert inv[1]["operation"] == "put"

    async def test_no_audit_entries_on_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().put("k1", b"x", make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().put("k1", b"x", make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# put — store raises
# ---------------------------------------------------------------------------


class TestPutStoreRaises:
    async def test_wraps_as_put_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = BlobRuntime(StubBlobStore(put_raises=RuntimeError("disk full")))
        with pytest.raises(BlobFailure) as exc_info:
            await rt.put("k1", b"x", make_options(audit_log))
        assert exc_info.value.code == "BLOB_PUT_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = RuntimeError("disk full")
        rt = BlobRuntime(StubBlobStore(put_raises=orig))
        with pytest.raises(BlobFailure) as exc_info:
            await rt.put("k1", b"x", make_options(audit_log))
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = BlobRuntime(StubBlobStore(put_raises=RuntimeError("boom")))
        with pytest.raises(BlobFailure):
            await rt.put("k1", b"x", make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "blob.put.failed"
        assert entry.key == "k1"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = BlobRuntime(StubBlobStore(put_raises=RuntimeError("boom")))
        with pytest.raises(BlobFailure):
            await rt.put("k1", b"x", make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_error_event_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = BlobRuntime(StubBlobStore(put_raises=RuntimeError("boom")))
        with pytest.raises(BlobFailure):
            await rt.put("k1", b"x", make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "blob.error" in event_names

    async def test_exception_recorded_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = RuntimeError("boom")
        rt = BlobRuntime(StubBlobStore(put_raises=orig))
        with pytest.raises(BlobFailure):
            await rt.put("k1", b"x", make_options(audit_log))
        assert orig in mock_span.recorded_exceptions

    async def test_on_error_callback(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[BlobFailure] = []
        rt = BlobRuntime(StubBlobStore(put_raises=RuntimeError("boom")))
        with pytest.raises(BlobFailure):
            await rt.put("k1", b"x", make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "BLOB_PUT_FAILED"

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = BlobRuntime(StubBlobStore(put_raises=RuntimeError("boom")))
        with pytest.raises(BlobFailure):
            await rt.put("k1", b"x", make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# get — success
# ---------------------------------------------------------------------------


class TestGetSuccess:
    async def test_returns_data(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        store = StubBlobStore()
        store._data["k1"] = b"world"
        result = await BlobRuntime(store).get("k1", make_options(audit_log))
        assert result == b"world"

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        store = StubBlobStore()
        store._data["k1"] = b"x"
        await BlobRuntime(store).get("k1", make_options(audit_log))
        assert mock_span.name == "blob.get"

    async def test_invocation_event_operation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        store = StubBlobStore()
        store._data["k1"] = b"x"
        await BlobRuntime(store).get("k1", make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "blob.invocation")
        assert inv[1]["operation"] == "get"

    async def test_no_audit_entries_on_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        store = StubBlobStore()
        store._data["k1"] = b"x"
        await BlobRuntime(store).get("k1", make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        store = StubBlobStore()
        store._data["k1"] = b"x"
        await BlobRuntime(store).get("k1", make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# get — key not found (BLOB_KEY_NOT_FOUND)
# ---------------------------------------------------------------------------


class TestGetKeyNotFound:
    async def test_raises_blob_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(BlobFailure) as exc_info:
            await make_runtime().get("missing", make_options(audit_log))
        assert exc_info.value.code == "BLOB_KEY_NOT_FOUND"

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(BlobFailure):
            await make_runtime().get("missing", make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "blob.get.failed"
        assert entry.key == "missing"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(BlobFailure):
            await make_runtime().get("missing", make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_on_error_callback(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[BlobFailure] = []
        with pytest.raises(BlobFailure):
            await make_runtime().get("missing", make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "BLOB_KEY_NOT_FOUND"

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(BlobFailure):
            await make_runtime().get("missing", make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# get — store raises unexpected exception
# ---------------------------------------------------------------------------


class TestGetStoreRaises:
    async def test_wraps_as_get_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = BlobRuntime(StubBlobStore(get_raises=OSError("read error")))
        with pytest.raises(BlobFailure) as exc_info:
            await rt.get("k1", make_options(audit_log))
        assert exc_info.value.code == "BLOB_GET_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = OSError("read error")
        rt = BlobRuntime(StubBlobStore(get_raises=orig))
        with pytest.raises(BlobFailure) as exc_info:
            await rt.get("k1", make_options(audit_log))
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = BlobRuntime(StubBlobStore(get_raises=OSError("boom")))
        with pytest.raises(BlobFailure):
            await rt.get("k1", make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].level == "error"


# ---------------------------------------------------------------------------
# delete — success
# ---------------------------------------------------------------------------


class TestDeleteSuccess:
    async def test_key_removed(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        store = StubBlobStore()
        store._data["k1"] = b"x"
        await BlobRuntime(store).delete("k1", make_options(audit_log))
        assert "k1" not in store._data

    async def test_missing_key_is_noop(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().delete("nonexistent", make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().delete("k1", make_options(audit_log))
        assert mock_span.name == "blob.delete"

    async def test_invocation_event_operation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().delete("k1", make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "blob.invocation")
        assert inv[1]["operation"] == "delete"

    async def test_no_audit_entries_on_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await make_runtime().delete("k1", make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await make_runtime().delete("k1", make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# delete — store raises
# ---------------------------------------------------------------------------


class TestDeleteStoreRaises:
    async def test_wraps_as_delete_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = BlobRuntime(StubBlobStore(delete_raises=OSError("locked")))
        with pytest.raises(BlobFailure) as exc_info:
            await rt.delete("k1", make_options(audit_log))
        assert exc_info.value.code == "BLOB_DELETE_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = OSError("locked")
        rt = BlobRuntime(StubBlobStore(delete_raises=orig))
        with pytest.raises(BlobFailure) as exc_info:
            await rt.delete("k1", make_options(audit_log))
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        rt = BlobRuntime(StubBlobStore(delete_raises=OSError("boom")))
        with pytest.raises(BlobFailure):
            await rt.delete("k1", make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].event == "blob.delete.failed"

    async def test_on_error_callback(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[BlobFailure] = []
        rt = BlobRuntime(StubBlobStore(delete_raises=OSError("boom")))
        with pytest.raises(BlobFailure):
            await rt.delete("k1", make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "BLOB_DELETE_FAILED"


# ---------------------------------------------------------------------------
# LocalBlobStore — filesystem integration
# ---------------------------------------------------------------------------


class TestLocalBlobStore:
    async def test_put_get_roundtrip(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path)
        await store.put("file1", b"data")
        assert await store.get("file1") == b"data"

    async def test_put_overwrites(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path)
        await store.put("k", b"first")
        await store.put("k", b"second")
        assert await store.get("k") == b"second"

    async def test_get_missing_key_raises(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path)
        with pytest.raises(BlobFailure) as exc_info:
            await store.get("missing")
        assert exc_info.value.code == "BLOB_KEY_NOT_FOUND"

    async def test_delete_removes_file(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path)
        await store.put("k", b"x")
        await store.delete("k")
        with pytest.raises(BlobFailure):
            await store.get("k")

    async def test_delete_missing_key_is_noop(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path)
        await store.delete("nonexistent")  # must not raise

    async def test_hierarchical_key_creates_dirs(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path)
        await store.put("a/b/c", b"nested")
        assert await store.get("a/b/c") == b"nested"
        assert (tmp_path / "a" / "b" / "c").exists()

    async def test_path_traversal_raises(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path)
        with pytest.raises(BlobFailure) as exc_info:
            await store.get("../../etc/passwd")
        assert exc_info.value.code == "BLOB_KEY_INVALID"

    async def test_absolute_key_raises(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path)
        with pytest.raises(BlobFailure) as exc_info:
            await store.put("/etc/shadow", b"x")
        assert exc_info.value.code == "BLOB_KEY_INVALID"

    async def test_large_payload(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path)
        data = b"x" * (4 * 1024 * 1024)  # 4 MiB
        await store.put("big", data)
        assert await store.get("big") == data

    async def test_empty_payload(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path)
        await store.put("empty", b"")
        assert await store.get("empty") == b""
