from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from pathlib import Path
import posixpath

from ._contract import BlobStore
from ._types import BlobFailure


def _now() -> str:
    return datetime.now(UTC).isoformat()


class LocalBlobStore(BlobStore):
    """
    Local filesystem blob store.

    Each key maps to a file under root_dir. Hierarchical keys (e.g.
    "files/abc123") map to nested directories. Keys that would escape the
    root via path traversal raise BlobFailure(BLOB_KEY_INVALID).
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)

    def _path(self, key: str) -> Path:
        normalized = posixpath.normpath(key)
        if normalized.startswith("..") or normalized.startswith("/"):
            raise BlobFailure(
                code="BLOB_KEY_INVALID",
                message=f"Key escapes storage root: {key!r}",
                key=key,
                timestamp=_now(),
            )
        return self._root / normalized

    async def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise BlobFailure(
                code="BLOB_KEY_NOT_FOUND",
                message=f"Key not found: {key!r}",
                key=key,
                timestamp=_now(),
            )
        return path.read_bytes()

    async def delete(self, key: str) -> None:
        path = self._path(key)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
