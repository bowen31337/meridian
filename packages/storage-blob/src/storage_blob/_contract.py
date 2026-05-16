from __future__ import annotations

from abc import ABC, abstractmethod


class BlobStore(ABC):
    """
    Contract every blob-store backend must implement.

    Pass a concrete implementation to BlobRuntime, which wraps each call with
    an OTel span, a structured invocation event, and audit-log writes on failure.
    """

    @abstractmethod
    async def put(self, key: str, data: bytes) -> None:
        """Write data under key, overwriting any existing value."""

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Return the bytes stored under key.

        Raises BlobFailure(BLOB_KEY_NOT_FOUND) if the key does not exist.
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove the blob stored under key. No-op if the key does not exist."""
