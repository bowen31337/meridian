from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from core_errors import AuditLog, AuditLogEntry, MeridianError

from ._signing import DaemonSigningKey


class AuditSignFailedError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="audit_sign_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class FileAuditLog(AuditLog):
    """Appends audit entries as NDJSON lines to $storage_root/audit.ndjson using O_APPEND.

    When signing_key is provided each entry includes a "sig" field: a base64-encoded
    Ed25519 signature over the canonical JSON (sort_keys=True) of the record without
    the "sig" field itself.  On signing failure the error is written to the log unsigned
    and AuditSignFailedError is raised so the caller is informed.
    """

    def __init__(self, storage_root: Path, *, signing_key: DaemonSigningKey | None = None) -> None:
        storage_root.mkdir(parents=True, exist_ok=True)
        self._path = storage_root / "audit.ndjson"
        self._signing_key = signing_key

    def write(self, entry: AuditLogEntry) -> None:
        record: dict[str, object] = {
            "level": entry.level,
            "event": entry.event,
            "code": entry.code,
            "timestamp": entry.timestamp,
        }
        if entry.detail:
            record["detail"] = entry.detail

        if self._signing_key is not None:
            try:
                payload = json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
                record["sig"] = self._signing_key.sign(payload)
            except Exception as exc:
                now = datetime.now(UTC).isoformat()
                with contextlib.suppress(Exception):
                    self._write_raw(
                        {
                            "level": "error",
                            "event": "audit.sign.failed",
                            "code": "audit_sign_failed",
                            "timestamp": now,
                            "detail": {"message": str(exc)},
                        }
                    )
                raise AuditSignFailedError(
                    message=f"Failed to sign audit entry: {exc}",
                    timestamp=now,
                    cause=exc,
                ) from exc

        self._write_raw(record)

    def _write_raw(self, record: dict[str, object]) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        fd = os.open(str(self._path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
