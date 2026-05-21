from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ImportAuditLog:
    """Per-import NDJSON audit log written to storage_root/meridian-import-<timestamp>.audit.ndjson.

    Each import invocation gets its own file.  Entries are appended using O_APPEND
    so concurrent writers (if any) interleave safely at line boundaries.
    """

    def __init__(self, storage_root: Path, *, source: str, timestamp: str) -> None:
        storage_root.mkdir(parents=True, exist_ok=True)
        safe_ts = timestamp.replace(":", "-").split("+")[0].split(".")[0]
        filename = f"meridian-import-{safe_ts}.audit.ndjson"
        self._path = storage_root / filename
        self._source = source

    @property
    def path(self) -> Path:
        return self._path

    def write_started(self, record_count: int, ts: str | None = None) -> None:
        self._append(
            {
                "type": "import_started",
                "source": self._source,
                "record_count": record_count,
                "ts": ts or _now(),
            }
        )

    def write_record_translated(
        self,
        *,
        seq: int,
        source_id: str,
        target_id: str,
        kind: str,
        lossy_fields: list[str],
        ts: str | None = None,
    ) -> None:
        self._append(
            {
                "type": "record_translated",
                "seq": seq,
                "source_id": source_id,
                "target_id": target_id,
                "kind": kind,
                "lossy_fields": lossy_fields,
                "ts": ts or _now(),
            }
        )

    def write_checklist(self, items: list[str], ts: str | None = None) -> None:
        self._append(
            {
                "type": "checklist",
                "items": items,
                "ts": ts or _now(),
            }
        )

    def write_completed(
        self,
        *,
        total: int,
        lossy_count: int,
        ts: str | None = None,
    ) -> None:
        self._append(
            {
                "type": "import_completed",
                "source": self._source,
                "total": total,
                "lossy_count": lossy_count,
                "audit_path": self._path.name,
                "ts": ts or _now(),
            }
        )

    def write_failed(self, *, code: str, message: str, ts: str | None = None) -> None:
        self._append(
            {
                "type": "import_failed",
                "source": self._source,
                "code": code,
                "message": message,
                "ts": ts or _now(),
            }
        )

    def _append(self, record: dict[str, object]) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        fd = os.open(str(self._path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
