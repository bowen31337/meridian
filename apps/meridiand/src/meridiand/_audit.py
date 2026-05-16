from __future__ import annotations

import json
import os
from pathlib import Path

from core_errors import AuditLog, AuditLogEntry


class FileAuditLog(AuditLog):
    """Appends audit entries as NDJSON lines to $storage_root/audit.ndjson using O_APPEND."""

    def __init__(self, storage_root: Path) -> None:
        storage_root.mkdir(parents=True, exist_ok=True)
        self._path = storage_root / "audit.ndjson"

    def write(self, entry: AuditLogEntry) -> None:
        record: dict[str, object] = {
            "level": entry.level,
            "event": entry.event,
            "code": entry.code,
            "timestamp": entry.timestamp,
        }
        if entry.detail:
            record["detail"] = entry.detail
        line = json.dumps(record, separators=(",", ":")) + "\n"
        fd = os.open(str(self._path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
