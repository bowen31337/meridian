from __future__ import annotations

from dataclasses import dataclass

from storage_blob._local import LocalBlobStore
from storage_event_log._local import LocalEventLogWriter

from ._audit import FileAuditLog
from ._config import DaemonConfig


@dataclass
class Services:
    audit_log: FileAuditLog
    blob_store: LocalBlobStore
    event_log: LocalEventLogWriter


def init_services(config: DaemonConfig) -> Services:
    root = config.storage_root
    root.mkdir(parents=True, exist_ok=True)
    return Services(
        audit_log=FileAuditLog(root),
        blob_store=LocalBlobStore(root),
        event_log=LocalEventLogWriter(root),
    )
