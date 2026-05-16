from __future__ import annotations

from dataclasses import dataclass

from meridian_plugin_loader import PluginLoader
from storage_blob._local import LocalBlobStore
from storage_event_log._local import LocalEventLogWriter

from ._audit import FileAuditLog
from ._config import DaemonConfig


@dataclass
class Services:
    audit_log: FileAuditLog
    blob_store: LocalBlobStore
    event_log: LocalEventLogWriter
    plugin_loader: PluginLoader


def init_services(config: DaemonConfig) -> Services:
    root = config.storage_root
    root.mkdir(parents=True, exist_ok=True)
    audit_log = FileAuditLog(root)
    return Services(
        audit_log=audit_log,
        blob_store=LocalBlobStore(root),
        event_log=LocalEventLogWriter(root),
        plugin_loader=PluginLoader(audit_log=audit_log),
    )
