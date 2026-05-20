from __future__ import annotations

from dataclasses import dataclass

from meridian_plugin_loader import PluginLoader
from storage_blob._local import LocalBlobStore
from storage_event_log._local import LocalEventLogWriter

from ._audit import FileAuditLog
from ._config import MeridianConfig
from ._signing import DaemonSigningKey


@dataclass
class Services:
    audit_log: FileAuditLog
    blob_store: LocalBlobStore
    event_log: LocalEventLogWriter
    plugin_loader: PluginLoader


def init_services(config: MeridianConfig) -> Services:
    root = config.storage_root
    root.mkdir(parents=True, exist_ok=True)
    signing_key = DaemonSigningKey(root) if config.audit_signing.enabled else None
    audit_log = FileAuditLog(root, signing_key=signing_key)
    return Services(
        audit_log=audit_log,
        blob_store=LocalBlobStore(root),
        event_log=LocalEventLogWriter(root),
        plugin_loader=PluginLoader(audit_log=audit_log),
    )
