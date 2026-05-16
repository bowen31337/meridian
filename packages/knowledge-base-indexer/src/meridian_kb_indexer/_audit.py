from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_DEFAULT_AUDIT_LOG = os.environ.get(
    "MERIDIAN_AUDIT_LOG",
    os.path.join(os.path.expanduser("~"), ".meridian", "audit.ndjson"),
)


def write_audit_event(
    event_type: str,
    file_path: str | None = None,
    error: dict[str, Any] | None = None,
    audit_log_path: str | None = None,
) -> None:
    """Append a single structured audit event to the NDJSON audit log.

    Called on every indexer failure so operators can reconstruct failures
    without live OTel infrastructure.  Never raises — audit writes must not
    mask the underlying error.
    """
    record: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type": event_type,
        "component": "kb_indexer",
    }
    if file_path:
        record["file_path"] = file_path
    if error:
        record["error"] = error

    line = json.dumps(record, separators=(",", ":"))
    path = Path(audit_log_path or _DEFAULT_AUDIT_LOG)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
