from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


# Default audit log location; overridable via MERIDIAN_AUDIT_LOG env var or
# the audit_log_path kwarg on individual tool builders.
_DEFAULT_AUDIT_LOG = os.environ.get(
    "MERIDIAN_AUDIT_LOG",
    os.path.join(os.path.expanduser("~"), ".meridian", "audit.ndjson"),
)


def write_audit_event(
    event_type: str,
    tool_name: str,
    session_id: str | None = None,
    error: dict[str, Any] | None = None,
    audit_log_path: str | None = None,
) -> None:
    """Append a single structured audit event to the audit log (NDJSON format).

    Called on every tool failure so the operator can reconstruct what went
    wrong without needing live OTel infrastructure (Architecture §22.4).
    """
    record: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type": event_type,
        "tool_name": tool_name,
    }
    if session_id:
        record["session_id"] = session_id
    if error:
        record["error"] = error

    line = json.dumps(record, separators=(",", ":"))

    path = Path(audit_log_path or _DEFAULT_AUDIT_LOG)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        # Never let an audit-log write failure propagate — the tool result
        # must still reach the caller.
        pass
