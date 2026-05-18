"""NDJSON audit log written to ~/.meridian/audit.ndjson on every CLI invocation."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

MERIDIAN_DIR = Path.home() / ".meridian"
AUDIT_LOG = MERIDIAN_DIR / "audit.ndjson"


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def write_audit(level: str, event: str, detail: dict[str, object] | None = None) -> None:
    MERIDIAN_DIR.mkdir(parents=True, exist_ok=True)
    entry: dict[str, object] = {"ts": _now(), "level": level, "event": event}
    if detail:
        entry["detail"] = detail
    with AUDIT_LOG.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
