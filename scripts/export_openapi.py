#!/usr/bin/env python
"""Export the meridiand FastAPI app's OpenAPI spec to packages/schemas/openapi.yaml.

Usage: uv run python scripts/export_openapi.py
"""

from __future__ import annotations

import datetime
import json
import sys
import tempfile
from pathlib import Path

import yaml
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_REPO_ROOT = Path(__file__).parent.parent
_DEST = _REPO_ROOT / "packages" / "schemas" / "openapi.yaml"
_AUDIT_DIR = _REPO_ROOT / ".meridian"
_AUDIT_LOG = _AUDIT_DIR / "make-audit.ndjson"

_TRACER_NAME = "meridian.make"
_tracer = trace.get_tracer(_TRACER_NAME)


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _audit(level: str, event: str, detail: dict | None = None) -> None:
    _AUDIT_DIR.mkdir(exist_ok=True)
    entry: dict = {"ts": _now(), "level": level, "event": event}
    if detail:
        entry["detail"] = detail
    with _AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> int:
    _audit("info", "openapi.export.invoked")
    with _tracer.start_as_current_span("openapi.export") as span:
        try:
            from core_errors import NoopAuditLog
            from meridiand._app import create_app

            with tempfile.TemporaryDirectory() as tmp:
                app = create_app(NoopAuditLog(), storage_root=Path(tmp))
                spec = app.openapi()

            yaml_content = yaml.dump(spec, default_flow_style=False, allow_unicode=True)
            _DEST.parent.mkdir(parents=True, exist_ok=True)
            _DEST.write_text(yaml_content)

            span.add_event(
                "openapi.export.done",
                {"openapi.dest": str(_DEST), "openapi.bytes": len(yaml_content)},
            )
            _audit("info", "openapi.export.ok", {"dest": str(_DEST), "bytes": len(yaml_content)})
            print(f"openapi.yaml written → {_DEST}")
            return 0

        except Exception as exc:
            msg = f"openapi export failed: {exc}"
            span.set_status(Status(StatusCode.ERROR, msg))
            span.add_event(
                "openapi.export.error",
                {"error.type": type(exc).__name__, "error.message": str(exc)},
            )
            span.record_exception(exc)
            _audit(
                "error",
                "openapi.export.failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            print(f"error: {msg}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
