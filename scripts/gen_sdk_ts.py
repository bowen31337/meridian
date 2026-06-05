#!/usr/bin/env python
"""Generate packages/sdk-ts/src from packages/schemas/openapi.yaml using openapi-typescript.

Usage: uv run python scripts/gen_sdk_ts.py
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import subprocess
import sys

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_REPO_ROOT = Path(__file__).parent.parent
_SPEC = _REPO_ROOT / "packages" / "schemas" / "openapi.yaml"
_OUT = _REPO_ROOT / "packages" / "sdk-ts" / "src" / "index.ts"
_BIN = _REPO_ROOT / "node_modules" / ".bin" / "openapi-typescript"
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
    _audit("info", "sdk_ts.gen.invoked")
    with _tracer.start_as_current_span("sdk_ts.gen") as span:
        try:
            if not _SPEC.exists():
                msg = (
                    f"openapi spec not found: {_SPEC} — run 'make codegen'"
                    " step by step or export_openapi.py first"
                )
                print(f"error: {msg}", file=sys.stderr)
                span.set_status(Status(StatusCode.ERROR, msg))
                _audit("error", "sdk_ts.gen.failed", {"error": msg})
                return 1

            if not _BIN.exists():
                msg = f"openapi-typescript not found at {_BIN} — run 'pnpm install' first"
                print(f"error: {msg}", file=sys.stderr)
                span.set_status(Status(StatusCode.ERROR, msg))
                _audit("error", "sdk_ts.gen.failed", {"error": msg})
                return 1

            _OUT.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(_BIN), str(_SPEC), "--output", str(_OUT)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))

            if result.returncode != 0:
                msg = result.stderr or result.stdout
                print(f"error: openapi-typescript failed:\n{msg}", file=sys.stderr)
                span.set_status(Status(StatusCode.ERROR, "openapi-typescript failed"))
                span.add_event(
                    "sdk_ts.gen.error",
                    {"error.type": "subprocess", "error.message": msg},
                )
                _audit(
                    "error",
                    "sdk_ts.gen.failed",
                    {"error": msg, "cmd": " ".join(str(c) for c in cmd)},
                )
                return 1

            bytes_written = _OUT.stat().st_size
            span.add_event(
                "sdk_ts.gen.done",
                {"sdk_ts.dest": str(_OUT), "sdk_ts.bytes": bytes_written},
            )
            _audit("info", "sdk_ts.gen.ok", {"dest": str(_OUT), "bytes": bytes_written})
            print(f"sdk-ts generated → {_OUT}")
            return 0

        except Exception as exc:
            msg = f"sdk-ts generation failed: {exc}"
            span.set_status(Status(StatusCode.ERROR, msg))
            span.record_exception(exc)
            _audit(
                "error",
                "sdk_ts.gen.failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            print(f"error: {msg}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
