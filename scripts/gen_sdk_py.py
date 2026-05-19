#!/usr/bin/env python
"""Generate packages/sdk-py/src from packages/schemas/openapi.yaml using datamodel-code-generator.

Usage: uv run python scripts/gen_sdk_py.py
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_REPO_ROOT = Path(__file__).parent.parent
_SPEC = _REPO_ROOT / "packages" / "schemas" / "openapi.yaml"
_OUT = _REPO_ROOT / "packages" / "sdk-py" / "src" / "meridian_sdk_py"
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
    _audit("info", "sdk_py.gen.invoked")
    with _tracer.start_as_current_span("sdk_py.gen") as span:
        try:
            if not _SPEC.exists():
                msg = f"openapi spec not found: {_SPEC} — run 'make codegen' step by step or export_openapi.py first"
                print(f"error: {msg}", file=sys.stderr)
                span.set_status(Status(StatusCode.ERROR, msg))
                _audit("error", "sdk_py.gen.failed", {"error": msg})
                return 1

            _OUT.mkdir(parents=True, exist_ok=True)
            cmd = [
                "uv",
                "run",
                "datamodel-codegen",
                "--input",
                str(_SPEC),
                "--input-file-type",
                "openapi",
                "--output",
                str(_OUT),
                "--output-model-type",
                "pydantic_v2.BaseModel",
                "--target-python-version",
                "3.11",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))

            if result.returncode != 0:
                msg = result.stderr or result.stdout
                print(f"error: datamodel-codegen failed:\n{msg}", file=sys.stderr)
                span.set_status(Status(StatusCode.ERROR, "datamodel-codegen failed"))
                span.add_event(
                    "sdk_py.gen.error",
                    {"error.type": "subprocess", "error.message": msg},
                )
                _audit(
                    "error",
                    "sdk_py.gen.failed",
                    {"error": msg, "cmd": " ".join(str(c) for c in cmd)},
                )
                return 1

            gen_files = list(_OUT.glob("**/*.py"))
            bytes_written = sum(f.stat().st_size for f in gen_files)
            span.add_event(
                "sdk_py.gen.done",
                {
                    "sdk_py.dest": str(_OUT),
                    "sdk_py.files": len(gen_files),
                    "sdk_py.bytes": bytes_written,
                },
            )
            _audit(
                "info",
                "sdk_py.gen.ok",
                {"dest": str(_OUT), "files": len(gen_files), "bytes": bytes_written},
            )
            print(f"sdk-py generated → {_OUT} ({len(gen_files)} files)")
            return 0

        except Exception as exc:
            msg = f"sdk-py generation failed: {exc}"
            span.set_status(Status(StatusCode.ERROR, msg))
            span.record_exception(exc)
            _audit(
                "error",
                "sdk_py.gen.failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            print(f"error: {msg}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
