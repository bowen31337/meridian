"""Helper for the *subprocess* side of the stdin/stdout JSON protocol.

Tool authors writing a binary in any language that will be invoked as a
Meridian subprocess tool should implement the same protocol themselves.
For Python-based subprocess tools, this module provides a convenience
helper so the boilerplate is a one-liner::

    # my_tool.py  (the subprocess binary)
    from meridian_sdk_tool.subprocess_server import run_subprocess_tool

    def handle(args: dict, ctx: dict) -> dict:
        return {"result": args["x"] * 2}

    if __name__ == "__main__":
        run_subprocess_tool(handle)

Protocol (Architecture §11.2):
    stdin  → {"args": ..., "context": {...}}
    stdout ← {"result": ...} | {"error": {"code": "...", "message": "..."}}
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def run_subprocess_tool(
    handler: Callable[[dict[str, Any], dict[str, Any]], Any],
    *,
    input_stream: Any = None,
    output_stream: Any = None,
) -> None:
    """Run *handler* once inside the subprocess protocol loop.

    Reads one JSON request from stdin, calls *handler(args, context)*,
    and writes one JSON response to stdout.  Exits non-zero on hard
    failures (e.g. stdin is not valid JSON).

    *input_stream* and *output_stream* can be overridden in tests.
    """
    inp = input_stream or sys.stdin
    out = output_stream or sys.stdout

    try:
        raw = inp.read()
        request: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        _write_error(out, "invalid_request", f"stdin is not valid JSON: {exc}")
        sys.exit(1)
        return  # guard: sys.exit may be monkeypatched in tests

    args = request.get("args", {})
    ctx = request.get("context", {})

    try:
        result = handler(args, ctx)
        _write_result(out, result)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        _write_error(out, "execution_failed", str(exc), traceback=tb)
        sys.exit(1)


def _write_result(stream: Any, result: Any) -> None:
    stream.write(json.dumps({"result": result}, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


def _write_error(stream: Any, code: str, message: str, **extra: str) -> None:
    payload: dict[str, Any] = {"code": code, "message": message}
    payload.update(extra)
    stream.write(json.dumps({"error": payload}, separators=(",", ":")))
    stream.write("\n")
    stream.flush()
