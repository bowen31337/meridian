"""Test double for the Claude Code CLI --server subprocess.

Reads JSON lines from stdin, responds on stdout.  Understands:
  - {"type": "call",     "id": "...", "model": "...", ...}  → streams events + done
  - {"type": "ping",     "id": "..."}                       → {"type": "pong", "id": "..."}
  - {"type": "shutdown"}                                     → exit 0

Set env var CLI_STUB_HANG=1 to make call responses hang indefinitely (for
timeout / restart tests).

Set env var CLI_STUB_ERROR=1 to make call responses return an error line
instead of events.

Set env var CLI_STUB_NO_PONG=1 to silently ignore pings (no pong emitted),
simulating a subprocess that is alive but unresponsive to health checks.
"""

import json
import os
import sys
import time

_hang = os.environ.get("CLI_STUB_HANG") == "1"
_error = os.environ.get("CLI_STUB_ERROR") == "1"
_no_pong = os.environ.get("CLI_STUB_NO_PONG") == "1"


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        t = msg.get("type")

        if t == "shutdown":
            sys.exit(0)

        if t == "ping":
            if not _no_pong:
                _emit({"type": "pong", "id": msg.get("id", "")})
            # If _no_pong, silently drop the ping — health check will time out.
            continue

        if t == "call":
            call_id = msg.get("id", "")
            model = msg.get("model", "claude-test")

            if _hang:
                # Hang forever — caller must time out and kill us.
                time.sleep(9999)
                continue

            if _error:
                _emit({
                    "type": "error",
                    "call_id": call_id,
                    "code": "cli_test_error",
                    "message": "stub configured to return error",
                })
                continue

            _emit({
                "type": "message_start",
                "call_id": call_id,
                "model": model,
                "input_tokens": 10,
            })
            _emit({
                "type": "text_delta",
                "call_id": call_id,
                "text": "Hello from stub!",
            })
            _emit({
                "type": "message_stop",
                "call_id": call_id,
                "input_tokens": 10,
                "output_tokens": 4,
                "stop_reason": "end_turn",
            })
            _emit({"type": "done", "call_id": call_id})


if __name__ == "__main__":
    main()
