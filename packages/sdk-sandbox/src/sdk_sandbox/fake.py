"""FakeSandboxAdapter — replay-based ToolDispatcher for deterministic testing.

Reads canned SandboxResult lines from fixtures/{tool.name}.ndjson. The n-th
call to dispatch() for a given tool name reads line n (0-based). When all
fixture lines are exhausted the last line is repeated (saturation semantics).

Emits a "fake.sandbox.dispatch" OTel child span with a "fake.dispatch"
invocation event on every dispatch() call. On fixture load failure the span
is marked ERROR, the audit log receives a "fake_sandbox.fixture.failed" entry,
and a SandboxFailure is raised so the Sandbox runtime surfaces it to the caller.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from ._audit import AuditLog, NoopAuditLog
from ._contract import ToolDispatcher
from ._telemetry import get_tracer, record_sandbox_failure
from ._types import AuditLogEntry, ExecutionContext, SandboxFailure, SandboxResult, ToolDefinition


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_sandbox_fixture(path: Path) -> list[SandboxResult]:
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    results: list[SandboxResult] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            data: dict[str, Any] = json.loads(line)
            results.append(SandboxResult(**data))
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: invalid SandboxResult: {exc}") from exc
    if not results:
        raise ValueError(f"Fixture is empty: {path}")
    return results


class FakeSandboxAdapter(ToolDispatcher):
    """ToolDispatcher that replays canned responses from fixtures/*.ndjson.

    Fixture file: ``{fixtures_dir}/{tool.name}.ndjson``
    Each non-blank line must be a JSON-serialized SandboxResult mapping.
    Calls advance through lines in order; the last line repeats when exhausted.

    Parameters
    ----------
    fixtures_dir:
        Directory containing fixture NDJSON files. Defaults to ``"fixtures"``
        relative to the process working directory.
    audit_log:
        Audit log sink. Defaults to ``NoopAuditLog``.
    """

    @property
    def kind(self) -> str:
        return "fake"

    def __init__(
        self,
        fixtures_dir: Path | str = "fixtures",
        *,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._fixtures_dir = Path(fixtures_dir)
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()
        self._call_counts: dict[str, int] = {}

    async def dispatch(
        self,
        tool: ToolDefinition,
        input: dict[str, Any],
        context: ExecutionContext,
    ) -> SandboxResult:
        fixture_path = self._fixtures_dir / f"{tool.name}.ndjson"
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "fake.sandbox.dispatch",
            attributes={"tool.name": tool.name, "session.id": context.session_id},
        ) as span:
            span.add_event(
                "fake.dispatch",
                {
                    "tool.name": tool.name,
                    "session.id": context.session_id,
                    "timestamp": now,
                },
            )
            call_index = self._call_counts.get(tool.name, 0)
            self._call_counts[tool.name] = call_index + 1

            try:
                rows = _load_sandbox_fixture(fixture_path)
            except Exception as exc:
                failure = SandboxFailure(
                    code="FAKE_FIXTURE_FAILED",
                    message=str(exc),
                    tool_name=tool.name,
                    session_id=context.session_id,
                    timestamp=now,
                    cause=exc,
                )
                record_sandbox_failure(span, failure)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="fake_sandbox.fixture.failed",
                        tool_name=tool.name,
                        session_id=context.session_id,
                        timestamp=now,
                        detail={"error": str(exc), "fixture": str(fixture_path)},
                    )
                )
                raise failure from exc

            row = rows[min(call_index, len(rows) - 1)]
            return row


def write_sandbox_fixture(path: Path, results: list[dict[str, Any]]) -> None:
    """Write a list of SandboxResult dicts to an NDJSON fixture file.

    Helper for test setup: creates parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in results) + "\n",
        encoding="utf-8",
    )
