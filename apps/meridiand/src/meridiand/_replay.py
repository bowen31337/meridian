from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter
from fastapi.responses import JSONResponse


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ReplayError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="replay_failed", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------


class FakeModelAdapter:
    """
    ModelProvider that plays back canned model responses from an NDJSON fixture file.

    Each line is a JSON array of event dicts representing one model call.
    """

    name = "fake"
    kind = "fake"

    def __init__(self, fixture_path: Path) -> None:
        self._calls: list[list[dict[str, Any]]] = []
        if fixture_path.exists():
            for raw in fixture_path.read_text().splitlines():
                line = raw.strip()
                if line:
                    self._calls.append(json.loads(line))
        self._index = 0

    async def call(self) -> AsyncIterator[dict[str, Any]]:
        if self._index < len(self._calls):
            events = self._calls[self._index]
            self._index += 1
            for event in events:
                yield event

    @property
    def call_count(self) -> int:
        return self._index


class FakeSandboxAdapter:
    """
    Tool dispatcher that plays back canned tool results from an NDJSON fixture file.

    Each line is a JSON object with at least a "content" field.
    """

    def __init__(self, fixture_path: Path) -> None:
        self._responses: list[dict[str, Any]] = []
        if fixture_path.exists():
            for raw in fixture_path.read_text().splitlines():
                line = raw.strip()
                if line:
                    self._responses.append(json.loads(line))
        self._index = 0

    def next_result(self) -> dict[str, Any]:
        if self._index < len(self._responses):
            result = self._responses[self._index]
            self._index += 1
            return result
        return {"content": ""}

    @property
    def dispatch_count(self) -> int:
        return self._index


# ---------------------------------------------------------------------------
# Harness loop
# ---------------------------------------------------------------------------


async def _run_harness(
    model_adapter: FakeModelAdapter,
    sandbox_adapter: FakeSandboxAdapter,
) -> tuple[int, int]:
    """Run the agent harness loop with fake adapters. Returns (model_calls, tool_calls)."""
    model_calls = 0
    tool_calls = 0

    while True:
        model_calls += 1
        tool_use_blocks: list[dict[str, Any]] = []
        current_tool: dict[str, Any] | None = None
        stop_reason = "end_turn"

        async for event in model_adapter.call():
            etype = event.get("type", "")
            if etype == "tool_use_start":
                current_tool = {
                    "id": event.get("id", ""),
                    "name": event.get("name", ""),
                    "input_json": "",
                }
                tool_use_blocks.append(current_tool)
            elif etype == "tool_input_delta" and current_tool is not None:
                current_tool["input_json"] += event.get("partial_json", "")
            elif etype == "message_stop":
                stop_reason = event.get("stop_reason") or "end_turn"

        if not tool_use_blocks or stop_reason != "tool_use":
            break

        for _block in tool_use_blocks:
            tool_calls += 1
            sandbox_adapter.next_result()

    return model_calls, tool_calls


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_replay_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/replay")
    async def replay_session(session_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        run_id = str(uuid.uuid4())
        fixture_dir = storage_root / "fixtures" / session_id

        with tracer.start_as_current_span(
            "replay.run",
            attributes={
                "session.id": session_id,
                "replay.run_id": run_id,
                "replay.fixture_dir": str(fixture_dir),
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="replay.run.invocation",
                    code="replay_run",
                    timestamp=now,
                ),
            )

            model_fixture = fixture_dir / "model_responses.ndjson"
            if not model_fixture.exists():
                err = ReplayError(
                    message=f"Fixture not found for session {session_id!r}: {model_fixture}",
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="replay.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            try:
                model_adapter = FakeModelAdapter(model_fixture)
                sandbox_adapter = FakeSandboxAdapter(fixture_dir / "tool_responses.ndjson")
                model_calls, tool_calls = await _run_harness(model_adapter, sandbox_adapter)
            except ReplayError:
                raise
            except Exception as exc:
                err = ReplayError(
                    message=f"Replay failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="replay.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "run_id": run_id,
                "session_id": session_id,
                "model_call_count": model_calls,
                "tool_call_count": tool_calls,
                "status": "completed",
            }
        )

    return router
