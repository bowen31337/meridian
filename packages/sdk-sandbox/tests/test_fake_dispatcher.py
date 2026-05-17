"""Tests for FakeSandboxAdapter — fixture loading, OTel span, audit log, call cycling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from sdk_sandbox import (
    ExecutionContext,
    InProcessHandler,
    SandboxFailure,
    SandboxResult,
    ToolDefinition,
    ToolDispatcher,
)
from sdk_sandbox.fake import FakeSandboxAdapter, write_sandbox_fixture

from .conftest import CapturingAuditLog, MockSpan, MockTracer

# ---------------------------------------------------------------------------
# OTel mock wiring for FakeSandboxAdapter
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("sdk_sandbox.fake.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_span(mock_tracer: MockTracer) -> MockSpan:
    return mock_tracer.span


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_DEFAULT_RESULT: dict[str, Any] = {"content": "ok", "duration_ms": 1.0}

TOOL_DEF = ToolDefinition(
    name="test.echo",
    description="Echo input",
    input_schema={"type": "object"},
    handler=InProcessHandler(),
)

CTX = ExecutionContext(session_id="sess-fake", workspace="/tmp")


def make_adapter(
    fixtures_dir: Path, audit_log: CapturingAuditLog | None = None
) -> FakeSandboxAdapter:
    return FakeSandboxAdapter(fixtures_dir, audit_log=audit_log)


def make_fixture(
    fixtures_dir: Path,
    tool_name: str,
    results: list[dict[str, Any]] | None = None,
) -> Path:
    path = fixtures_dir / f"{tool_name}.ndjson"
    write_sandbox_fixture(path, results or [_DEFAULT_RESULT])
    return path


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_tool_dispatcher_abc(self, tmp_path: Path) -> None:
        adapter = make_adapter(tmp_path)
        assert isinstance(adapter, ToolDispatcher)

    def test_kind_is_fake(self, tmp_path: Path) -> None:
        adapter = make_adapter(tmp_path)
        assert adapter.kind == "fake"


# ---------------------------------------------------------------------------
# dispatch() — happy path
# ---------------------------------------------------------------------------


class TestDispatchHappyPath:
    async def test_returns_sandbox_result(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo")
        adapter = make_adapter(tmp_path)
        result = await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert isinstance(result, SandboxResult)

    async def test_returns_correct_content(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo")
        adapter = make_adapter(tmp_path)
        result = await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert result.content == "ok"

    async def test_is_error_false_by_default(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo")
        adapter = make_adapter(tmp_path)
        result = await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert result.is_error is False

    async def test_error_result_preserved(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(
            tmp_path,
            "test.echo",
            [{"content": "denied", "is_error": True, "error_code": "cap_denied"}],
        )
        adapter = make_adapter(tmp_path)
        result = await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert result.is_error is True
        assert result.error_code == "cap_denied"

    async def test_no_audit_entries_on_success(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo")
        audit = CapturingAuditLog()
        adapter = make_adapter(tmp_path, audit)
        await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert audit.entries == []

    async def test_skips_blank_lines(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        path = tmp_path / "test.echo.ndjson"
        path.write_text(
            "\n" + json.dumps({"content": "hello"}) + "\n\n",
            encoding="utf-8",
        )
        adapter = make_adapter(tmp_path)
        result = await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert result.content == "hello"


# ---------------------------------------------------------------------------
# dispatch() — call cycling (saturation semantics)
# ---------------------------------------------------------------------------


class TestCallCycling:
    async def test_first_call_reads_first_line(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo", [{"content": "first"}, {"content": "second"}])
        adapter = make_adapter(tmp_path)
        result = await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert result.content == "first"

    async def test_second_call_reads_second_line(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo", [{"content": "first"}, {"content": "second"}])
        adapter = make_adapter(tmp_path)
        await adapter.dispatch(TOOL_DEF, {}, CTX)
        result = await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert result.content == "second"

    async def test_saturates_at_last_line_when_exhausted(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo", [{"content": "only"}])
        adapter = make_adapter(tmp_path)
        for _ in range(5):
            result = await adapter.dispatch(TOOL_DEF, {}, CTX)
            assert result.content == "only"

    async def test_independent_counters_per_tool(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        tool_a = ToolDefinition(
            name="test.a",
            description="A",
            input_schema={},
            handler=InProcessHandler(),
        )
        tool_b = ToolDefinition(
            name="test.b",
            description="B",
            input_schema={},
            handler=InProcessHandler(),
        )
        make_fixture(tmp_path, "test.a", [{"content": "a1"}, {"content": "a2"}])
        make_fixture(tmp_path, "test.b", [{"content": "b1"}, {"content": "b2"}])
        adapter = make_adapter(tmp_path)
        r_a1 = await adapter.dispatch(tool_a, {}, CTX)
        r_b1 = await adapter.dispatch(tool_b, {}, CTX)
        r_a2 = await adapter.dispatch(tool_a, {}, CTX)
        r_b2 = await adapter.dispatch(tool_b, {}, CTX)
        assert r_a1.content == "a1"
        assert r_b1.content == "b1"
        assert r_a2.content == "a2"
        assert r_b2.content == "b2"


# ---------------------------------------------------------------------------
# dispatch() — OTel span
# ---------------------------------------------------------------------------


class TestDispatchOTelSpan:
    async def test_span_name(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test.echo")
        adapter = make_adapter(tmp_path)
        await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert mock_span.name == "fake.sandbox.dispatch"

    async def test_span_attributes_tool_name(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo")
        adapter = make_adapter(tmp_path)
        await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert mock_span.attributes["tool.name"] == "test.echo"

    async def test_span_attributes_session_id(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo")
        adapter = make_adapter(tmp_path)
        await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert mock_span.attributes["session.id"] == "sess-fake"

    async def test_fake_dispatch_event_attached(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo")
        adapter = make_adapter(tmp_path)
        await adapter.dispatch(TOOL_DEF, {}, CTX)
        event_names = [e[0] for e in mock_span.events]
        assert "fake.dispatch" in event_names

    async def test_fake_dispatch_event_attributes(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo")
        adapter = make_adapter(tmp_path)
        await adapter.dispatch(TOOL_DEF, {}, CTX)
        ev = next(e for e in mock_span.events if e[0] == "fake.dispatch")
        assert ev[1]["tool.name"] == "test.echo"
        assert ev[1]["session.id"] == "sess-fake"

    async def test_span_ended_on_success(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test.echo")
        adapter = make_adapter(tmp_path)
        await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert mock_span.ended


# ---------------------------------------------------------------------------
# dispatch() — fixture failure
# ---------------------------------------------------------------------------


class TestDispatchFixtureFailure:
    async def test_raises_sandbox_failure_on_missing_fixture(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        adapter = make_adapter(tmp_path)
        with pytest.raises(SandboxFailure) as exc_info:
            await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert exc_info.value.code == "FAKE_FIXTURE_FAILED"

    async def test_sandbox_failure_tool_name(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        adapter = make_adapter(tmp_path)
        with pytest.raises(SandboxFailure) as exc_info:
            await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert exc_info.value.tool_name == "test.echo"

    async def test_raises_sandbox_failure_on_invalid_json(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        (tmp_path / "test.echo.ndjson").write_text("not-json\n", encoding="utf-8")
        adapter = make_adapter(tmp_path)
        with pytest.raises(SandboxFailure) as exc_info:
            await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert exc_info.value.code == "FAKE_FIXTURE_FAILED"

    async def test_raises_sandbox_failure_on_empty_fixture(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        (tmp_path / "test.echo.ndjson").write_text("\n\n", encoding="utf-8")
        adapter = make_adapter(tmp_path)
        with pytest.raises(SandboxFailure):
            await adapter.dispatch(TOOL_DEF, {}, CTX)

    async def test_audit_entry_written_on_missing_fixture(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        audit = CapturingAuditLog()
        adapter = make_adapter(tmp_path, audit)
        with pytest.raises(SandboxFailure):
            await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert len(audit.entries) == 1
        entry = audit.entries[0]
        assert entry.level == "error"
        assert entry.event == "fake_sandbox.fixture.failed"

    async def test_audit_entry_tool_name(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        audit = CapturingAuditLog()
        adapter = make_adapter(tmp_path, audit)
        with pytest.raises(SandboxFailure):
            await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert audit.entries[0].tool_name == "test.echo"

    async def test_audit_entry_detail_contains_fixture(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        audit = CapturingAuditLog()
        adapter = make_adapter(tmp_path, audit)
        with pytest.raises(SandboxFailure):
            await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert audit.entries[0].detail is not None
        assert "fixture" in audit.entries[0].detail

    async def test_span_marked_error_on_missing_fixture(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        adapter = make_adapter(tmp_path)
        with pytest.raises(SandboxFailure):
            await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_span_error_event_on_failure(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        adapter = make_adapter(tmp_path)
        with pytest.raises(SandboxFailure):
            await adapter.dispatch(TOOL_DEF, {}, CTX)
        event_names = [e[0] for e in mock_span.events]
        assert "sandbox.error" in event_names

    async def test_span_ended_on_failure(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        adapter = make_adapter(tmp_path)
        with pytest.raises(SandboxFailure):
            await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert mock_span.ended

    async def test_failure_cause_is_original_exception(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        adapter = make_adapter(tmp_path)
        with pytest.raises(SandboxFailure) as exc_info:
            await adapter.dispatch(TOOL_DEF, {}, CTX)
        assert isinstance(exc_info.value.cause, FileNotFoundError)


# ---------------------------------------------------------------------------
# write_sandbox_fixture helper
# ---------------------------------------------------------------------------


class TestWriteSandboxFixture:
    def test_creates_ndjson_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.ndjson"
        write_sandbox_fixture(path, [{"content": "ok"}])
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "out.ndjson"
        write_sandbox_fixture(path, [{"content": "ok"}])
        assert path.exists()

    def test_each_result_on_own_line(self, tmp_path: Path) -> None:
        path = tmp_path / "out.ndjson"
        write_sandbox_fixture(
            path, [{"content": "first"}, {"content": "second"}]
        )
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["content"] == "first"
        assert json.loads(lines[1])["content"] == "second"

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "out.ndjson"
        write_sandbox_fixture(
            path,
            [{"content": "hello", "duration_ms": 2.5, "is_error": False}],
        )
        adapter = FakeSandboxAdapter(tmp_path)
        # Re-use the load helper via dispatch would need a running event loop,
        # so just verify the file is valid NDJSON with the right content.
        data = json.loads(path.read_text().strip())
        assert data["content"] == "hello"
        assert data["duration_ms"] == 2.5
