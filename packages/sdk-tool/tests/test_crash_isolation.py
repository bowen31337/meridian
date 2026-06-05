"""Crash isolation tests for tool/hook/skill-forge subprocess failures.

Verifies that subprocess crashes:
  - do not propagate exceptions to the harness (returns ToolResult.err instead)
  - emit a ``tool_call.error`` OTel span event with structured attributes
  - attach stderr_tail to the event and error details on subprocess crash
  - write an audit log entry on every failure
  - surface a caller-readable error message

Architecture references: §11.4 (subprocess failure handling), §22.4 (audit log).
"""

from __future__ import annotations

import json
import sys
import textwrap
from typing import TYPE_CHECKING, Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from meridian_sdk_tool import ToolContext, subprocess_tool
from meridian_sdk_tool._otel import record_tool_call_error, record_tool_call_result
from meridian_sdk_tool._types import ToolResult
from meridian_sdk_tool.subprocess_tool import SubprocessCrashError

if TYPE_CHECKING:
    from pathlib import Path

_CTX = ToolContext(workspace="/workspace", session_id="sess_iso")


# ---------------------------------------------------------------------------
# OTel in-memory provider fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def otel_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Install an in-memory OTel TracerProvider; return its exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    import opentelemetry.trace as otel_trace

    monkeypatch.setattr(otel_trace, "get_tracer_provider", lambda: provider)

    # Patch the module-level get_tracer call used by sdk_tool's _otel.py

    def patched_get_tracer(*args: Any, **kwargs: Any) -> Any:
        return provider.get_tracer(*args, **kwargs)

    monkeypatch.setattr("meridian_sdk_tool._otel.trace.get_tracer", patched_get_tracer)
    monkeypatch.setattr(
        "meridian_sdk_tool._otel.trace.get_current_span",
        otel_trace.get_current_span,
    )
    return exporter


# ---------------------------------------------------------------------------
# SubprocessCrashError unit tests
# ---------------------------------------------------------------------------


class TestSubprocessCrashError:
    def test_is_runtime_error(self) -> None:
        assert issubclass(SubprocessCrashError, RuntimeError)

    def test_carries_stderr_tail(self) -> None:
        err = SubprocessCrashError("bad exit", stderr_tail="traceback here")
        assert err.stderr_tail == "traceback here"

    def test_default_stderr_tail_is_empty_string(self) -> None:
        err = SubprocessCrashError("bad exit")
        assert err.stderr_tail == ""

    def test_message_accessible_via_str(self) -> None:
        err = SubprocessCrashError("subprocess died")
        assert str(err) == "subprocess died"


# ---------------------------------------------------------------------------
# record_tool_call_error unit tests
# ---------------------------------------------------------------------------


class TestRecordToolCallError:
    def test_adds_tool_call_error_event(self, otel_exporter: InMemorySpanExporter) -> None:
        import opentelemetry.trace as otel_trace

        tracer = otel_trace.get_tracer("test")
        with tracer.start_as_current_span("test.span"):
            record_tool_call_error("execution_failed", "boom")

        spans = otel_exporter.get_finished_spans()
        assert len(spans) == 1
        event_names = [e.name for e in spans[0].events]
        assert "tool_call.error" in event_names

    def test_event_contains_error_code(self, otel_exporter: InMemorySpanExporter) -> None:
        import opentelemetry.trace as otel_trace

        tracer = otel_trace.get_tracer("test")
        with tracer.start_as_current_span("test.span"):
            record_tool_call_error("my_code", "some message")

        spans = otel_exporter.get_finished_spans()
        event = next(e for e in spans[0].events if e.name == "tool_call.error")
        assert event.attributes["error.code"] == "my_code"

    def test_event_contains_error_message(self, otel_exporter: InMemorySpanExporter) -> None:
        import opentelemetry.trace as otel_trace

        tracer = otel_trace.get_tracer("test")
        with tracer.start_as_current_span("test.span"):
            record_tool_call_error("my_code", "some message")

        spans = otel_exporter.get_finished_spans()
        event = next(e for e in spans[0].events if e.name == "tool_call.error")
        assert event.attributes["error.message"] == "some message"

    def test_event_contains_stderr_tail_when_provided(
        self, otel_exporter: InMemorySpanExporter
    ) -> None:
        import opentelemetry.trace as otel_trace

        tracer = otel_trace.get_tracer("test")
        with tracer.start_as_current_span("test.span"):
            record_tool_call_error("execution_failed", "crash", stderr_tail="last line")

        spans = otel_exporter.get_finished_spans()
        event = next(e for e in spans[0].events if e.name == "tool_call.error")
        assert event.attributes["subprocess.stderr_tail"] == "last line"

    def test_event_omits_stderr_tail_when_absent(self, otel_exporter: InMemorySpanExporter) -> None:
        import opentelemetry.trace as otel_trace

        tracer = otel_trace.get_tracer("test")
        with tracer.start_as_current_span("test.span"):
            record_tool_call_error("execution_failed", "crash")

        spans = otel_exporter.get_finished_spans()
        event = next(e for e in spans[0].events if e.name == "tool_call.error")
        assert "subprocess.stderr_tail" not in event.attributes

    def test_span_status_set_to_error(self, otel_exporter: InMemorySpanExporter) -> None:
        import opentelemetry.trace as otel_trace

        tracer = otel_trace.get_tracer("test")
        with tracer.start_as_current_span("test.span"):
            record_tool_call_error("execution_failed", "crash")

        spans = otel_exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR

    def test_no_active_span_does_not_raise(self) -> None:
        # Outside any span context — must be a no-op, not raise
        record_tool_call_error("execution_failed", "crash", stderr_tail="oops")


# ---------------------------------------------------------------------------
# Subprocess crash isolation — integration tests (real subprocesses)
# ---------------------------------------------------------------------------


def _write_crashing_script(tmp_path: Path, *, exit_code: int = 1) -> Path:
    script = tmp_path / "crash_tool.py"
    script.write_text(
        textwrap.dedent(f"""\
            #!{sys.executable}
            import sys
            sys.stderr.write("fatal error in subprocess\\n")
            sys.stderr.flush()
            sys.exit({exit_code})
        """)
    )
    script.chmod(0o755)
    return script


def _write_invalid_json_script(tmp_path: Path) -> Path:
    script = tmp_path / "bad_json_tool.py"
    script.write_text(
        textwrap.dedent(f"""\
            #!{sys.executable}
            import sys
            sys.stderr.write("JSON encoding failed\\n")
            sys.stdout.write("NOT JSON OUTPUT")
            sys.stdout.flush()
        """)
    )
    script.chmod(0o755)
    return script


class TestSubprocessCrashIsolation:
    @pytest.mark.anyio
    async def test_nonzero_exit_returns_is_error(self, tmp_path: Path) -> None:
        script = _write_crashing_script(tmp_path)
        tool = subprocess_tool(
            name="crash",
            description="crashes",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        result = await tool.execute({}, _CTX)
        assert result.is_error

    @pytest.mark.anyio
    async def test_nonzero_exit_does_not_raise(self, tmp_path: Path) -> None:
        script = _write_crashing_script(tmp_path)
        tool = subprocess_tool(
            name="crash",
            description="crashes",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )
        # Must return, not raise — harness stability guarantee
        result = await tool.execute({}, _CTX)
        assert isinstance(result, ToolResult)

    @pytest.mark.anyio
    async def test_stderr_tail_in_error_details(self, tmp_path: Path) -> None:
        script = _write_crashing_script(tmp_path)
        tool = subprocess_tool(
            name="crash",
            description="crashes",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        result = await tool.execute({}, _CTX)
        assert result.is_error
        assert result.error is not None
        assert "stderr_tail" in result.error.details
        assert "fatal error in subprocess" in result.error.details["stderr_tail"]

    @pytest.mark.anyio
    async def test_invalid_json_output_returns_is_error(self, tmp_path: Path) -> None:
        script = _write_invalid_json_script(tmp_path)
        tool = subprocess_tool(
            name="bad_json",
            description="writes bad json",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        result = await tool.execute({}, _CTX)
        assert result.is_error

    @pytest.mark.anyio
    async def test_invalid_json_stderr_tail_in_details(self, tmp_path: Path) -> None:
        script = _write_invalid_json_script(tmp_path)
        tool = subprocess_tool(
            name="bad_json",
            description="writes bad json",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        result = await tool.execute({}, _CTX)
        assert result.is_error
        assert result.error is not None
        assert "stderr_tail" in result.error.details

    @pytest.mark.anyio
    async def test_error_message_surfaced_to_caller(self, tmp_path: Path) -> None:
        script = _write_crashing_script(tmp_path)
        tool = subprocess_tool(
            name="crash",
            description="crashes",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        result = await tool.execute({}, _CTX)
        assert result.error is not None
        assert result.error.message  # non-empty message surfaced to caller

    @pytest.mark.anyio
    async def test_crash_writes_audit_log(self, tmp_path: Path) -> None:
        script = _write_crashing_script(tmp_path)
        audit_path = tmp_path / "audit.ndjson"
        tool = subprocess_tool(
            name="crash",
            description="crashes",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
            audit_log_path=str(audit_path),
        )

        await tool.execute({}, _CTX)
        assert audit_path.exists()
        record = json.loads(audit_path.read_text().strip())
        assert record["type"] == "tool.execution_failed"
        assert record["tool_name"] == "crash"

    @pytest.mark.anyio
    async def test_crash_emits_tool_call_error_otel_event(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        script = _write_crashing_script(tmp_path)
        tool = subprocess_tool(
            name="crash",
            description="crashes",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        await tool.execute({}, _CTX)

        spans = otel_exporter.get_finished_spans()
        assert spans, "expected at least one OTel span"
        tool_span = next((s for s in spans if s.name == "tool.call"), None)
        assert tool_span is not None, "expected a 'tool.call' span"
        event_names = [e.name for e in tool_span.events]
        assert "tool_call.error" in event_names

    @pytest.mark.anyio
    async def test_crash_otel_event_has_stderr_tail(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        script = _write_crashing_script(tmp_path)
        tool = subprocess_tool(
            name="crash",
            description="crashes",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        await tool.execute({}, _CTX)

        spans = otel_exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool.call")
        event = next(e for e in tool_span.events if e.name == "tool_call.error")
        assert "subprocess.stderr_tail" in event.attributes
        assert "fatal error in subprocess" in event.attributes["subprocess.stderr_tail"]

    @pytest.mark.anyio
    async def test_crash_otel_span_status_is_error(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        script = _write_crashing_script(tmp_path)
        tool = subprocess_tool(
            name="crash",
            description="crashes",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        await tool.execute({}, _CTX)

        spans = otel_exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool.call")
        assert tool_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# record_tool_call_result unit tests
# ---------------------------------------------------------------------------


class TestRecordToolCallResult:
    def test_adds_tool_call_result_event(self, otel_exporter: InMemorySpanExporter) -> None:
        import opentelemetry.trace as otel_trace

        tracer = otel_trace.get_tracer("test")
        with tracer.start_as_current_span("test.span"):
            record_tool_call_result()

        spans = otel_exporter.get_finished_spans()
        assert len(spans) == 1
        event_names = [e.name for e in spans[0].events]
        assert "tool_call.result" in event_names

    def test_event_contains_stderr_tail_when_provided(
        self, otel_exporter: InMemorySpanExporter
    ) -> None:
        import opentelemetry.trace as otel_trace

        tracer = otel_trace.get_tracer("test")
        with tracer.start_as_current_span("test.span"):
            record_tool_call_result(stderr_tail="some debug output")

        spans = otel_exporter.get_finished_spans()
        event = next(e for e in spans[0].events if e.name == "tool_call.result")
        assert event.attributes["subprocess.stderr_tail"] == "some debug output"

    def test_event_omits_stderr_tail_when_absent(self, otel_exporter: InMemorySpanExporter) -> None:
        import opentelemetry.trace as otel_trace

        tracer = otel_trace.get_tracer("test")
        with tracer.start_as_current_span("test.span"):
            record_tool_call_result()

        spans = otel_exporter.get_finished_spans()
        event = next(e for e in spans[0].events if e.name == "tool_call.result")
        assert "subprocess.stderr_tail" not in event.attributes

    def test_no_active_span_does_not_raise(self) -> None:
        record_tool_call_result(stderr_tail="some output")


# ---------------------------------------------------------------------------
# Success-path stderr capture — integration tests (real subprocesses)
# ---------------------------------------------------------------------------


def _write_noisy_success_script(tmp_path: Path) -> Path:
    """Write a subprocess that writes to stderr but exits successfully."""
    script = tmp_path / "noisy_tool.py"
    script.write_text(
        textwrap.dedent(f"""\
            #!{sys.executable}
            import json, sys
            sys.stderr.write("debug output from subprocess\\n")
            sys.stderr.flush()
            _ = sys.stdin.read()
            sys.stdout.write(json.dumps({{"result": "ok"}}))
            sys.stdout.flush()
        """)
    )
    script.chmod(0o755)
    return script


class TestSubprocessSuccessStderr:
    @pytest.mark.anyio
    async def test_success_emits_tool_call_result_otel_event(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        script = _write_noisy_success_script(tmp_path)
        tool = subprocess_tool(
            name="noisy",
            description="writes stderr on success",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        result = await tool.execute({}, _CTX)
        assert not result.is_error, result.error

        spans = otel_exporter.get_finished_spans()
        assert spans, "expected at least one OTel span"
        found = next((s for s in spans if s.name == "tool.call"), None)
        assert found is not None
        event_names = [e.name for e in found.events]
        assert "tool_call.result" in event_names

    @pytest.mark.anyio
    async def test_success_otel_event_has_stderr_tail(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        script = _write_noisy_success_script(tmp_path)
        tool = subprocess_tool(
            name="noisy",
            description="writes stderr on success",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        await tool.execute({}, _CTX)

        spans = otel_exporter.get_finished_spans()
        found = next(s for s in spans if s.name == "tool.call")
        event = next(e for e in found.events if e.name == "tool_call.result")
        assert "subprocess.stderr_tail" in event.attributes
        assert "debug output from subprocess" in event.attributes["subprocess.stderr_tail"]

    @pytest.mark.anyio
    async def test_success_with_empty_stderr_emits_event_without_stderr_tail(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        script = tmp_path / "quiet_tool.py"
        script.write_text(
            textwrap.dedent(f"""\
                #!{sys.executable}
                import json, sys
                _ = sys.stdin.read()
                sys.stdout.write(json.dumps({{"result": "ok"}}))
                sys.stdout.flush()
            """)
        )
        script.chmod(0o755)
        tool = subprocess_tool(
            name="quiet",
            description="no stderr",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        result = await tool.execute({}, _CTX)
        assert not result.is_error, result.error

        spans = otel_exporter.get_finished_spans()
        found = next(s for s in spans if s.name == "tool.call")
        event = next((e for e in found.events if e.name == "tool_call.result"), None)
        assert event is not None, "tool_call.result event should always be emitted on success"
        assert "subprocess.stderr_tail" not in event.attributes

    @pytest.mark.anyio
    async def test_application_error_response_has_stderr_in_details(self, tmp_path: Path) -> None:
        """Subprocess returning {"error": ...} includes stderr_tail in error details."""
        script = tmp_path / "app_error_tool.py"
        script.write_text(
            textwrap.dedent(f"""\
                #!{sys.executable}
                import json, sys
                sys.stderr.write("app error context\\n")
                sys.stderr.flush()
                _ = sys.stdin.read()
                sys.stdout.write(
                    json.dumps({{"error": {{"code": "not_found", "message": "item missing"}}}})
                )
                sys.stdout.flush()
            """)
        )
        script.chmod(0o755)
        tool = subprocess_tool(
            name="app_err",
            description="returns structured error",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
        )

        result = await tool.execute({}, _CTX)
        assert result.is_error
        assert result.error is not None
        assert "stderr_tail" in result.error.details
        assert "app error context" in result.error.details["stderr_tail"]

    @pytest.mark.anyio
    async def test_application_error_response_writes_audit_log(self, tmp_path: Path) -> None:
        """Subprocess returning {"error": ...} writes to the audit log."""
        script = tmp_path / "app_error_tool2.py"
        script.write_text(
            textwrap.dedent(f"""\
                #!{sys.executable}
                import json, sys
                _ = sys.stdin.read()
                sys.stdout.write(
                    json.dumps({{"error": {{"code": "bad_state", "message": "oops"}}}})
                )
                sys.stdout.flush()
            """)
        )
        script.chmod(0o755)
        audit_path = tmp_path / "audit.ndjson"
        tool = subprocess_tool(
            name="app_err2",
            description="returns structured error",
            path=str(script),
            input_schema={"type": "object"},
            timeout_ms=5_000,
            audit_log_path=str(audit_path),
        )

        await tool.execute({}, _CTX)
        assert audit_path.exists()
        record = json.loads(audit_path.read_text().strip())
        assert record["tool_name"] == "app_err2"
        assert "execution_failed" in record["type"]
