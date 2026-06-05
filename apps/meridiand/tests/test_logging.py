"""
JSON structured logging conformance suite.

Tests cover:
  - JsonFormatter output is valid JSON.
  - ts field is present and ISO 8601.
  - level maps correctly for debug, info, warning, error, critical.
  - component maps from logger name.
  - msg field carries the formatted message.
  - session_id absent when not set in extra.
  - session_id present when set in extra.
  - agent_id absent when not set in extra.
  - agent_id present when set in extra.
  - tool_name absent when not set in extra.
  - tool_name present when set in extra.
  - provider absent when not set in extra.
  - provider present when set in extra.
  - exc field present when record has exc_info.
  - exc field absent when no exc_info.
  - Output contains no domain event log fields (seq, type, data, thread_id).
  - Output is single-line (no embedded newlines).
  - configure_json_logging installs exactly one handler on root logger.
  - configure_json_logging handler uses JsonFormatter.
  - configure_json_logging clears pre-existing root handlers.
  - configure_json_logging sets root log level.
  - configure_json_logging level arg is case-insensitive.
  - configure_json_logging writes JSON to stderr.
  - configure_json_logging output does not appear on stdout.
  - configure_json_logging raises LoggingConfigError on handler creation failure.
  - configure_json_logging writes audit entry on failure.
  - configure_json_logging audit entry level is error.
  - configure_json_logging audit entry event is logging.configure.failed.
  - emit_early_error writes JSON to stderr.
  - emit_early_error does not write to stdout.
  - emit_early_error output is valid JSON.
  - emit_early_error has ts field.
  - emit_early_error level is always error.
  - emit_early_error component field matches argument.
  - emit_early_error msg field matches argument.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import sys
from typing import Any

from core_errors import AuditLog, AuditLogEntry
from meridiand._logging import (
    JsonFormatter,
    LoggingConfigError,
    configure_json_logging,
    emit_early_error,
)
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


def _make_record(
    name: str = "meridiand",
    level: int = logging.INFO,
    msg: str = "hello",
    **extra: Any,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, val in extra.items():
        setattr(record, key, val)
    return record


def _parse(record: logging.LogRecord) -> dict[str, Any]:
    return json.loads(JsonFormatter().format(record))


# ---------------------------------------------------------------------------
# TestJsonFormatter
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def test_output_is_valid_json(self) -> None:
        result = JsonFormatter().format(_make_record())
        assert isinstance(json.loads(result), dict)

    def test_output_is_single_line(self) -> None:
        result = JsonFormatter().format(_make_record())
        assert "\n" not in result

    def test_has_ts_field(self) -> None:
        assert "ts" in _parse(_make_record())

    def test_ts_is_iso8601(self) -> None:
        ts = _parse(_make_record())["ts"]
        datetime.fromisoformat(ts)

    def test_ts_has_utc_offset(self) -> None:
        ts = _parse(_make_record())["ts"]
        assert "+" in ts or ts.endswith("Z") or ts.endswith("+00:00")

    def test_level_info(self) -> None:
        assert _parse(_make_record(level=logging.INFO))["level"] == "info"

    def test_level_debug(self) -> None:
        assert _parse(_make_record(level=logging.DEBUG))["level"] == "debug"

    def test_level_warning(self) -> None:
        assert _parse(_make_record(level=logging.WARNING))["level"] == "warning"

    def test_level_error(self) -> None:
        assert _parse(_make_record(level=logging.ERROR))["level"] == "error"

    def test_level_critical(self) -> None:
        assert _parse(_make_record(level=logging.CRITICAL))["level"] == "critical"

    def test_component_from_logger_name(self) -> None:
        assert _parse(_make_record(name="meridiand._app"))["component"] == "meridiand._app"

    def test_component_uvicorn(self) -> None:
        assert _parse(_make_record(name="uvicorn"))["component"] == "uvicorn"

    def test_msg_field(self) -> None:
        assert _parse(_make_record(msg="hello world"))["msg"] == "hello world"

    # Optional context fields – absent when not set

    def test_session_id_absent_when_not_set(self) -> None:
        assert "session_id" not in _parse(_make_record())

    def test_agent_id_absent_when_not_set(self) -> None:
        assert "agent_id" not in _parse(_make_record())

    def test_tool_name_absent_when_not_set(self) -> None:
        assert "tool_name" not in _parse(_make_record())

    def test_provider_absent_when_not_set(self) -> None:
        assert "provider" not in _parse(_make_record())

    # Optional context fields – present when set via extra

    def test_session_id_present_when_set(self) -> None:
        record = _make_record(session_id="sess_abc")
        assert _parse(record)["session_id"] == "sess_abc"

    def test_agent_id_present_when_set(self) -> None:
        record = _make_record(agent_id="agent_xyz")
        assert _parse(record)["agent_id"] == "agent_xyz"

    def test_tool_name_present_when_set(self) -> None:
        record = _make_record(tool_name="bash")
        assert _parse(record)["tool_name"] == "bash"

    def test_provider_present_when_set(self) -> None:
        record = _make_record(provider="anthropic")
        assert _parse(record)["provider"] == "anthropic"

    def test_all_context_fields_together(self) -> None:
        record = _make_record(
            session_id="sess_1",
            agent_id="agent_2",
            tool_name="grep",
            provider="openai",
        )
        parsed = _parse(record)
        assert parsed["session_id"] == "sess_1"
        assert parsed["agent_id"] == "agent_2"
        assert parsed["tool_name"] == "grep"
        assert parsed["provider"] == "openai"

    # exc field

    def test_exc_absent_when_no_exc_info(self) -> None:
        assert "exc" not in _parse(_make_record())

    def test_exc_present_when_exc_info(self) -> None:
        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()
        record = _make_record(level=logging.ERROR)
        record.exc_info = exc_info
        assert "exc" in _parse(record)

    def test_exc_contains_exception_type(self) -> None:
        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()
        record = _make_record(level=logging.ERROR)
        record.exc_info = exc_info
        assert "ValueError" in _parse(record)["exc"]

    # No domain event log fields

    def test_no_seq_field(self) -> None:
        assert "seq" not in _parse(_make_record())

    def test_no_type_field(self) -> None:
        assert "type" not in _parse(_make_record())

    def test_no_data_field(self) -> None:
        assert "data" not in _parse(_make_record())

    def test_no_thread_id_field(self) -> None:
        assert "thread_id" not in _parse(_make_record())


# ---------------------------------------------------------------------------
# TestConfigureJsonLogging
# ---------------------------------------------------------------------------


class TestConfigureJsonLogging:
    @pytest.fixture(autouse=True)
    def _save_restore_root_logger(self) -> Any:
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        yield
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)

    def test_installs_one_handler(self) -> None:
        configure_json_logging("info")
        assert len(logging.getLogger().handlers) == 1

    def test_handler_is_stream_handler(self) -> None:
        configure_json_logging("info")
        assert isinstance(logging.getLogger().handlers[0], logging.StreamHandler)

    def test_handler_uses_json_formatter(self) -> None:
        configure_json_logging("info")
        assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)

    def test_clears_existing_handlers(self) -> None:
        root = logging.getLogger()
        root.addHandler(logging.NullHandler())
        root.addHandler(logging.NullHandler())
        configure_json_logging("info")
        assert len(root.handlers) == 1

    def test_sets_level_info(self) -> None:
        configure_json_logging("info")
        assert logging.getLogger().level == logging.INFO

    def test_sets_level_debug(self) -> None:
        configure_json_logging("debug")
        assert logging.getLogger().level == logging.DEBUG

    def test_sets_level_warning(self) -> None:
        configure_json_logging("warning")
        assert logging.getLogger().level == logging.WARNING

    def test_sets_level_error(self) -> None:
        configure_json_logging("error")
        assert logging.getLogger().level == logging.ERROR

    def test_level_arg_case_insensitive(self) -> None:
        configure_json_logging("INFO")
        assert logging.getLogger().level == logging.INFO

    def test_writes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_json_logging("info")
        logging.getLogger("meridiand").info("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.err

    def test_output_not_on_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_json_logging("info")
        logging.getLogger("meridiand").info("test message")
        captured = capsys.readouterr()
        assert "test message" not in captured.out

    def test_output_is_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_json_logging("info")
        logging.getLogger("meridiand").info("test message")
        captured = capsys.readouterr()
        for line in captured.err.strip().splitlines():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_output_has_required_fields(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_json_logging("info")
        logging.getLogger("meridiand").info("test message")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip().splitlines()[-1])
        assert "ts" in parsed
        assert "level" in parsed
        assert "component" in parsed
        assert "msg" in parsed

    def test_context_fields_via_extra(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_json_logging("info")
        logging.getLogger("meridiand").info(
            "tool ran",
            extra={"session_id": "sess_1", "tool_name": "bash"},
        )
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip().splitlines()[-1])
        assert parsed["session_id"] == "sess_1"
        assert parsed["tool_name"] == "bash"

    def test_raises_logging_config_error_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _bad(*args: Any, **kwargs: Any) -> None:
            raise OSError("cannot create handler")

        monkeypatch.setattr(logging, "StreamHandler", _bad)
        with pytest.raises(LoggingConfigError):
            configure_json_logging("info")

    def test_failure_writes_audit_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _bad(*args: Any, **kwargs: Any) -> None:
            raise OSError("cannot create handler")

        monkeypatch.setattr(logging, "StreamHandler", _bad)
        audit = _CapturingAuditLog()
        with pytest.raises(LoggingConfigError):
            configure_json_logging("info", audit_log=audit)
        assert any(e.code == "logging_config_failed" for e in audit.entries)

    def test_failure_audit_entry_level_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _bad(*args: Any, **kwargs: Any) -> None:
            raise OSError("cannot create handler")

        monkeypatch.setattr(logging, "StreamHandler", _bad)
        audit = _CapturingAuditLog()
        with pytest.raises(LoggingConfigError):
            configure_json_logging("info", audit_log=audit)
        entry = next(e for e in audit.entries if e.code == "logging_config_failed")
        assert entry.level == "error"

    def test_failure_audit_entry_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _bad(*args: Any, **kwargs: Any) -> None:
            raise OSError("cannot create handler")

        monkeypatch.setattr(logging, "StreamHandler", _bad)
        audit = _CapturingAuditLog()
        with pytest.raises(LoggingConfigError):
            configure_json_logging("info", audit_log=audit)
        entry = next(e for e in audit.entries if e.code == "logging_config_failed")
        assert entry.event == "logging.configure.failed"

    def test_failure_audit_detail_has_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _bad(*args: Any, **kwargs: Any) -> None:
            raise OSError("cannot create handler")

        monkeypatch.setattr(logging, "StreamHandler", _bad)
        audit = _CapturingAuditLog()
        with pytest.raises(LoggingConfigError):
            configure_json_logging("info", audit_log=audit)
        entry = next(e for e in audit.entries if e.code == "logging_config_failed")
        assert entry.detail is not None
        assert "cannot create handler" in entry.detail["message"]

    def test_failure_without_audit_log_raises_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _bad(*args: Any, **kwargs: Any) -> None:
            raise OSError("cannot create handler")

        monkeypatch.setattr(logging, "StreamHandler", _bad)
        with pytest.raises(LoggingConfigError):
            configure_json_logging("info", audit_log=None)

    def test_application_log_distinct_from_domain_event_log(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_json_logging("info")
        logging.getLogger("meridiand").info("app event")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip().splitlines()[-1])
        for domain_field in ("seq", "type", "data", "thread_id"):
            assert domain_field not in parsed


# ---------------------------------------------------------------------------
# TestEmitEarlyError
# ---------------------------------------------------------------------------


class TestEmitEarlyError:
    def test_writes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_early_error("meridiand", "something failed")
        captured = capsys.readouterr()
        assert "something failed" in captured.err

    def test_does_not_write_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_early_error("meridiand", "something failed")
        captured = capsys.readouterr()
        assert "something failed" not in captured.out

    def test_output_is_valid_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_early_error("meridiand", "something failed")
        captured = capsys.readouterr()
        for line in captured.err.strip().splitlines():
            assert isinstance(json.loads(line), dict)

    def test_has_ts_field(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_early_error("meridiand", "something failed")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert "ts" in parsed

    def test_ts_is_iso8601(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_early_error("meridiand", "something failed")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        datetime.fromisoformat(parsed["ts"])

    def test_level_is_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_early_error("meridiand", "something failed")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert parsed["level"] == "error"

    def test_component_field(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_early_error("my.component", "something failed")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert parsed["component"] == "my.component"

    def test_msg_field(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_early_error("meridiand", "config not found")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert parsed["msg"] == "config not found"

    def test_single_line_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_early_error("meridiand", "something failed")
        captured = capsys.readouterr()
        lines = [line for line in captured.err.splitlines() if line.strip()]
        assert len(lines) == 1
