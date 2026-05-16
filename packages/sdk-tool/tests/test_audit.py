"""Tests for the audit log writer (Architecture §22.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meridian_sdk_tool._audit import write_audit_event


def test_audit_event_written_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    write_audit_event(
        event_type="tool.execution_failed",
        tool_name="my_tool",
        session_id="sess_abc",
        error={"code": "execution_failed", "message": "boom"},
        audit_log_path=str(log_path),
    )

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "tool.execution_failed"
    assert record["tool_name"] == "my_tool"
    assert record["session_id"] == "sess_abc"
    assert record["error"]["code"] == "execution_failed"


def test_multiple_events_appended(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    for i in range(3):
        write_audit_event(
            event_type="tool.input_validation_failed",
            tool_name=f"tool_{i}",
            audit_log_path=str(log_path),
        )

    lines = log_path.read_text().splitlines()
    assert len(lines) == 3


def test_no_session_id_is_omitted(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.ndjson"
    write_audit_event(
        event_type="tool.execution_failed",
        tool_name="t",
        audit_log_path=str(log_path),
    )
    record = json.loads(log_path.read_text().strip())
    assert "session_id" not in record


def test_parent_dir_created_automatically(tmp_path: Path) -> None:
    log_path = tmp_path / "deep" / "nested" / "audit.ndjson"
    write_audit_event(
        event_type="tool.execution_failed",
        tool_name="t",
        audit_log_path=str(log_path),
    )
    assert log_path.exists()


def test_write_failure_does_not_propagate() -> None:
    # Passing a path in a directory that can't be created should not raise.
    write_audit_event(
        event_type="tool.execution_failed",
        tool_name="t",
        audit_log_path="/dev/null/impossible/path/audit.ndjson",
    )
