"""
System prompt template expansion conformance suite.

Tests cover:
  - Template with no {{ memory.* }} refs returns the original string unchanged.
  - Single {{ memory.KEY }} reference is replaced with the memory value.
  - Multiple distinct references are all expanded.
  - A reference that appears twice is resolved once and substituted at both sites.
  - Whitespace inside {{ }} delimiters is tolerated (e.g. {{  memory.key  }}).
  - Missing memory key raises TemplateMemoryNotFoundError and writes error audit entry.
  - Missing memory key audit entry has event "system_prompt.template.expand.failed".
  - Missing memory key audit entry has code "template_memory_not_found".
  - Missing memory key audit detail contains message field.
  - Unexpected I/O error raises TemplateExpandError and writes error audit entry.
  - Unexpected error audit entry has event "system_prompt.template.expand.failed".
  - Unexpected error audit entry has code "template_expand_failed".
  - OTel span "system_prompt.template.expand" is emitted on success.
  - OTel span is emitted on failure.
  - Failure span has ERROR status.
  - Span attribute template.ref_count equals zero when there are no refs.
  - Span attribute template.ref_count equals the number of distinct placeholder sites.
  - Audit entry level is "error" on failure.
  - Memory value from file is used verbatim in substitution.
  - Dotted keys like user.preferences.commit_style are handled correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from meridiand._audit import FileAuditLog
from meridiand._system_prompt_template import (
    TemplateExpandError,
    TemplateMemoryNotFoundError,
    expand_system_prompt,
)

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_memory(storage_root: Path, key: str, value: str) -> None:
    mem_dir = storage_root / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    safe_key = key.replace("/", "_").replace("\x00", "_")
    (mem_dir / f"{safe_key}.json").write_text(
        json.dumps({"key": key, "value": value, "type": "text"})
    )


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# No template references
# ---------------------------------------------------------------------------


class TestNoRefs:
    def test_plain_string_returned_unchanged(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        result = expand_system_prompt(
            "You are a helpful assistant.", storage_root=storage_root, audit_log=audit
        )
        assert result == "You are a helpful assistant."

    def test_empty_string_returned_unchanged(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        result = expand_system_prompt("", storage_root=storage_root, audit_log=audit)
        assert result == ""

    def test_no_audit_entry_when_no_refs(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        expand_system_prompt("plain text", storage_root=storage_root, audit_log=audit)
        assert _audit_records(storage_root) == []


# ---------------------------------------------------------------------------
# Successful expansion
# ---------------------------------------------------------------------------


class TestSuccessfulExpansion:
    def test_single_ref_replaced(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.commit_style", "conventional-commits")
        audit = FileAuditLog(storage_root)
        result = expand_system_prompt(
            "Use {{ memory.user.commit_style }} format.",
            storage_root=storage_root,
            audit_log=audit,
        )
        assert result == "Use conventional-commits format."

    def test_multiple_distinct_refs_replaced(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.name", "Alice")
        _write_memory(storage_root, "user.lang", "Python")
        audit = FileAuditLog(storage_root)
        result = expand_system_prompt(
            "Hello {{ memory.user.name }}, write in {{ memory.user.lang }}.",
            storage_root=storage_root,
            audit_log=audit,
        )
        assert result == "Hello Alice, write in Python."

    def test_duplicate_ref_resolved_once_substituted_twice(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.tone", "friendly")
        audit = FileAuditLog(storage_root)
        result = expand_system_prompt(
            "Be {{ memory.user.tone }} and {{ memory.user.tone }}.",
            storage_root=storage_root,
            audit_log=audit,
        )
        assert result == "Be friendly and friendly."

    def test_whitespace_inside_delimiters_tolerated(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.style", "terse")
        audit = FileAuditLog(storage_root)
        result = expand_system_prompt(
            "Style: {{  memory.user.style  }}.",
            storage_root=storage_root,
            audit_log=audit,
        )
        assert result == "Style: terse."

    def test_dotted_key_user_preferences_commit_style(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.preferences.commit_style", "conventional-commits")
        audit = FileAuditLog(storage_root)
        result = expand_system_prompt(
            "Commit style: {{ memory.user.preferences.commit_style }}.",
            storage_root=storage_root,
            audit_log=audit,
        )
        assert result == "Commit style: conventional-commits."

    def test_memory_value_used_verbatim(self, storage_root: Path) -> None:
        _write_memory(storage_root, "agent.greeting", "Hello, world!\nLine 2.")
        audit = FileAuditLog(storage_root)
        result = expand_system_prompt(
            "{{ memory.agent.greeting }}",
            storage_root=storage_root,
            audit_log=audit,
        )
        assert result == "Hello, world!\nLine 2."

    def test_no_audit_entry_on_success(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.x", "y")
        audit = FileAuditLog(storage_root)
        expand_system_prompt(
            "{{ memory.user.x }}", storage_root=storage_root, audit_log=audit
        )
        assert _audit_records(storage_root) == []


# ---------------------------------------------------------------------------
# Missing memory key
# ---------------------------------------------------------------------------


class TestMissingMemory:
    def test_missing_key_raises_template_memory_not_found(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        with pytest.raises(TemplateMemoryNotFoundError):
            expand_system_prompt(
                "{{ memory.user.missing_key }}",
                storage_root=storage_root,
                audit_log=audit,
            )

    def test_missing_key_writes_audit_entry(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        with pytest.raises(TemplateMemoryNotFoundError):
            expand_system_prompt(
                "{{ memory.user.missing }}",
                storage_root=storage_root,
                audit_log=audit,
            )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "system_prompt.template.expand.failed" for r in records)

    def test_missing_key_audit_code(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        with pytest.raises(TemplateMemoryNotFoundError):
            expand_system_prompt(
                "{{ memory.user.missing }}",
                storage_root=storage_root,
                audit_log=audit,
            )
        rec = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "system_prompt.template.expand.failed"
        )
        assert rec["code"] == "template_memory_not_found"

    def test_missing_key_audit_level_is_error(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        with pytest.raises(TemplateMemoryNotFoundError):
            expand_system_prompt(
                "{{ memory.user.missing }}",
                storage_root=storage_root,
                audit_log=audit,
            )
        rec = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "system_prompt.template.expand.failed"
        )
        assert rec["level"] == "error"

    def test_missing_key_audit_detail_has_message(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        with pytest.raises(TemplateMemoryNotFoundError):
            expand_system_prompt(
                "{{ memory.user.missing }}",
                storage_root=storage_root,
                audit_log=audit,
            )
        rec = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "system_prompt.template.expand.failed"
        )
        assert len(rec["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Unexpected error → TemplateExpandError
# ---------------------------------------------------------------------------


class TestUnexpectedError:
    def test_io_error_raises_template_expand_error(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.key", "val")
        audit = FileAuditLog(storage_root)
        with patch(
            "meridiand._system_prompt_template.json.loads",
            side_effect=OSError("disk read failed"),
        ):
            with pytest.raises(TemplateExpandError):
                expand_system_prompt(
                    "{{ memory.user.key }}",
                    storage_root=storage_root,
                    audit_log=audit,
                )

    def test_unexpected_error_writes_audit_entry(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.key", "val")
        audit = FileAuditLog(storage_root)
        with patch(
            "meridiand._system_prompt_template.json.loads",
            side_effect=OSError("disk read failed"),
        ):
            with pytest.raises(TemplateExpandError):
                expand_system_prompt(
                    "{{ memory.user.key }}",
                    storage_root=storage_root,
                    audit_log=audit,
                )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "system_prompt.template.expand.failed" for r in records)

    def test_unexpected_error_audit_code(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.key", "val")
        audit = FileAuditLog(storage_root)
        with patch(
            "meridiand._system_prompt_template.json.loads",
            side_effect=OSError("disk read failed"),
        ):
            with pytest.raises(TemplateExpandError):
                expand_system_prompt(
                    "{{ memory.user.key }}",
                    storage_root=storage_root,
                    audit_log=audit,
                )
        rec = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "system_prompt.template.expand.failed"
        )
        assert rec["code"] == "template_expand_failed"

    def test_unexpected_error_audit_level_is_error(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.key", "val")
        audit = FileAuditLog(storage_root)
        with patch(
            "meridiand._system_prompt_template.json.loads",
            side_effect=OSError("disk read failed"),
        ):
            with pytest.raises(TemplateExpandError):
                expand_system_prompt(
                    "{{ memory.user.key }}",
                    storage_root=storage_root,
                    audit_log=audit,
                )
        rec = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "system_prompt.template.expand.failed"
        )
        assert rec["level"] == "error"


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOtelSpans:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_span(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        expand_system_prompt("no templates here", storage_root=storage_root, audit_log=audit)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "system_prompt.template.expand" in span_names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        with pytest.raises(TemplateMemoryNotFoundError):
            expand_system_prompt(
                "{{ memory.user.missing }}",
                storage_root=storage_root,
                audit_log=audit,
            )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "system_prompt.template.expand" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        audit = FileAuditLog(storage_root)
        with pytest.raises(TemplateMemoryNotFoundError):
            expand_system_prompt(
                "{{ memory.user.missing }}",
                storage_root=storage_root,
                audit_log=audit,
            )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("system_prompt.template.expand")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_ref_count_zero_for_no_refs(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        expand_system_prompt("plain text", storage_root=storage_root, audit_log=audit)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("system_prompt.template.expand")
        assert span is not None
        assert span.attributes["template.ref_count"] == 0

    def test_span_ref_count_matches_placeholder_sites(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.a", "x")
        _write_memory(storage_root, "user.b", "y")
        audit = FileAuditLog(storage_root)
        expand_system_prompt(
            "{{ memory.user.a }} and {{ memory.user.b }} and {{ memory.user.a }}",
            storage_root=storage_root,
            audit_log=audit,
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("system_prompt.template.expand")
        assert span is not None
        # Three placeholder sites (even though only two distinct keys)
        assert span.attributes["template.ref_count"] == 3
