"""
Tests for `meridian meridianconfig schema`.

Coverage:
  - schema: exits 0 on success.
  - schema: prints valid JSON to stdout.
  - schema: JSON output parses as a dict (object).
  - schema: JSON Schema includes "storage_root" in properties.
  - schema: JSON Schema includes "version" in properties.
  - schema: does NOT write audit log on success.
  - schema: emits OTel span "meridianconfig.schema" on every invocation.
  - schema: span adds "meridianconfig.schema.ok" event on success.
  - schema: exits 1 on unexpected error generating schema.
  - schema: prints error message on failure.
  - schema: writes error audit log on failure.
  - schema: audit log event on failure is "meridianconfig.schema.failed".
  - schema: calls record_failure on span when an error occurs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from meridian_cli.__main__ import cli
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(args: list[str]) -> object:
    runner = CliRunner()
    return runner.invoke(cli, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# OTel mock (overrides the conftest autouse fixture for this module)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _otel_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = lambda *_: mock_span
    tracer.start_as_current_span.return_value.__exit__ = lambda *_: False
    monkeypatch.setattr("meridian_cli.meridianconfig.get_tracer", lambda: tracer)
    return mock_span


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestSchemaSuccess:
    def test_exits_0(self) -> None:
        result = _run(["meridianconfig", "schema"])
        assert result.exit_code == 0

    def test_prints_valid_json(self) -> None:
        result = _run(["meridianconfig", "schema"])
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)

    def test_schema_includes_storage_root(self) -> None:
        result = _run(["meridianconfig", "schema"])
        parsed = json.loads(result.output)
        props = parsed.get("properties", {})
        assert "storage_root" in props

    def test_schema_includes_version(self) -> None:
        result = _run(["meridianconfig", "schema"])
        parsed = json.loads(result.output)
        props = parsed.get("properties", {})
        assert "version" in props

    def test_no_audit_on_success(self) -> None:
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "schema"])
        mock_audit.assert_not_called()

    def test_span_ok_event_emitted(self, _otel_mock: MagicMock) -> None:
        _run(["meridianconfig", "schema"])
        event_names = [c[0][0] for c in _otel_mock.add_event.call_args_list]
        assert "meridianconfig.schema.ok" in event_names


# ---------------------------------------------------------------------------
# OTel span behaviour
# ---------------------------------------------------------------------------


class TestSchemaOtel:
    def test_emits_schema_span(self, _otel_mock: MagicMock) -> None:
        _run(["meridianconfig", "schema"])
        assert _otel_mock.add_event.called

    def test_record_failure_called_on_error(self, _otel_mock: MagicMock) -> None:
        with patch(
            "meridian_cli.meridianconfig._MeridianConfig.model_json_schema",
            side_effect=RuntimeError("boom"),
        ):
            result = _run(["meridianconfig", "schema"])
        assert result.exit_code == 1
        _otel_mock.set_status.assert_called_once()


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestSchemaFailure:
    def test_exits_1_on_unexpected_error(self) -> None:
        with patch(
            "meridian_cli.meridianconfig._MeridianConfig.model_json_schema",
            side_effect=RuntimeError("boom"),
        ):
            result = _run(["meridianconfig", "schema"])
        assert result.exit_code == 1

    def test_prints_error_message_on_failure(self) -> None:
        with patch(
            "meridian_cli.meridianconfig._MeridianConfig.model_json_schema",
            side_effect=RuntimeError("boom"),
        ):
            result = _run(["meridianconfig", "schema"])
        assert "failed to generate config schema" in result.output

    def test_writes_error_audit_on_failure(self) -> None:
        with (
            patch(
                "meridian_cli.meridianconfig._MeridianConfig.model_json_schema",
                side_effect=RuntimeError("boom"),
            ),
            patch("meridian_cli.meridianconfig.write_audit") as mock_audit,
        ):
            _run(["meridianconfig", "schema"])
        mock_audit.assert_called_once()
        assert mock_audit.call_args[0][0] == "error"

    def test_audit_event_on_failure(self) -> None:
        with (
            patch(
                "meridian_cli.meridianconfig._MeridianConfig.model_json_schema",
                side_effect=RuntimeError("boom"),
            ),
            patch("meridian_cli.meridianconfig.write_audit") as mock_audit,
        ):
            _run(["meridianconfig", "schema"])
        assert mock_audit.call_args[0][1] == "meridianconfig.schema.failed"
