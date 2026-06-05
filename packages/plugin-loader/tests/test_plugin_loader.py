"""Tests for the system plugin loader.

Covers:
  - Discovery from entry_points (installed packages)
  - Discovery from ~/.meridian/plugins.yml (explicit declarations)
  - Manifest parsing: kind, sandbox_mode, capabilities
  - Merged results from both sources
  - Per-plugin failure: audit log entry written, error surfaced in result
  - Source-level failure: audit log entry written, error surfaced in result
  - OpenTelemetry span emitted with correct name and events
  - Structured log event emitted on each invocation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
from meridian_plugin_loader import (
    PluginLoader,
    PluginManifest,
)
from meridian_plugin_loader._discovery import discover_from_yml
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "test-tool",
        "kind": "tool",
        "sandbox_mode": "out_of_process",
        "entry_point": "my_pkg:my_tool",
        "capabilities": [],
    }
    base.update(overrides)
    return base


def _write_plugins_yml(path: Path, plugins: list[dict[str, Any]]) -> Path:
    yml = path / "plugins.yml"
    yml.write_text(yaml.dump({"plugins": plugins}))
    return yml


class FileAuditLog(AuditLog):
    """Minimal file-backed audit log for test assertions."""

    def __init__(self, storage_root: Path) -> None:
        storage_root.mkdir(parents=True, exist_ok=True)
        self._path = storage_root / "audit.ndjson"

    def write(self, entry: AuditLogEntry) -> None:
        import json
        import os

        record: dict[str, object] = {
            "level": entry.level,
            "event": entry.event,
            "code": entry.code,
            "timestamp": entry.timestamp,
        }
        if entry.detail:
            record["detail"] = entry.detail
        line = json.dumps(record) + "\n"
        fd = os.open(str(self._path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)

    def read_entries(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        return [json.loads(line) for line in self._path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# OTel fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def otel_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Install an in-memory OTel TracerProvider and return its exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def patched_get_tracer(*args: Any, **kwargs: Any) -> Any:
        return provider.get_tracer(*args, **kwargs)

    monkeypatch.setattr("meridian_plugin_loader._loader.trace.get_tracer", patched_get_tracer)
    return exporter


# ---------------------------------------------------------------------------
# discover_from_yml unit tests
# ---------------------------------------------------------------------------


class TestDiscoverFromYml:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert discover_from_yml(tmp_path / "missing.yml") == []

    def test_non_mapping_yaml_returns_empty(self, tmp_path: Path) -> None:
        yml = tmp_path / "plugins.yml"
        yml.write_text("- item\n")
        assert discover_from_yml(yml) == []

    def test_missing_plugins_key_returns_empty(self, tmp_path: Path) -> None:
        yml = tmp_path / "plugins.yml"
        yml.write_text("other_key: []\n")
        assert discover_from_yml(yml) == []

    def test_plugins_key_non_list_returns_empty(self, tmp_path: Path) -> None:
        yml = tmp_path / "plugins.yml"
        yml.write_text("plugins: not-a-list\n")
        assert discover_from_yml(yml) == []

    def test_returns_plugin_dicts(self, tmp_path: Path) -> None:
        yml = _write_plugins_yml(tmp_path, [_manifest_dict()])
        result = discover_from_yml(yml)
        assert len(result) == 1
        assert result[0]["name"] == "test-tool"

    def test_filters_non_dict_entries(self, tmp_path: Path) -> None:
        yml = tmp_path / "plugins.yml"
        yml.write_text(yaml.dump({"plugins": [_manifest_dict(), "not-a-dict"]}))
        result = discover_from_yml(yml)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# PluginLoader — discovery from plugins.yml
# ---------------------------------------------------------------------------


class TestPluginLoaderFromYml:
    def test_discovers_manifest_from_yml(self, tmp_path: Path) -> None:
        yml = _write_plugins_yml(tmp_path, [_manifest_dict()])
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            result = loader.load_all()
        assert len(result.manifests) == 1
        assert result.manifests[0].name == "test-tool"

    def test_missing_yml_returns_empty_manifests(self, tmp_path: Path) -> None:
        loader = PluginLoader(
            audit_log=NoopAuditLog(),
            plugins_yml=tmp_path / "missing.yml",
        )
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            result = loader.load_all()
        assert result.manifests == []
        assert result.errors == []

    def test_multiple_plugins_from_yml(self, tmp_path: Path) -> None:
        yml = _write_plugins_yml(
            tmp_path,
            [
                _manifest_dict(name="tool-a"),
                _manifest_dict(name="tool-b", kind="provider"),
            ],
        )
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            result = loader.load_all()
        names = [m.name for m in result.manifests]
        assert "tool-a" in names
        assert "tool-b" in names


# ---------------------------------------------------------------------------
# PluginLoader — discovery from entry_points
# ---------------------------------------------------------------------------


class TestPluginLoaderFromEntryPoints:
    def test_discovers_manifest_from_entry_points(self, tmp_path: Path) -> None:
        loader = PluginLoader(
            audit_log=NoopAuditLog(),
            plugins_yml=tmp_path / "missing.yml",
        )
        ep_data = [_manifest_dict(name="ep-tool")]
        with patch(
            "meridian_plugin_loader._loader.discover_from_entry_points", return_value=ep_data
        ):
            result = loader.load_all()
        assert len(result.manifests) == 1
        assert result.manifests[0].name == "ep-tool"

    def test_entry_points_failure_surfaces_error_in_result(self, tmp_path: Path) -> None:
        loader = PluginLoader(
            audit_log=NoopAuditLog(),
            plugins_yml=tmp_path / "missing.yml",
        )
        with patch(
            "meridian_plugin_loader._loader.discover_from_entry_points",
            side_effect=RuntimeError("pkg metadata broken"),
        ):
            result = loader.load_all()
        assert len(result.errors) == 1
        assert "pkg metadata broken" in result.errors[0].message

    def test_entry_points_failure_writes_audit_log(self, tmp_path: Path) -> None:
        audit = FileAuditLog(tmp_path)
        loader = PluginLoader(audit_log=audit, plugins_yml=tmp_path / "missing.yml")
        with patch(
            "meridian_plugin_loader._loader.discover_from_entry_points",
            side_effect=RuntimeError("boom"),
        ):
            loader.load_all()
        entries = audit.read_entries()
        assert any(e["code"] == "plugin_loader.entry_points_failed" for e in entries)

    def test_entry_points_failure_audit_detail_has_message(self, tmp_path: Path) -> None:
        audit = FileAuditLog(tmp_path)
        loader = PluginLoader(audit_log=audit, plugins_yml=tmp_path / "missing.yml")
        with patch(
            "meridian_plugin_loader._loader.discover_from_entry_points",
            side_effect=RuntimeError("boom"),
        ):
            loader.load_all()
        entries = audit.read_entries()
        entry = next(e for e in entries if e["code"] == "plugin_loader.entry_points_failed")
        assert "boom" in entry["detail"]["message"]


# ---------------------------------------------------------------------------
# PluginLoader — merged results from both sources
# ---------------------------------------------------------------------------


class TestPluginLoaderMerged:
    def test_merges_entry_points_and_yml(self, tmp_path: Path) -> None:
        yml = _write_plugins_yml(tmp_path, [_manifest_dict(name="yml-tool")])
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        ep_data = [_manifest_dict(name="ep-tool")]
        with patch(
            "meridian_plugin_loader._loader.discover_from_entry_points", return_value=ep_data
        ):
            result = loader.load_all()
        names = {m.name for m in result.manifests}
        assert "yml-tool" in names
        assert "ep-tool" in names

    def test_no_errors_on_clean_run(self, tmp_path: Path) -> None:
        yml = _write_plugins_yml(tmp_path, [_manifest_dict()])
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            result = loader.load_all()
        assert result.errors == []


# ---------------------------------------------------------------------------
# Manifest validation: kind, sandbox_mode, capabilities
# ---------------------------------------------------------------------------


class TestManifestParsing:
    def _load_single(self, tmp_path: Path, overrides: dict[str, Any]) -> PluginManifest:
        yml = _write_plugins_yml(tmp_path, [_manifest_dict(**overrides)])
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            result = loader.load_all()
        assert len(result.manifests) == 1
        return result.manifests[0]

    def test_kind_tool(self, tmp_path: Path) -> None:
        assert self._load_single(tmp_path, {"kind": "tool"}).kind == "tool"

    def test_kind_provider(self, tmp_path: Path) -> None:
        assert self._load_single(tmp_path, {"kind": "provider"}).kind == "provider"

    def test_kind_environment(self, tmp_path: Path) -> None:
        assert self._load_single(tmp_path, {"kind": "environment"}).kind == "environment"

    def test_kind_channel(self, tmp_path: Path) -> None:
        assert self._load_single(tmp_path, {"kind": "channel"}).kind == "channel"

    def test_sandbox_mode_in_daemon(self, tmp_path: Path) -> None:
        m = self._load_single(tmp_path, {"sandbox_mode": "in_daemon"})
        assert m.sandbox_mode == "in_daemon"

    def test_sandbox_mode_out_of_process(self, tmp_path: Path) -> None:
        m = self._load_single(tmp_path, {"sandbox_mode": "out_of_process"})
        assert m.sandbox_mode == "out_of_process"

    def test_capabilities_parsed(self, tmp_path: Path) -> None:
        caps = ["fs.read[/workspace/**]", "net.fetch[api.example.com]"]
        m = self._load_single(tmp_path, {"capabilities": caps})
        assert m.capabilities == caps

    def test_empty_capabilities_default(self, tmp_path: Path) -> None:
        yml = tmp_path / "plugins.yml"
        raw = _manifest_dict()
        raw.pop("capabilities")
        yml.write_text(yaml.dump({"plugins": [raw]}))
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            result = loader.load_all()
        assert result.manifests[0].capabilities == []

    def test_entry_point_stored(self, tmp_path: Path) -> None:
        m = self._load_single(tmp_path, {"entry_point": "acme.plugins:my_tool"})
        assert m.entry_point == "acme.plugins:my_tool"

    def test_metadata_stored(self, tmp_path: Path) -> None:
        m = self._load_single(tmp_path, {"metadata": {"timeout": 30}})
        assert m.metadata["timeout"] == 30


# ---------------------------------------------------------------------------
# Invalid manifest — error surfaced, audit log written
# ---------------------------------------------------------------------------


class TestInvalidManifest:
    def test_invalid_manifest_surfaces_error_in_result(self, tmp_path: Path) -> None:
        yml = tmp_path / "plugins.yml"
        yml.write_text(yaml.dump({"plugins": [{"name": "bad", "kind": "unknown_kind"}]}))
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            result = loader.load_all()
        assert len(result.errors) == 1
        assert result.errors[0].plugin_name == "bad"
        assert result.errors[0].code == "manifest_invalid"

    def test_invalid_manifest_error_message_non_empty(self, tmp_path: Path) -> None:
        yml = tmp_path / "plugins.yml"
        yml.write_text(yaml.dump({"plugins": [{"name": "bad", "kind": "unknown_kind"}]}))
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            result = loader.load_all()
        assert result.errors[0].message

    def test_invalid_manifest_writes_audit_log(self, tmp_path: Path) -> None:
        audit = FileAuditLog(tmp_path)
        yml = tmp_path / "plugins.yml"
        yml.write_text(yaml.dump({"plugins": [{"name": "bad", "kind": "unknown_kind"}]}))
        loader = PluginLoader(audit_log=audit, plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            loader.load_all()
        entries = audit.read_entries()
        assert any(e["code"] == "plugin_loader.manifest_invalid" for e in entries)

    def test_invalid_manifest_audit_detail_has_plugin_name(self, tmp_path: Path) -> None:
        audit = FileAuditLog(tmp_path)
        yml = tmp_path / "plugins.yml"
        yml.write_text(yaml.dump({"plugins": [{"name": "bad-plugin", "kind": "unknown_kind"}]}))
        loader = PluginLoader(audit_log=audit, plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            loader.load_all()
        entries = audit.read_entries()
        entry = next(e for e in entries if e["code"] == "plugin_loader.manifest_invalid")
        assert entry["detail"]["plugin_name"] == "bad-plugin"

    def test_valid_plugins_still_loaded_when_one_invalid(self, tmp_path: Path) -> None:
        yml = _write_plugins_yml(
            tmp_path,
            [
                _manifest_dict(name="good"),
                {"name": "bad", "kind": "unknown_kind"},
            ],
        )
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            result = loader.load_all()
        assert len(result.manifests) == 1
        assert result.manifests[0].name == "good"
        assert len(result.errors) == 1

    def test_missing_required_fields_errors(self, tmp_path: Path) -> None:
        yml = tmp_path / "plugins.yml"
        yml.write_text(yaml.dump({"plugins": [{"name": "incomplete"}]}))
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            result = loader.load_all()
        assert len(result.errors) == 1
        assert result.errors[0].code == "manifest_invalid"


# ---------------------------------------------------------------------------
# OpenTelemetry span
# ---------------------------------------------------------------------------


class TestOtel:
    def test_emits_load_all_span(self, tmp_path: Path, otel_exporter: InMemorySpanExporter) -> None:
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=tmp_path / "missing.yml")
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            loader.load_all()
        spans = otel_exporter.get_finished_spans()
        assert any(s.name == "plugin_loader.load_all" for s in spans)

    def test_span_has_invocation_event(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=tmp_path / "missing.yml")
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            loader.load_all()
        spans = otel_exporter.get_finished_spans()
        span = next(s for s in spans if s.name == "plugin_loader.load_all")
        event_names = [e.name for e in span.events]
        assert "plugin_loader.invocation" in event_names

    def test_invocation_event_has_plugins_yml_attr(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        plugins_yml = tmp_path / "plugins.yml"
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=plugins_yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            loader.load_all()
        spans = otel_exporter.get_finished_spans()
        span = next(s for s in spans if s.name == "plugin_loader.load_all")
        event = next(e for e in span.events if e.name == "plugin_loader.invocation")
        assert event.attributes["plugins_yml"] == str(plugins_yml)

    def test_span_has_plugin_loaded_event_per_plugin(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        yml = _write_plugins_yml(tmp_path, [_manifest_dict(name="my-tool")])
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            loader.load_all()
        spans = otel_exporter.get_finished_spans()
        span = next(s for s in spans if s.name == "plugin_loader.load_all")
        loaded_events = [e for e in span.events if e.name == "plugin_loader.plugin_loaded"]
        assert len(loaded_events) == 1
        assert loaded_events[0].attributes["plugin.name"] == "my-tool"

    def test_span_records_exception_on_manifest_failure(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        yml = tmp_path / "plugins.yml"
        yml.write_text(yaml.dump({"plugins": [{"name": "bad", "kind": "unknown_kind"}]}))
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            loader.load_all()
        spans = otel_exporter.get_finished_spans()
        span = next(s for s in spans if s.name == "plugin_loader.load_all")
        event_names = [e.name for e in span.events]
        assert "exception" in event_names

    def test_span_has_loaded_count_attribute(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        yml = _write_plugins_yml(tmp_path, [_manifest_dict()])
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]):
            loader.load_all()
        spans = otel_exporter.get_finished_spans()
        span = next(s for s in spans if s.name == "plugin_loader.load_all")
        assert span.attributes["plugin_loader.loaded_count"] == 1

    def test_span_records_exception_on_entry_points_failure(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=tmp_path / "missing.yml")
        with patch(
            "meridian_plugin_loader._loader.discover_from_entry_points",
            side_effect=RuntimeError("broken"),
        ):
            loader.load_all()
        spans = otel_exporter.get_finished_spans()
        span = next(s for s in spans if s.name == "plugin_loader.load_all")
        event_names = [e.name for e in span.events]
        assert "exception" in event_names


# ---------------------------------------------------------------------------
# Structured log events
# ---------------------------------------------------------------------------


class TestStructuredLog:
    def test_logs_load_all_started(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=tmp_path / "missing.yml")
        with (
            caplog.at_level(logging.INFO, logger="meridian.plugin_loader"),
            patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]),
        ):
            loader.load_all()
        assert any("plugin_loader.load_all started" in r.message for r in caplog.records)

    def test_logs_plugin_loaded(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        yml = _write_plugins_yml(tmp_path, [_manifest_dict(name="my-tool")])
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with (
            caplog.at_level(logging.INFO, logger="meridian.plugin_loader"),
            patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]),
        ):
            loader.load_all()
        assert any("plugin.loaded" in r.message for r in caplog.records)

    def test_logs_load_all_complete(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=tmp_path / "missing.yml")
        with (
            caplog.at_level(logging.INFO, logger="meridian.plugin_loader"),
            patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]),
        ):
            loader.load_all()
        assert any("plugin_loader.load_all complete" in r.message for r in caplog.records)

    def test_logs_error_on_invalid_manifest(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        yml = tmp_path / "plugins.yml"
        yml.write_text(yaml.dump({"plugins": [{"name": "bad", "kind": "unknown_kind"}]}))
        loader = PluginLoader(audit_log=NoopAuditLog(), plugins_yml=yml)
        with (
            caplog.at_level(logging.ERROR, logger="meridian.plugin_loader"),
            patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]),
        ):
            loader.load_all()
        assert any(r.levelno == logging.ERROR for r in caplog.records)
