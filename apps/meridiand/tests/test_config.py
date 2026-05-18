"""
MeridianConfig conformance suite.

Tests cover:
  - MeridianConfig: version field defaults to 1; nested BindConfig/CorsConfig defaults.
  - MeridianConfig: storage_root is required; expanduser is applied.
  - MERIDIAN_CONFIG_VERSION constant equals 1.
  - load_config: emits OTel span "config.load" on every invocation.
  - load_config: span carries "config.path" attribute.
  - load_config: span carries "config.version" attribute on success.
  - load_config: structured event "meridian.error.invocation" attached to span.
  - load_config: version mismatch raises ConfigLoadError.
  - load_config: version mismatch writes audit log entry with event "config.load.failed".
  - load_config: version mismatch sets span status to ERROR.
  - load_config: missing file raises ConfigLoadError and writes audit log.
  - load_config: invalid YAML raises ConfigLoadError and writes audit log.
  - load_config: missing storage_root raises ConfigLoadError and writes audit log.
  - load_config: accepts optional audit_log; uses NoopAuditLog when None.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
from opentelemetry.trace import StatusCode

from meridiand._config import (
    BindConfig,
    ConfigLoadError,
    CorsConfig,
    MERIDIAN_CONFIG_VERSION,
    MeridianConfig,
    load_config,
)

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


def _write_cfg(tmp_path: Path, **extra: object) -> Path:
    cfg = tmp_path / "config.yaml"
    data: dict[str, object] = {"storage_root": str(tmp_path / "storage")}
    data.update(extra)
    cfg.write_text(yaml.dump(data))
    return cfg


# ---------------------------------------------------------------------------
# TestMeridianConfigVersion
# ---------------------------------------------------------------------------


class TestMeridianConfigVersion:
    def test_binary_version_constant_is_1(self) -> None:
        assert MERIDIAN_CONFIG_VERSION == 1

    def test_model_version_default_is_1(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.version == 1

    def test_model_accepts_explicit_version(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path, version=2)
        assert m.version == 2


# ---------------------------------------------------------------------------
# TestMeridianConfigModel
# ---------------------------------------------------------------------------


class TestMeridianConfigModel:
    def test_storage_root_required(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MeridianConfig.model_validate({})

    def test_storage_root_expanduser(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=Path("~/meridian-store"))
        assert not str(m.storage_root).startswith("~")

    def test_default_bind_host(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.bind.host == "127.0.0.1"

    def test_default_bind_port(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.bind.port == 7432

    def test_default_bind_socket_none(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.bind.socket is None

    def test_default_log_level(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.log_level == "info"

    def test_default_cors_origins_empty(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.cors.allow_origins == []

    def test_default_cors_allow_credentials_false(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.cors.allow_credentials is False

    def test_bind_config_is_pydantic_model(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert isinstance(m.bind, BindConfig)

    def test_cors_config_is_pydantic_model(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert isinstance(m.cors, CorsConfig)

    def test_model_validate_from_dict(self, tmp_path: Path) -> None:
        m = MeridianConfig.model_validate({"storage_root": str(tmp_path)})
        assert m.storage_root == tmp_path

    def test_model_validate_nested_bind(self, tmp_path: Path) -> None:
        m = MeridianConfig.model_validate(
            {"storage_root": str(tmp_path), "bind": {"host": "0.0.0.0", "port": 9000}}
        )
        assert m.bind.host == "0.0.0.0"
        assert m.bind.port == 9000

    def test_model_is_frozen(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        with pytest.raises(Exception):
            m.log_level = "debug"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestLoadConfigOtel
# ---------------------------------------------------------------------------


class TestLoadConfigOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_emits_config_load_span(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        load_config(cfg)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "config.load" in span_names

    def test_span_has_config_path_attribute(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        load_config(cfg)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "config.load")
        assert span.attributes["config.path"] == str(cfg)

    def test_span_has_config_version_attribute_on_success(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        load_config(cfg)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "config.load")
        assert span.attributes["config.version"] == 1

    def test_span_has_invocation_event(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        load_config(cfg)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "config.load")
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_invocation_event_has_code(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        load_config(cfg)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "config.load")
        evt = next(e for e in span.events if e.name == "meridian.error.invocation")
        assert evt.attributes["code"] == "config_load"

    def test_failure_sets_span_error_status(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoadError):
            load_config(tmp_path / "missing.yaml")
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "config.load")
        assert span.status.status_code == StatusCode.ERROR

    def test_version_mismatch_sets_span_error_status(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, version=99)
        with pytest.raises(ConfigLoadError):
            load_config(cfg)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "config.load")
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# TestLoadConfigVersionCheck
# ---------------------------------------------------------------------------


class TestLoadConfigVersionCheck:
    def test_matching_version_succeeds(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, version=MERIDIAN_CONFIG_VERSION)
        result = load_config(cfg)
        assert result.version == MERIDIAN_CONFIG_VERSION

    def test_omitted_version_defaults_to_1_and_succeeds(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        result = load_config(cfg)
        assert result.version == 1

    def test_version_mismatch_raises_config_load_error(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, version=99)
        with pytest.raises(ConfigLoadError):
            load_config(cfg)

    def test_version_mismatch_error_mentions_version(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, version=99)
        with pytest.raises(ConfigLoadError, match="99"):
            load_config(cfg)

    def test_version_mismatch_error_code(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, version=99)
        with pytest.raises(ConfigLoadError) as exc_info:
            load_config(cfg)
        assert exc_info.value.code == "config_load_failed"

    def test_version_mismatch_writes_audit_log(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, version=99)
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigLoadError):
            load_config(cfg, audit_log=audit)
        assert any(e.event == "config.load.failed" for e in audit.entries)

    def test_version_mismatch_audit_level_is_error(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, version=99)
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigLoadError):
            load_config(cfg, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.load.failed")
        assert entry.level == "error"

    def test_version_mismatch_audit_code(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, version=99)
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigLoadError):
            load_config(cfg, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.load.failed")
        assert entry.code == "config_load_failed"

    def test_version_mismatch_audit_detail_has_path(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path, version=99)
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigLoadError):
            load_config(cfg, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.load.failed")
        assert entry.detail["path"] == str(cfg)


# ---------------------------------------------------------------------------
# TestLoadConfigFailureAudit
# ---------------------------------------------------------------------------


class TestLoadConfigFailureAudit:
    def test_missing_file_raises_config_load_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoadError):
            load_config(tmp_path / "missing.yaml")

    def test_missing_file_writes_audit_log(self, tmp_path: Path) -> None:
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigLoadError):
            load_config(tmp_path / "missing.yaml", audit_log=audit)
        assert any(e.event == "config.load.failed" for e in audit.entries)

    def test_missing_file_audit_detail_has_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.yaml"
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigLoadError):
            load_config(missing, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.load.failed")
        assert entry.detail["path"] == str(missing)

    def test_invalid_yaml_raises_config_load_error(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- item\n")
        with pytest.raises(ConfigLoadError):
            load_config(cfg)

    def test_invalid_yaml_writes_audit_log(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- item\n")
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigLoadError):
            load_config(cfg, audit_log=audit)
        assert any(e.event == "config.load.failed" for e in audit.entries)

    def test_missing_storage_root_raises_config_load_error(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("log_level: info\n")
        with pytest.raises(ConfigLoadError):
            load_config(cfg)

    def test_missing_storage_root_writes_audit_log(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("log_level: info\n")
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigLoadError):
            load_config(cfg, audit_log=audit)
        assert any(e.event == "config.load.failed" for e in audit.entries)

    def test_no_audit_log_arg_does_not_raise(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- item\n")
        with pytest.raises(ConfigLoadError):
            load_config(cfg)

    def test_noop_audit_log_accepted(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- item\n")
        with pytest.raises(ConfigLoadError):
            load_config(cfg, audit_log=NoopAuditLog())
