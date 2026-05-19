"""
MeridianConfig conformance suite.

Tests cover:
  - MeridianConfig: version field defaults to 2; nested BindConfig/CorsConfig defaults.
  - MeridianConfig: storage_root is required; expanduser is applied.
  - MeridianConfig: vaults defaults to empty list; daemon and storage default to None.
  - MeridianConfig: accepts vaults, daemon, and storage sections when present.
  - MERIDIAN_CONFIG_VERSION constant equals 2.
  - VaultConfig: id required; backend defaults to "os_keychain"; model is frozen.
  - DaemonConfig: bind/workspace_root/log_level defaults; workspace_root expanduser.
  - StorageConfig: database/event_log/blob_store default to None; model is frozen.
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
  - validate_config: emits OTel span "config.validate" on every invocation.
  - validate_config: structured event "meridian.error.invocation" with code "config_validate".
  - validate_config: raises ConfigValidateError on empty vault id.
  - validate_config: raises ConfigValidateError on duplicate vault ids.
  - validate_config: raises ConfigValidateError on invalid vault backend.
  - validate_config: raises ConfigValidateError on invalid daemon log_level.
  - validate_config: raises ConfigValidateError on out-of-range daemon bind port.
  - validate_config: succeeds with no vaults and no daemon section.
  - validate_config: failure sets span status to ERROR.
  - validate_config: failure writes audit log entry with event "config.validate.failed".
  - validate_config: failure audit detail contains errors list.
  - validate_config: success does not write to audit log.
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
    ConfigValidateError,
    CorsConfig,
    DaemonConfig,
    DEFAULT_SOCKET_PATH,
    MERIDIAN_CONFIG_VERSION,
    MeridianConfig,
    StorageConfig,
    VaultConfig,
    load_config,
    validate_config,
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
    def test_binary_version_constant_is_2(self) -> None:
        assert MERIDIAN_CONFIG_VERSION == 2

    def test_model_version_default_is_2(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.version == 2

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
        assert m.bind.port == 8888

    def test_default_bind_socket_is_default_path(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.bind.socket == str(DEFAULT_SOCKET_PATH)

    def test_bind_socket_none_disables_socket(self, tmp_path: Path) -> None:
        m = MeridianConfig.model_validate(
            {"storage_root": str(tmp_path), "bind": {"socket": None}}
        )
        assert m.bind.socket is None

    def test_bind_socket_expanduser(self, tmp_path: Path) -> None:
        m = MeridianConfig.model_validate(
            {"storage_root": str(tmp_path), "bind": {"socket": "~/custom.sock"}}
        )
        assert not m.bind.socket.startswith("~")

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
        assert span.attributes["config.version"] == 2

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

    def test_omitted_version_defaults_to_2_and_succeeds(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        result = load_config(cfg)
        assert result.version == 2

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


# ---------------------------------------------------------------------------
# TestVaultConfig
# ---------------------------------------------------------------------------


class TestVaultConfig:
    def test_id_field_is_required(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VaultConfig.model_validate({})

    def test_backend_defaults_to_os_keychain(self) -> None:
        v = VaultConfig(id="my-vault")
        assert v.backend == "os_keychain"

    def test_accepts_encrypted_file_backend(self) -> None:
        v = VaultConfig(id="my-vault", backend="encrypted_file")
        assert v.backend == "encrypted_file"

    def test_model_is_frozen(self) -> None:
        v = VaultConfig(id="my-vault")
        with pytest.raises(Exception):
            v.backend = "encrypted_file"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestDaemonConfig
# ---------------------------------------------------------------------------


class TestDaemonConfig:
    def test_bind_defaults_to_bind_config(self) -> None:
        d = DaemonConfig()
        assert isinstance(d.bind, BindConfig)

    def test_workspace_root_defaults_to_home_meridian(self) -> None:
        d = DaemonConfig()
        from pathlib import Path

        assert d.workspace_root == Path.home() / ".meridian"

    def test_log_level_defaults_to_info(self) -> None:
        d = DaemonConfig()
        assert d.log_level == "info"

    def test_workspace_root_expanduser(self) -> None:
        d = DaemonConfig.model_validate({"workspace_root": "~/custom-workspace"})
        assert not str(d.workspace_root).startswith("~")

    def test_model_is_frozen(self) -> None:
        d = DaemonConfig()
        with pytest.raises(Exception):
            d.log_level = "debug"  # type: ignore[misc]

    def test_accepts_custom_bind(self) -> None:
        d = DaemonConfig.model_validate({"bind": {"host": "0.0.0.0", "port": 9000}})
        assert d.bind.host == "0.0.0.0"
        assert d.bind.port == 9000


# ---------------------------------------------------------------------------
# TestStorageConfig
# ---------------------------------------------------------------------------


class TestStorageConfig:
    def test_database_defaults_to_none(self) -> None:
        s = StorageConfig()
        assert s.database is None

    def test_event_log_defaults_to_none(self) -> None:
        s = StorageConfig()
        assert s.event_log is None

    def test_blob_store_defaults_to_none(self) -> None:
        s = StorageConfig()
        assert s.blob_store is None

    def test_model_is_frozen(self) -> None:
        s = StorageConfig()
        with pytest.raises(Exception):
            s.database = "/some/path"  # type: ignore[misc]

    def test_accepts_all_paths(self) -> None:
        s = StorageConfig(database="/db.sqlite", event_log="/events", blob_store="/blobs")
        assert s.database == "/db.sqlite"
        assert s.event_log == "/events"
        assert s.blob_store == "/blobs"


# ---------------------------------------------------------------------------
# TestMeridianConfigNewSections
# ---------------------------------------------------------------------------


class TestMeridianConfigNewSections:
    def test_vaults_defaults_to_empty_list(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.vaults == []

    def test_daemon_defaults_to_none(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.daemon is None

    def test_storage_defaults_to_none(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.storage is None

    def test_accepts_vaults_list(self, tmp_path: Path) -> None:
        m = MeridianConfig.model_validate(
            {
                "storage_root": str(tmp_path),
                "vaults": [{"id": "v1", "backend": "os_keychain"}],
            }
        )
        assert len(m.vaults) == 1
        assert m.vaults[0].id == "v1"

    def test_accepts_daemon_section(self, tmp_path: Path) -> None:
        m = MeridianConfig.model_validate(
            {
                "storage_root": str(tmp_path),
                "daemon": {"log_level": "debug", "workspace_root": str(tmp_path)},
            }
        )
        assert m.daemon is not None
        assert m.daemon.log_level == "debug"

    def test_accepts_storage_section(self, tmp_path: Path) -> None:
        m = MeridianConfig.model_validate(
            {
                "storage_root": str(tmp_path),
                "storage": {"database": "/tmp/db.sqlite"},
            }
        )
        assert m.storage is not None
        assert m.storage.database == "/tmp/db.sqlite"


# ---------------------------------------------------------------------------
# TestValidateConfigOtel
# ---------------------------------------------------------------------------


class TestValidateConfigOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_emits_config_validate_span(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path)
        validate_config(config)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "config.validate" in span_names

    def test_span_has_invocation_event(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path)
        validate_config(config)
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.validate"
        )
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_invocation_event_has_code(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path)
        validate_config(config)
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.validate"
        )
        evt = next(e for e in span.events if e.name == "meridian.error.invocation")
        assert evt.attributes["code"] == "config_validate"

    def test_failure_sets_span_error_status(self, tmp_path: Path) -> None:
        from opentelemetry.trace import StatusCode

        config = MeridianConfig(
            storage_root=tmp_path,
            vaults=[VaultConfig(id="")],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.validate"
        )
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# TestValidateConfigVaults
# ---------------------------------------------------------------------------


class TestValidateConfigVaults:
    def test_empty_vault_id_raises_config_validate_error(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path, vaults=[VaultConfig(id="")])
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_whitespace_vault_id_raises_config_validate_error(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path, vaults=[VaultConfig(id="   ")])
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_duplicate_vault_ids_raise_config_validate_error(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            vaults=[VaultConfig(id="dup"), VaultConfig(id="dup")],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_invalid_vault_backend_raises_config_validate_error(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            vaults=[VaultConfig(id="v1", backend="unknown_backend")],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_valid_single_vault_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            vaults=[VaultConfig(id="v1", backend="os_keychain")],
        )
        validate_config(config)

    def test_valid_multiple_vaults_pass(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            vaults=[
                VaultConfig(id="v1", backend="os_keychain"),
                VaultConfig(id="v2", backend="encrypted_file"),
            ],
        )
        validate_config(config)

    def test_failure_writes_audit_log(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path, vaults=[VaultConfig(id="")])
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigValidateError):
            validate_config(config, audit_log=audit)
        assert any(e.event == "config.validate.failed" for e in audit.entries)

    def test_failure_audit_level_is_error(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path, vaults=[VaultConfig(id="")])
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigValidateError):
            validate_config(config, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.validate.failed")
        assert entry.level == "error"

    def test_failure_audit_code(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path, vaults=[VaultConfig(id="")])
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigValidateError):
            validate_config(config, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.validate.failed")
        assert entry.code == "config_validate_failed"

    def test_failure_audit_detail_has_errors(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path, vaults=[VaultConfig(id="")])
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigValidateError):
            validate_config(config, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.validate.failed")
        assert isinstance(entry.detail["errors"], list)
        assert len(entry.detail["errors"]) > 0

    def test_success_does_not_write_audit_log(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path)
        audit = _CapturingAuditLog()
        validate_config(config, audit_log=audit)
        assert audit.entries == []


# ---------------------------------------------------------------------------
# TestValidateConfigDaemon
# ---------------------------------------------------------------------------


class TestValidateConfigDaemon:
    def test_none_daemon_section_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path, daemon=None)
        validate_config(config)

    def test_valid_daemon_section_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            daemon=DaemonConfig(log_level="debug"),
        )
        validate_config(config)

    def test_invalid_log_level_raises_config_validate_error(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            daemon=DaemonConfig(log_level="verbose"),
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_all_valid_log_levels_pass(self, tmp_path: Path) -> None:
        for level in ("debug", "info", "warning", "error", "critical"):
            config = MeridianConfig(
                storage_root=tmp_path,
                daemon=DaemonConfig(log_level=level),
            )
            validate_config(config)

    def test_invalid_log_level_error_code(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            daemon=DaemonConfig(log_level="verbose"),
        )
        with pytest.raises(ConfigValidateError) as exc_info:
            validate_config(config)
        assert exc_info.value.code == "config_validate_failed"

    def test_invalid_log_level_writes_audit_log(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            daemon=DaemonConfig(log_level="verbose"),
        )
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigValidateError):
            validate_config(config, audit_log=audit)
        assert any(e.event == "config.validate.failed" for e in audit.entries)


# ---------------------------------------------------------------------------
# TestValidateConfigStorage
# ---------------------------------------------------------------------------


class TestValidateConfigStorage:
    def test_none_storage_section_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path, storage=None)
        validate_config(config)

    def test_storage_with_all_paths_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            storage=StorageConfig(
                database=str(tmp_path / "db.sqlite"),
                event_log=str(tmp_path / "events"),
                blob_store=str(tmp_path / "blobs"),
            ),
        )
        validate_config(config)

    def test_storage_with_partial_paths_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            storage=StorageConfig(database=str(tmp_path / "db.sqlite")),
        )
        validate_config(config)
