"""
MeridianConfig conformance suite.

Tests cover:
  - MeridianConfig: version field defaults to 2; nested BindConfig/CorsConfig defaults.
  - MeridianConfig: storage_root is required; expanduser is applied.
  - MeridianConfig: vaults defaults to empty list; daemon and storage default to None.
  - MeridianConfig: providers defaults to empty list.
  - MeridianConfig: accepts vaults, providers, daemon, and storage sections when present.
  - MERIDIAN_CONFIG_VERSION constant equals 2.
  - VaultConfig: id required; backend defaults to "os_keychain"; model is frozen.
  - ProviderConfig: name and kind required; mode/base_url/auth default to None; model is frozen.
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
  - validate_config: raises ConfigValidateError on empty provider name.
  - validate_config: raises ConfigValidateError on duplicate provider names.
  - validate_config: raises ConfigValidateError on unknown provider kind.
  - validate_config: raises ConfigValidateError on malformed secret_ref auth.
  - validate_config: accepts plaintext auth and valid secret_ref auth.
  - validate_config: plaintext auth writes audit log entry with event
    "config.plaintext_secret" per provider.
  - validate_config: plaintext auth audit entry level is "warn".
  - validate_config: plaintext auth audit entry detail includes provider name.
  - validate_config: multiple plaintext providers each write one config.plaintext_secret entry.
  - validate_config: secret_ref auth does not write config.plaintext_secret entry.
  - validate_config: None auth does not write config.plaintext_secret entry.
  - validate_config: raises ConfigValidateError on invalid daemon log_level.
  - validate_config: raises ConfigValidateError on out-of-range daemon bind port.
  - validate_config: succeeds with no vaults, no providers, and no daemon section.
  - validate_config: failure sets span status to ERROR.
  - validate_config: failure writes audit log entry with event "config.validate.failed".
  - validate_config: failure audit detail contains errors list.
  - validate_config: success with no providers does not write to audit log.
  - resolve_config_location: emits OTel span "config.resolve_location" on every invocation.
  - resolve_config_location: invocation event has code "config_resolve_location".
  - resolve_config_location: returns $MERIDIAN_CONFIG path when env var is set and file exists.
  - resolve_config_location: span has config.source="env" and config.path when env var used.
  - resolve_config_location: returns user path when env unset and user file exists.
  - resolve_config_location: span has config.source="user" when user path used.
  - resolve_config_location: returns system path when env unset and user file absent.
  - resolve_config_location: span has config.source="system" when system path used.
  - resolve_config_location: raises ConfigResolveError when no config file found.
  - resolve_config_location: $MERIDIAN_CONFIG set to missing path raises
    ConfigResolveError (no fallthrough).
  - resolve_config_location: failure writes audit log entry with event
    "config.resolve_location.failed".
  - resolve_config_location: failure sets span status to ERROR.
  - resolve_config_location: accepts optional audit_log; uses NoopAuditLog when None.
  - resolve_config_location: ConfigResolveError has code "config_resolve_failed".
  - validate_config: routing section None passes.
  - validate_config: routing.default None passes.
  - validate_config: routing.default empty rules and fallbacks pass.
  - validate_config: routing.default.rules[i].model must be in provider_name:model_id form.
  - validate_config: routing.default.rules[i].model provider must be declared in providers section.
  - validate_config: routing.default.rules[i].when.skill_id accepted.
  - validate_config: routing.default.rules[i].when.role accepted.
  - validate_config: routing.default.rules[i].when.metadata_match accepted.
  - validate_config: routing.default.rules[i].when.estimated_input_tokens.gt accepted.
  - validate_config: routing.default.rules[i].when.estimated_input_tokens.lte accepted.
  - validate_config: routing.default.rules[i].when.estimated_input_tokens with no
    bounds raises ConfigValidateError.
  - validate_config: routing.default.rules[i].when full clause accepted.
  - validate_config: routing.default.fallbacks[i].model must be in
    provider_name:model_id form.
  - validate_config: routing.default.fallbacks[i].model provider must be declared
    in providers section.
  - validate_config: routing.default.fallbacks[i].on all valid values pass.
  - validate_config: routing failure writes audit log.
  - validate_config: routing failure audit detail contains errors list.
"""

from __future__ import annotations

from pathlib import Path

from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
from meridiand._config import (
    DEFAULT_SOCKET_PATH,
    MERIDIAN_CONFIG_VERSION,
    BindConfig,
    ConfigLoadError,
    ConfigResolveError,
    ConfigValidateError,
    CorsConfig,
    DaemonConfig,
    FallbackRuleConfig,
    MeridianConfig,
    ProviderConfig,
    RoutingConditionConfig,
    RoutingConfig,
    RoutingDefaultConfig,
    RoutingRuleConfig,
    StorageConfig,
    TokenRangeConfig,
    VaultConfig,
    load_config,
    resolve_config_location,
    validate_config,
)
from opentelemetry.trace import StatusCode
from pydantic import ValidationError
import pytest
import yaml

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
        m = MeridianConfig.model_validate({"storage_root": str(tmp_path), "bind": {"socket": None}})
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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
            s.database = "/some/path"  # type: ignore[misc]

    def test_accepts_all_paths(self) -> None:
        s = StorageConfig(database="/db.sqlite", event_log="/events", blob_store="/blobs")
        assert s.database == "/db.sqlite"
        assert s.event_log == "/events"
        assert s.blob_store == "/blobs"


# ---------------------------------------------------------------------------
# TestProviderConfig
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_name_and_kind_are_required(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProviderConfig.model_validate({})

    def test_name_required(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProviderConfig.model_validate({"kind": "anthropic"})

    def test_kind_required(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProviderConfig.model_validate({"name": "my-provider"})

    def test_mode_defaults_to_none(self) -> None:
        p = ProviderConfig(name="p", kind="anthropic")
        assert p.mode is None

    def test_base_url_defaults_to_none(self) -> None:
        p = ProviderConfig(name="p", kind="anthropic")
        assert p.base_url is None

    def test_auth_defaults_to_none(self) -> None:
        p = ProviderConfig(name="p", kind="anthropic")
        assert p.auth is None

    def test_accepts_all_optional_fields(self) -> None:
        p = ProviderConfig(
            name="p",
            kind="openai",
            mode="api",
            base_url="https://api.openai.com/v1",
            auth="sk-secret",
        )
        assert p.mode == "api"
        assert p.base_url == "https://api.openai.com/v1"
        assert p.auth == "sk-secret"

    def test_model_is_frozen(self) -> None:
        p = ProviderConfig(name="p", kind="anthropic")
        with pytest.raises(ValidationError):
            p.kind = "openai"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestMeridianConfigNewSections
# ---------------------------------------------------------------------------


class TestMeridianConfigNewSections:
    def test_vaults_defaults_to_empty_list(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.vaults == []

    def test_providers_defaults_to_empty_list(self, tmp_path: Path) -> None:
        m = MeridianConfig(storage_root=tmp_path)
        assert m.providers == []

    def test_accepts_providers_list(self, tmp_path: Path) -> None:
        m = MeridianConfig.model_validate(
            {
                "storage_root": str(tmp_path),
                "providers": [{"name": "claude", "kind": "anthropic"}],
            }
        )
        assert len(m.providers) == 1
        assert m.providers[0].name == "claude"
        assert m.providers[0].kind == "anthropic"

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
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "config.validate")
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_invocation_event_has_code(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path)
        validate_config(config)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "config.validate")
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
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "config.validate")
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
# TestValidateConfigProviders
# ---------------------------------------------------------------------------


class TestValidateConfigProviders:
    def test_empty_providers_list_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path)
        validate_config(config)

    def test_empty_provider_name_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="", kind="anthropic")],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_whitespace_provider_name_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="   ", kind="anthropic")],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_duplicate_provider_names_raise(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                ProviderConfig(name="claude", kind="anthropic"),
                ProviderConfig(name="claude", kind="openai"),
            ],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_unknown_kind_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="p", kind="cohere")],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_unknown_kind_error_mentions_kind(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="p", kind="cohere")],
        )
        with pytest.raises(ConfigValidateError, match="cohere"):
            validate_config(config)

    def test_all_valid_kinds_pass(self, tmp_path: Path) -> None:
        for i, kind in enumerate(("anthropic", "openai", "local")):
            config = MeridianConfig(
                storage_root=tmp_path,
                providers=[ProviderConfig(name=f"p{i}", kind=kind)],
            )
            validate_config(config)

    def test_valid_provider_with_all_optional_fields_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                ProviderConfig(
                    name="claude",
                    kind="anthropic",
                    mode="api",
                    base_url="https://api.anthropic.com",
                    auth="sk-ant-123",
                )
            ],
        )
        validate_config(config)

    def test_plaintext_auth_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="p", kind="anthropic", auth="sk-plaintext")],
        )
        validate_config(config)

    def test_valid_secret_ref_auth_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                ProviderConfig(
                    name="p",
                    kind="anthropic",
                    auth="secret_ref://vault/my-vault/api-key",
                )
            ],
        )
        validate_config(config)

    def test_malformed_secret_ref_missing_key_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                ProviderConfig(name="p", kind="anthropic", auth="secret_ref://vault/my-vault/")
            ],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_malformed_secret_ref_missing_vault_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                ProviderConfig(name="p", kind="anthropic", auth="secret_ref://vault//my-key")
            ],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_malformed_secret_ref_wrong_scheme_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                ProviderConfig(name="p", kind="anthropic", auth="secret_ref://other/vault/key")
            ],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_malformed_secret_ref_no_slash_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                ProviderConfig(name="p", kind="anthropic", auth="secret_ref://vault/no-key-here")
            ],
        )
        with pytest.raises(ConfigValidateError):
            validate_config(config)

    def test_multiple_valid_providers_pass(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                ProviderConfig(name="claude", kind="anthropic"),
                ProviderConfig(name="gpt", kind="openai", base_url="https://api.openai.com/v1"),
                ProviderConfig(name="llama", kind="local"),
            ],
        )
        validate_config(config)

    def test_failure_writes_audit_log(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="p", kind="unknown-kind")],
        )
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigValidateError):
            validate_config(config, audit_log=audit)
        assert any(e.event == "config.validate.failed" for e in audit.entries)

    def test_failure_audit_detail_has_errors(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="p", kind="unknown-kind")],
        )
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigValidateError):
            validate_config(config, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.validate.failed")
        assert isinstance(entry.detail["errors"], list)
        assert len(entry.detail["errors"]) > 0

    def test_error_message_includes_prefix(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="p", kind="unknown-kind")],
        )
        with pytest.raises(ConfigValidateError, match=r"providers\[0\]"):
            validate_config(config)


# ---------------------------------------------------------------------------
# TestValidateConfigPlaintextSecretWarning
# ---------------------------------------------------------------------------


class TestValidateConfigPlaintextSecretWarning:
    def test_plaintext_auth_writes_warn_audit_entry(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="p", kind="anthropic", auth="sk-ant-123")],
        )
        audit = _CapturingAuditLog()
        validate_config(config, audit_log=audit)
        assert any(e.event == "config.plaintext_secret" for e in audit.entries)

    def test_plaintext_auth_audit_level_is_warn(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="p", kind="anthropic", auth="sk-ant-123")],
        )
        audit = _CapturingAuditLog()
        validate_config(config, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.plaintext_secret")
        assert entry.level == "warn"

    def test_plaintext_auth_audit_detail_has_provider_name(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="myp", kind="anthropic", auth="sk-ant-123")],
        )
        audit = _CapturingAuditLog()
        validate_config(config, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.plaintext_secret")
        assert entry.detail["provider"] == "myp"

    def test_multiple_plaintext_providers_each_write_warn_entry(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                ProviderConfig(name="p1", kind="anthropic", auth="sk-ant-a"),
                ProviderConfig(name="p2", kind="openai", auth="sk-openai-b"),
            ],
        )
        audit = _CapturingAuditLog()
        validate_config(config, audit_log=audit)
        warn_entries = [e for e in audit.entries if e.event == "config.plaintext_secret"]
        assert len(warn_entries) == 2

    def test_secret_ref_auth_does_not_write_plaintext_warn(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                ProviderConfig(
                    name="p",
                    kind="anthropic",
                    auth="secret_ref://vault/my-vault/api-key",
                )
            ],
        )
        audit = _CapturingAuditLog()
        validate_config(config, audit_log=audit)
        assert not any(e.event == "config.plaintext_secret" for e in audit.entries)

    def test_none_auth_does_not_write_plaintext_warn(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[ProviderConfig(name="p", kind="anthropic")],
        )
        audit = _CapturingAuditLog()
        validate_config(config, audit_log=audit)
        assert not any(e.event == "config.plaintext_secret" for e in audit.entries)


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


# ---------------------------------------------------------------------------
# TestResolveConfigLocation
# ---------------------------------------------------------------------------


class TestResolveConfigLocation:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    # --- OTel span emission ---

    def test_emits_resolve_location_span(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("storage_root: /tmp/storage\n")
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", cfg)
        resolve_config_location()
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "config.resolve_location" in span_names

    def test_span_has_invocation_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("storage_root: /tmp/storage\n")
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", cfg)
        resolve_config_location()
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.resolve_location"
        )
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_invocation_event_has_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("storage_root: /tmp/storage\n")
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", cfg)
        resolve_config_location()
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.resolve_location"
        )
        evt = next(e for e in span.events if e.name == "meridian.error.invocation")
        assert evt.attributes["code"] == "config_resolve_location"

    # --- env var ($MERIDIAN_CONFIG) ---

    def test_env_var_returns_that_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "env-config.yml"
        cfg.write_text("storage_root: /tmp/s\n")
        monkeypatch.setenv("MERIDIAN_CONFIG", str(cfg))
        assert resolve_config_location() == cfg

    def test_env_var_span_source_is_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "env-config.yml"
        cfg.write_text("storage_root: /tmp/s\n")
        monkeypatch.setenv("MERIDIAN_CONFIG", str(cfg))
        resolve_config_location()
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.resolve_location"
        )
        assert span.attributes["config.source"] == "env"

    def test_env_var_span_has_config_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "env-config.yml"
        cfg.write_text("storage_root: /tmp/s\n")
        monkeypatch.setenv("MERIDIAN_CONFIG", str(cfg))
        resolve_config_location()
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.resolve_location"
        )
        assert span.attributes["config.path"] == str(cfg)

    def test_env_var_missing_file_raises_config_resolve_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERIDIAN_CONFIG", str(tmp_path / "missing.yml"))
        with pytest.raises(ConfigResolveError):
            resolve_config_location()

    def test_env_var_missing_file_does_not_fall_through_to_user(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_cfg = tmp_path / "config.yml"
        user_cfg.write_text("storage_root: /tmp/s\n")
        monkeypatch.setenv("MERIDIAN_CONFIG", str(tmp_path / "missing.yml"))
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", user_cfg)
        with pytest.raises(ConfigResolveError):
            resolve_config_location()

    def test_env_var_missing_file_error_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERIDIAN_CONFIG", str(tmp_path / "missing.yml"))
        with pytest.raises(ConfigResolveError) as exc_info:
            resolve_config_location()
        assert exc_info.value.code == "config_resolve_failed"

    def test_env_var_missing_file_sets_span_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERIDIAN_CONFIG", str(tmp_path / "missing.yml"))
        with pytest.raises(ConfigResolveError):
            resolve_config_location()
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.resolve_location"
        )
        assert span.status.status_code == StatusCode.ERROR

    # --- user path (~/.meridian/config.yml) ---

    def test_user_path_returned_when_env_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_cfg = tmp_path / "config.yml"
        user_cfg.write_text("storage_root: /tmp/s\n")
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", user_cfg)
        assert resolve_config_location() == user_cfg

    def test_user_path_span_source_is_user(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_cfg = tmp_path / "config.yml"
        user_cfg.write_text("storage_root: /tmp/s\n")
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", user_cfg)
        resolve_config_location()
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.resolve_location"
        )
        assert span.attributes["config.source"] == "user"

    def test_user_path_span_has_config_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_cfg = tmp_path / "config.yml"
        user_cfg.write_text("storage_root: /tmp/s\n")
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", user_cfg)
        resolve_config_location()
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.resolve_location"
        )
        assert span.attributes["config.path"] == str(user_cfg)

    # --- system path (/etc/meridian/config.yml) ---

    def test_system_path_returned_when_user_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        system_cfg = tmp_path / "system-config.yml"
        system_cfg.write_text("storage_root: /tmp/s\n")
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr("meridiand._config.SYSTEM_CONFIG_PATH", system_cfg)
        assert resolve_config_location() == system_cfg

    def test_system_path_span_source_is_system(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        system_cfg = tmp_path / "system-config.yml"
        system_cfg.write_text("storage_root: /tmp/s\n")
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr("meridiand._config.SYSTEM_CONFIG_PATH", system_cfg)
        resolve_config_location()
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.resolve_location"
        )
        assert span.attributes["config.source"] == "system"

    def test_system_path_span_has_config_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        system_cfg = tmp_path / "system-config.yml"
        system_cfg.write_text("storage_root: /tmp/s\n")
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr("meridiand._config.SYSTEM_CONFIG_PATH", system_cfg)
        resolve_config_location()
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.resolve_location"
        )
        assert span.attributes["config.path"] == str(system_cfg)

    # --- no config found ---

    def test_no_config_found_raises_config_resolve_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr(
            "meridiand._config.SYSTEM_CONFIG_PATH", tmp_path / "nonexistent-sys.yml"
        )
        with pytest.raises(ConfigResolveError):
            resolve_config_location()

    def test_no_config_found_error_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr(
            "meridiand._config.SYSTEM_CONFIG_PATH", tmp_path / "nonexistent-sys.yml"
        )
        with pytest.raises(ConfigResolveError) as exc_info:
            resolve_config_location()
        assert exc_info.value.code == "config_resolve_failed"

    def test_no_config_found_sets_span_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr(
            "meridiand._config.SYSTEM_CONFIG_PATH", tmp_path / "nonexistent-sys.yml"
        )
        with pytest.raises(ConfigResolveError):
            resolve_config_location()
        span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "config.resolve_location"
        )
        assert span.status.status_code == StatusCode.ERROR

    def test_no_config_found_writes_audit_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr(
            "meridiand._config.SYSTEM_CONFIG_PATH", tmp_path / "nonexistent-sys.yml"
        )
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigResolveError):
            resolve_config_location(audit_log=audit)
        assert any(e.event == "config.resolve_location.failed" for e in audit.entries)

    def test_failure_audit_level_is_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr(
            "meridiand._config.SYSTEM_CONFIG_PATH", tmp_path / "nonexistent-sys.yml"
        )
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigResolveError):
            resolve_config_location(audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.resolve_location.failed")
        assert entry.level == "error"

    def test_failure_audit_code(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr(
            "meridiand._config.SYSTEM_CONFIG_PATH", tmp_path / "nonexistent-sys.yml"
        )
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigResolveError):
            resolve_config_location(audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.resolve_location.failed")
        assert entry.code == "config_resolve_failed"

    def test_no_audit_log_arg_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr(
            "meridiand._config.SYSTEM_CONFIG_PATH", tmp_path / "nonexistent-sys.yml"
        )
        with pytest.raises(ConfigResolveError):
            resolve_config_location()

    def test_noop_audit_log_accepted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr("meridiand._config._USER_CONFIG_PATH", tmp_path / "nonexistent.yml")
        monkeypatch.setattr(
            "meridiand._config.SYSTEM_CONFIG_PATH", tmp_path / "nonexistent-sys.yml"
        )
        with pytest.raises(ConfigResolveError):
            resolve_config_location(audit_log=NoopAuditLog())


# ---------------------------------------------------------------------------
# TestValidateConfigRoutingDefault
# ---------------------------------------------------------------------------


def _make_provider(name: str = "claude", kind: str = "anthropic") -> ProviderConfig:
    return ProviderConfig(name=name, kind=kind)


def _make_routing_rule(
    model: str = "claude:claude-3-5-sonnet",
    when: RoutingConditionConfig | None = None,
) -> RoutingRuleConfig:
    return RoutingRuleConfig(model=model, when=when)


def _make_fallback(
    on: str = "rate_limit",
    model: str = "claude:claude-haiku",
) -> FallbackRuleConfig:
    return FallbackRuleConfig(on=on, model=model)  # type: ignore[arg-type]


class TestValidateConfigRoutingDefault:
    def test_routing_none_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(storage_root=tmp_path, routing=None)
        validate_config(config)

    def test_routing_default_none_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            routing=RoutingConfig(default=None),
        )
        validate_config(config)

    def test_routing_default_empty_rules_and_fallbacks_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider()],
            routing=RoutingConfig(default=RoutingDefaultConfig(rules=[], fallbacks=[])),
        )
        validate_config(config)

    def test_rule_valid_model_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(rules=[_make_routing_rule("claude:claude-3-5-sonnet")])
            ),
        )
        validate_config(config)

    def test_rule_model_missing_colon_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(rules=[_make_routing_rule("no-colon-here")])
            ),
        )
        with pytest.raises(ConfigValidateError, match="provider_name:model_id"):
            validate_config(config)

    def test_rule_model_missing_colon_error_mentions_index(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(rules=[_make_routing_rule("no-colon")])
            ),
        )
        with pytest.raises(ConfigValidateError, match=r"routing\.default\.rules\[0\]"):
            validate_config(config)

    def test_rule_model_unknown_provider_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(rules=[_make_routing_rule("unknown:claude-3")])
            ),
        )
        with pytest.raises(ConfigValidateError, match="unknown"):
            validate_config(config)

    def test_rule_when_skill_id_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(
                    rules=[
                        _make_routing_rule(
                            "claude:claude-3-5-sonnet",
                            when=RoutingConditionConfig(skill_id="my-skill"),
                        )
                    ]
                )
            ),
        )
        validate_config(config)

    def test_rule_when_role_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(
                    rules=[
                        _make_routing_rule(
                            "claude:claude-3-5-sonnet",
                            when=RoutingConditionConfig(role="assistant"),
                        )
                    ]
                )
            ),
        )
        validate_config(config)

    def test_rule_when_metadata_match_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(
                    rules=[
                        _make_routing_rule(
                            "claude:claude-3-5-sonnet",
                            when=RoutingConditionConfig(metadata_match={"env": "prod"}),
                        )
                    ]
                )
            ),
        )
        validate_config(config)

    def test_rule_when_estimated_input_tokens_gt_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(
                    rules=[
                        _make_routing_rule(
                            "claude:claude-3-5-sonnet",
                            when=RoutingConditionConfig(
                                estimated_input_tokens=TokenRangeConfig(gt=1000)
                            ),
                        )
                    ]
                )
            ),
        )
        validate_config(config)

    def test_rule_when_estimated_input_tokens_lte_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(
                    rules=[
                        _make_routing_rule(
                            "claude:claude-3-5-sonnet",
                            when=RoutingConditionConfig(
                                estimated_input_tokens=TokenRangeConfig(lte=5000)
                            ),
                        )
                    ]
                )
            ),
        )
        validate_config(config)

    def test_rule_when_estimated_input_tokens_empty_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(
                    rules=[
                        _make_routing_rule(
                            "claude:claude-3-5-sonnet",
                            when=RoutingConditionConfig(estimated_input_tokens=TokenRangeConfig()),
                        )
                    ]
                )
            ),
        )
        with pytest.raises(ConfigValidateError, match="estimated_input_tokens"):
            validate_config(config)

    def test_rule_when_full_clause_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(
                    rules=[
                        _make_routing_rule(
                            "claude:claude-3-5-sonnet",
                            when=RoutingConditionConfig(
                                skill_id="forge",
                                estimated_input_tokens=TokenRangeConfig(gt=500, lte=10000),
                                metadata_match={"tier": "pro"},
                                role="user",
                            ),
                        )
                    ]
                )
            ),
        )
        validate_config(config)

    def test_fallback_valid_passes(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(
                    fallbacks=[_make_fallback("rate_limit", "claude:claude-haiku")]
                )
            ),
        )
        validate_config(config)

    def test_fallback_all_valid_on_values_pass(self, tmp_path: Path) -> None:
        for on_val in ("rate_limit", "timeout", "5xx", "any"):
            config = MeridianConfig(
                storage_root=tmp_path,
                providers=[_make_provider("claude")],
                routing=RoutingConfig(
                    default=RoutingDefaultConfig(
                        fallbacks=[_make_fallback(on_val, "claude:claude-haiku")]
                    )
                ),
            )
            validate_config(config)

    def test_fallback_model_missing_colon_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(fallbacks=[_make_fallback("rate_limit", "no-colon")])
            ),
        )
        with pytest.raises(ConfigValidateError, match="provider_name:model_id"):
            validate_config(config)

    def test_fallback_model_missing_colon_error_mentions_index(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(fallbacks=[_make_fallback("rate_limit", "no-colon")])
            ),
        )
        with pytest.raises(ConfigValidateError, match=r"routing\.default\.fallbacks\[0\]"):
            validate_config(config)

    def test_fallback_unknown_provider_raises(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(
                    fallbacks=[_make_fallback("rate_limit", "unknown:claude-haiku")]
                )
            ),
        )
        with pytest.raises(ConfigValidateError, match="unknown"):
            validate_config(config)

    def test_multiple_rules_and_fallbacks_pass(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[
                _make_provider("claude", "anthropic"),
                _make_provider("gpt", "openai"),
            ],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(
                    rules=[
                        _make_routing_rule(
                            "claude:claude-3-5-sonnet",
                            when=RoutingConditionConfig(skill_id="coding"),
                        ),
                        _make_routing_rule("gpt:gpt-4o"),
                    ],
                    fallbacks=[
                        _make_fallback("rate_limit", "gpt:gpt-4o-mini"),
                        _make_fallback("timeout", "claude:claude-haiku"),
                    ],
                )
            ),
        )
        validate_config(config)

    def test_routing_failure_writes_audit_log(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(rules=[_make_routing_rule("no-colon")])
            ),
        )
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigValidateError):
            validate_config(config, audit_log=audit)
        assert any(e.event == "config.validate.failed" for e in audit.entries)

    def test_routing_failure_audit_detail_has_errors(self, tmp_path: Path) -> None:
        config = MeridianConfig(
            storage_root=tmp_path,
            providers=[_make_provider("claude")],
            routing=RoutingConfig(
                default=RoutingDefaultConfig(rules=[_make_routing_rule("no-colon")])
            ),
        )
        audit = _CapturingAuditLog()
        with pytest.raises(ConfigValidateError):
            validate_config(config, audit_log=audit)
        entry = next(e for e in audit.entries if e.event == "config.validate.failed")
        assert isinstance(entry.detail["errors"], list)
        assert len(entry.detail["errors"]) > 0
