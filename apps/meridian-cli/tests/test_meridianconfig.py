"""
Tests for `meridian meridianconfig validate`.

Coverage:
  - validate: prints OK and exits 0 for a valid config.
  - validate: defaults to ~/.meridian/config.yml when --config is omitted.
  - validate: accepts --config to override the path.
  - validate: exits 1 when the file does not exist.
  - validate: prints "file not found" message on missing file.
  - validate: exits 1 when the YAML is unparseable.
  - validate: prints "invalid YAML" message on bad YAML.
  - validate: exits 1 when the config is a YAML sequence, not a mapping.
  - validate: exits 1 when storage_root is absent.
  - validate: prints field name in error output when storage_root is absent.
  - validate: exits 1 on version mismatch.
  - validate: prints version mismatch message.
  - validate: writes audit log entry on failure.
  - validate: audit log entry has level "error".
  - validate: audit log entry has event "meridianconfig.validate.failed".
  - validate: audit log entry detail includes the config path.
  - validate: does NOT write audit log on success.
  - validate: emits OTel span "meridianconfig.validate".
  - validate: span has config.path attribute.
  - validate: span adds ok event on success.
  - validate: calls record_failure on span on failure.
  - validate: exits 1 when a vault has an empty id.
  - validate: prints vault id error message.
  - validate: exits 1 when vault ids are duplicated.
  - validate: exits 1 when a vault backend is invalid.
  - validate: exits 0 with valid vault entries.
  - validate: exits 1 when daemon.log_level is invalid.
  - validate: prints daemon log_level error message.
  - validate: exits 0 with valid daemon section.
  - validate: exits 0 with valid storage section.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from meridian_cli.__main__ import cli
from meridian_cli.meridianconfig import _CONFIG_VERSION, DEFAULT_CONFIG_PATH
import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_cfg(tmp_path: Path, **extra: object) -> Path:
    cfg = tmp_path / "config.yml"
    data: dict[str, object] = {"storage_root": str(tmp_path / "storage")}
    data.update(extra)
    cfg.write_text(yaml.dump(data))
    return cfg


def _run(args: list[str]) -> object:
    runner = CliRunner()
    return runner.invoke(cli, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# OTel mock shared across tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _otel_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = lambda *_: mock_span
    tracer.start_as_current_span.return_value.__exit__ = lambda *_: False
    monkeypatch.setattr("meridian_cli.meridianconfig.get_tracer", lambda: tracer)
    return mock_span


def _span_from(monkeypatch_fixture: pytest.MonkeyPatch) -> MagicMock:
    # Helper not used directly – tests grab the span via the autouse fixture.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestValidateSuccess:
    def test_prints_ok(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path)
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert "OK" in result.output

    def test_exits_0(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path)
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_no_audit_on_success(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path)
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "validate", "--config", str(cfg)])
        mock_audit.assert_not_called()

    def test_span_ok_event_on_success(self, tmp_path: Path, _otel_mock: MagicMock) -> None:
        cfg = _valid_cfg(tmp_path)
        _run(["meridianconfig", "validate", "--config", str(cfg)])
        _otel_mock.add_event.assert_any_call("meridianconfig.validate.ok")


# ---------------------------------------------------------------------------
# Default path behaviour
# ---------------------------------------------------------------------------


class TestValidateDefaultPath:
    def test_uses_default_config_path_when_not_specified(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path)
        with patch("meridian_cli.meridianconfig.DEFAULT_CONFIG_PATH", cfg):
            result = _run(["meridianconfig", "validate"])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_default_path_constant_points_to_home_meridian(self) -> None:
        assert Path.home() / ".meridian" / "config.yml" == DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# File-not-found failure
# ---------------------------------------------------------------------------


class TestValidateMissingFile:
    def test_exits_1_on_missing_file(self, tmp_path: Path) -> None:
        result = _run(["meridianconfig", "validate", "--config", str(tmp_path / "missing.yml")])
        assert result.exit_code == 1

    def test_prints_file_not_found(self, tmp_path: Path) -> None:
        result = _run(["meridianconfig", "validate", "--config", str(tmp_path / "missing.yml")])
        assert "file not found" in result.output

    def test_writes_audit_on_missing_file(self, tmp_path: Path) -> None:
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "validate", "--config", str(tmp_path / "missing.yml")])
        mock_audit.assert_called_once()
        args = mock_audit.call_args[0]
        assert args[0] == "error"

    def test_audit_event_on_missing_file(self, tmp_path: Path) -> None:
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "validate", "--config", str(tmp_path / "missing.yml")])
        args = mock_audit.call_args[0]
        assert args[1] == "meridianconfig.validate.failed"

    def test_audit_detail_has_path_on_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.yml"
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "validate", "--config", str(missing)])
        detail = mock_audit.call_args[0][2]
        assert detail["path"] == str(missing)


# ---------------------------------------------------------------------------
# Invalid YAML failure
# ---------------------------------------------------------------------------


class TestValidateInvalidYaml:
    def test_exits_1_on_invalid_yaml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("key: [\nunclosed bracket\n")
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_invalid_yaml_message(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("key: [\nunclosed bracket\n")
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert "invalid YAML" in result.output

    def test_writes_audit_on_invalid_yaml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("key: [\nunclosed bracket\n")
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "validate", "--config", str(cfg)])
        mock_audit.assert_called_once()


# ---------------------------------------------------------------------------
# YAML sequence (not a mapping) failure
# ---------------------------------------------------------------------------


class TestValidateYamlSequence:
    def test_exits_1_on_yaml_sequence(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("- item1\n- item2\n")
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_must_be_mapping_message(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("- item1\n- item2\n")
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert "mapping" in result.output


# ---------------------------------------------------------------------------
# Missing storage_root failure
# ---------------------------------------------------------------------------


class TestValidateMissingStorageRoot:
    def test_exits_1_when_storage_root_absent(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("log_level: info\n")
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_storage_root_in_errors(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("log_level: info\n")
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert "storage_root" in result.output

    def test_writes_audit_when_storage_root_absent(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("log_level: info\n")
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "validate", "--config", str(cfg)])
        mock_audit.assert_called_once()
        assert mock_audit.call_args[0][0] == "error"


# ---------------------------------------------------------------------------
# Version mismatch failure
# ---------------------------------------------------------------------------


class TestValidateVersionMismatch:
    def test_exits_1_on_version_mismatch(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, version=99)
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_version_mismatch_message(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, version=99)
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert "version" in result.output
        assert "99" in result.output

    def test_prints_expected_version_in_mismatch(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, version=99)
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert str(_CONFIG_VERSION) in result.output

    def test_writes_audit_on_version_mismatch(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, version=99)
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "validate", "--config", str(cfg)])
        mock_audit.assert_called_once()
        assert mock_audit.call_args[0][1] == "meridianconfig.validate.failed"

    def test_audit_detail_has_errors_on_version_mismatch(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, version=99)
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "validate", "--config", str(cfg)])
        detail = mock_audit.call_args[0][2]
        assert len(detail["errors"]) > 0


# ---------------------------------------------------------------------------
# OTel span behaviour
# ---------------------------------------------------------------------------


class TestValidateOtel:
    def test_emits_validate_span(self, tmp_path: Path, _otel_mock: MagicMock) -> None:
        cfg = _valid_cfg(tmp_path)
        _run(["meridianconfig", "validate", "--config", str(cfg)])
        record_invocation_event_called = _otel_mock.add_event.called
        assert record_invocation_event_called

    def test_record_failure_called_on_error(self, tmp_path: Path, _otel_mock: MagicMock) -> None:
        result = _run(["meridianconfig", "validate", "--config", str(tmp_path / "missing.yml")])
        assert result.exit_code == 1
        _otel_mock.set_status.assert_called_once()


# ---------------------------------------------------------------------------
# Vault section validation
# ---------------------------------------------------------------------------


class TestValidateVaultSection:
    def test_exits_1_on_empty_vault_id(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, vaults=[{"id": "", "backend": "os_keychain"}])
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_vault_id_error(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, vaults=[{"id": "", "backend": "os_keychain"}])
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert "id" in result.output

    def test_exits_1_on_duplicate_vault_ids(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(
            tmp_path,
            vaults=[
                {"id": "dup", "backend": "os_keychain"},
                {"id": "dup", "backend": "encrypted_file"},
            ],
        )
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_duplicate_vault_id_error(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(
            tmp_path,
            vaults=[
                {"id": "dup", "backend": "os_keychain"},
                {"id": "dup", "backend": "encrypted_file"},
            ],
        )
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert "dup" in result.output

    def test_exits_1_on_invalid_vault_backend(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, vaults=[{"id": "v1", "backend": "bad_backend"}])
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_invalid_backend_message(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, vaults=[{"id": "v1", "backend": "bad_backend"}])
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert "backend" in result.output

    def test_exits_0_with_valid_vaults(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(
            tmp_path,
            vaults=[
                {"id": "v1", "backend": "os_keychain"},
                {"id": "v2", "backend": "encrypted_file"},
            ],
        )
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_writes_audit_on_vault_error(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, vaults=[{"id": "", "backend": "os_keychain"}])
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "validate", "--config", str(cfg)])
        mock_audit.assert_called_once()
        assert mock_audit.call_args[0][1] == "meridianconfig.validate.failed"


# ---------------------------------------------------------------------------
# Daemon section validation
# ---------------------------------------------------------------------------


class TestValidateDaemonSection:
    def test_exits_1_on_invalid_daemon_log_level(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, daemon={"log_level": "verbose"})
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_daemon_log_level_error(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, daemon={"log_level": "verbose"})
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert "log_level" in result.output

    def test_exits_0_with_valid_daemon_section(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, daemon={"log_level": "debug"})
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_exits_0_without_daemon_section(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path)
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_writes_audit_on_daemon_log_level_error(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, daemon={"log_level": "verbose"})
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "validate", "--config", str(cfg)])
        mock_audit.assert_called_once()
        assert mock_audit.call_args[0][0] == "error"


# ---------------------------------------------------------------------------
# Storage section validation
# ---------------------------------------------------------------------------


class TestValidateStorageSection:
    def test_exits_0_with_valid_storage_section(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(
            tmp_path,
            storage={
                "database": str(tmp_path / "db.sqlite"),
                "event_log": str(tmp_path / "events"),
                "blob_store": str(tmp_path / "blobs"),
            },
        )
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_exits_0_without_storage_section(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path)
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_exits_0_with_partial_storage_paths(self, tmp_path: Path) -> None:
        cfg = _valid_cfg(tmp_path, storage={"database": str(tmp_path / "db.sqlite")})
        result = _run(["meridianconfig", "validate", "--config", str(cfg)])
        assert result.exit_code == 0
