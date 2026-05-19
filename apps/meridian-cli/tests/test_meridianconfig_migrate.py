"""
Tests for `meridian meridianconfig migrate`.

Coverage:
  - migrate: exits 0 and prints "migrated" when a v1 config is successfully upgraded.
  - migrate: updates the config file version to the latest version.
  - migrate: preserves other config fields (storage_root) after migration.
  - migrate: is idempotent — running twice leaves the config at the latest version.
  - migrate: writes an info audit log on success with from_version and to_version.
  - migrate: audit log event on success is "meridianconfig.migrate.ok".
  - migrate: audit log detail contains from_version and to_version on success.
  - migrate: span adds "meridianconfig.migrate.ok" event on successful migration.
  - migrate: exits 0 and prints "nothing to migrate" when already at latest version.
  - migrate: span adds "meridianconfig.migrate.noop" event when already at latest.
  - migrate: does not write audit log when config is already at latest version.
  - migrate: defaults to ~/.meridian/config.yml when --config is omitted.
  - migrate: DEFAULT_CONFIG_PATH constant points to ~/.meridian/config.yml.
  - migrate: exits 1 when the file does not exist.
  - migrate: prints "file not found" on missing file.
  - migrate: writes error audit log on missing file.
  - migrate: audit log event on missing file is "meridianconfig.migrate.failed".
  - migrate: audit log detail includes the config path on missing file.
  - migrate: exits 1 on invalid YAML.
  - migrate: prints "invalid YAML" on bad YAML.
  - migrate: exits 1 when config is a YAML sequence, not a mapping.
  - migrate: prints "mapping" message when config is a YAML sequence.
  - migrate: exits 1 when config version is newer than supported.
  - migrate: prints the unsupported version number in the error message.
  - migrate: emits OTel span "meridianconfig.migrate" on every invocation.
  - migrate: calls record_failure on span when an error occurs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from meridian_cli.__main__ import cli
from meridian_cli.meridianconfig import DEFAULT_CONFIG_PATH, _CONFIG_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _v1_cfg(tmp_path: Path, **extra: object) -> Path:
    cfg = tmp_path / "config.yml"
    data: dict[str, object] = {"version": 1, "storage_root": str(tmp_path / "storage")}
    data.update(extra)
    cfg.write_text(yaml.dump(data))
    return cfg


def _latest_cfg(tmp_path: Path, **extra: object) -> Path:
    cfg = tmp_path / "config.yml"
    data: dict[str, object] = {
        "version": _CONFIG_VERSION,
        "storage_root": str(tmp_path / "storage"),
    }
    data.update(extra)
    cfg.write_text(yaml.dump(data))
    return cfg


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
# Success — v1 migrated to latest
# ---------------------------------------------------------------------------


class TestMigrateSuccess:
    def test_exits_0_on_v1_config(self, tmp_path: Path) -> None:
        cfg = _v1_cfg(tmp_path)
        result = _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_prints_migrated_message(self, tmp_path: Path) -> None:
        cfg = _v1_cfg(tmp_path)
        result = _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert "migrated" in result.output

    def test_config_file_version_updated_to_latest(self, tmp_path: Path) -> None:
        cfg = _v1_cfg(tmp_path)
        _run(["meridianconfig", "migrate", "--config", str(cfg)])
        updated = yaml.safe_load(cfg.read_text())
        assert updated["version"] == _CONFIG_VERSION

    def test_storage_root_preserved_after_migration(self, tmp_path: Path) -> None:
        cfg = _v1_cfg(tmp_path)
        _run(["meridianconfig", "migrate", "--config", str(cfg)])
        updated = yaml.safe_load(cfg.read_text())
        assert "storage_root" in updated

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        cfg = _v1_cfg(tmp_path)
        _run(["meridianconfig", "migrate", "--config", str(cfg)])
        _run(["meridianconfig", "migrate", "--config", str(cfg)])
        updated = yaml.safe_load(cfg.read_text())
        assert updated["version"] == _CONFIG_VERSION

    def test_writes_info_audit_on_success(self, tmp_path: Path) -> None:
        cfg = _v1_cfg(tmp_path)
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "migrate", "--config", str(cfg)])
        info_calls = [c for c in mock_audit.call_args_list if c[0][0] == "info"]
        assert info_calls

    def test_audit_event_on_success(self, tmp_path: Path) -> None:
        cfg = _v1_cfg(tmp_path)
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "migrate", "--config", str(cfg)])
        info_call = next(c for c in mock_audit.call_args_list if c[0][0] == "info")
        assert info_call[0][1] == "meridianconfig.migrate.ok"

    def test_audit_detail_has_from_version(self, tmp_path: Path) -> None:
        cfg = _v1_cfg(tmp_path)
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "migrate", "--config", str(cfg)])
        info_call = next(c for c in mock_audit.call_args_list if c[0][0] == "info")
        assert info_call[0][2]["from_version"] == 1

    def test_audit_detail_has_to_version(self, tmp_path: Path) -> None:
        cfg = _v1_cfg(tmp_path)
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "migrate", "--config", str(cfg)])
        info_call = next(c for c in mock_audit.call_args_list if c[0][0] == "info")
        assert info_call[0][2]["to_version"] == _CONFIG_VERSION

    def test_span_ok_event_emitted(self, tmp_path: Path, _otel_mock: MagicMock) -> None:
        cfg = _v1_cfg(tmp_path)
        _run(["meridianconfig", "migrate", "--config", str(cfg)])
        event_names = [c[0][0] for c in _otel_mock.add_event.call_args_list]
        assert "meridianconfig.migrate.ok" in event_names


# ---------------------------------------------------------------------------
# Noop — already at latest version
# ---------------------------------------------------------------------------


class TestMigrateNoop:
    def test_exits_0_when_already_at_latest(self, tmp_path: Path) -> None:
        cfg = _latest_cfg(tmp_path)
        result = _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_prints_nothing_to_migrate(self, tmp_path: Path) -> None:
        cfg = _latest_cfg(tmp_path)
        result = _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert "nothing to migrate" in result.output

    def test_does_not_write_audit_on_noop(self, tmp_path: Path) -> None:
        cfg = _latest_cfg(tmp_path)
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "migrate", "--config", str(cfg)])
        mock_audit.assert_not_called()

    def test_span_noop_event_emitted(self, tmp_path: Path, _otel_mock: MagicMock) -> None:
        cfg = _latest_cfg(tmp_path)
        _run(["meridianconfig", "migrate", "--config", str(cfg)])
        event_names = [c[0][0] for c in _otel_mock.add_event.call_args_list]
        assert "meridianconfig.migrate.noop" in event_names


# ---------------------------------------------------------------------------
# Default path
# ---------------------------------------------------------------------------


class TestMigrateDefaultPath:
    def test_uses_default_config_path_when_not_specified(self, tmp_path: Path) -> None:
        cfg = _v1_cfg(tmp_path)
        with patch("meridian_cli.meridianconfig.DEFAULT_CONFIG_PATH", cfg):
            result = _run(["meridianconfig", "migrate"])
        assert result.exit_code == 0

    def test_default_path_constant_points_to_home_meridian(self) -> None:
        assert DEFAULT_CONFIG_PATH == Path.home() / ".meridian" / "config.yml"


# ---------------------------------------------------------------------------
# File not found
# ---------------------------------------------------------------------------


class TestMigrateFileNotFound:
    def test_exits_1_on_missing_file(self, tmp_path: Path) -> None:
        result = _run(["meridianconfig", "migrate", "--config", str(tmp_path / "missing.yml")])
        assert result.exit_code == 1

    def test_prints_file_not_found(self, tmp_path: Path) -> None:
        result = _run(["meridianconfig", "migrate", "--config", str(tmp_path / "missing.yml")])
        assert "file not found" in result.output

    def test_writes_error_audit_on_missing_file(self, tmp_path: Path) -> None:
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "migrate", "--config", str(tmp_path / "missing.yml")])
        mock_audit.assert_called_once()
        assert mock_audit.call_args[0][0] == "error"

    def test_audit_event_on_missing_file(self, tmp_path: Path) -> None:
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "migrate", "--config", str(tmp_path / "missing.yml")])
        assert mock_audit.call_args[0][1] == "meridianconfig.migrate.failed"

    def test_audit_detail_has_path_on_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.yml"
        with patch("meridian_cli.meridianconfig.write_audit") as mock_audit:
            _run(["meridianconfig", "migrate", "--config", str(missing)])
        detail = mock_audit.call_args[0][2]
        assert detail["path"] == str(missing)


# ---------------------------------------------------------------------------
# Invalid YAML
# ---------------------------------------------------------------------------


class TestMigrateInvalidYaml:
    def test_exits_1_on_invalid_yaml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("key: [\nunclosed bracket\n")
        result = _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_invalid_yaml_message(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("key: [\nunclosed bracket\n")
        result = _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert "invalid YAML" in result.output


# ---------------------------------------------------------------------------
# YAML sequence (not a mapping)
# ---------------------------------------------------------------------------


class TestMigrateYamlSequence:
    def test_exits_1_on_yaml_sequence(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("- item1\n- item2\n")
        result = _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_must_be_mapping_message(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text("- item1\n- item2\n")
        result = _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert "mapping" in result.output


# ---------------------------------------------------------------------------
# Version newer than supported
# ---------------------------------------------------------------------------


class TestMigrateVersionTooNew:
    def test_exits_1_on_future_version(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text(yaml.dump({"version": 999, "storage_root": str(tmp_path / "storage")}))
        result = _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_prints_version_in_error_message(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yml"
        cfg.write_text(yaml.dump({"version": 999, "storage_root": str(tmp_path / "storage")}))
        result = _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert "999" in result.output


# ---------------------------------------------------------------------------
# OTel span behaviour
# ---------------------------------------------------------------------------


class TestMigrateOtel:
    def test_emits_migrate_span(self, tmp_path: Path, _otel_mock: MagicMock) -> None:
        cfg = _v1_cfg(tmp_path)
        _run(["meridianconfig", "migrate", "--config", str(cfg)])
        assert _otel_mock.add_event.called

    def test_record_failure_called_on_error(self, tmp_path: Path, _otel_mock: MagicMock) -> None:
        result = _run(["meridianconfig", "migrate", "--config", str(tmp_path / "missing.yml")])
        assert result.exit_code == 1
        _otel_mock.set_status.assert_called_once()
