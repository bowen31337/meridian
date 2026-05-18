"""
Meridiand conformance suite.

Tests cover:
  - load_config(): storage_root, bind (TCP + Unix socket), log_level, defaults,
    missing required keys, missing file.
  - FileAuditLog: NDJSON append, all entry fields, O_APPEND multi-write.
  - create_app(): "meridiand ready" logged on startup via lifespan.
  - main(): missing/invalid config returns non-zero + stderr message;
    successful boot calls uvicorn.run with correct kwargs;
    uvicorn failure writes audit entry and prints to stderr.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from core_errors import AuditLogEntry
from fastapi.testclient import TestClient
from meridiand.__main__ import main
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._config import ConfigLoadError, load_config

# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_parses_storage_root(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"storage_root: {tmp_path / 'storage'}\n")
        assert load_config(cfg).storage_root == tmp_path / "storage"

    def test_default_bind_host(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"storage_root: {tmp_path / 'storage'}\n")
        assert load_config(cfg).bind.host == "127.0.0.1"

    def test_default_bind_port(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"storage_root: {tmp_path / 'storage'}\n")
        assert load_config(cfg).bind.port == 8888

    def test_default_log_level(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"storage_root: {tmp_path / 'storage'}\n")
        assert load_config(cfg).log_level == "info"

    def test_custom_bind_host_and_port(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "storage_root": str(tmp_path / "storage"),
                    "bind": {"host": "0.0.0.0", "port": 8080},
                }
            )
        )
        result = load_config(cfg)
        assert result.bind.host == "0.0.0.0"
        assert result.bind.port == 8080

    def test_unix_socket_bind(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "storage_root": str(tmp_path / "storage"),
                    "bind": {"socket": "/run/meridiand.sock"},
                }
            )
        )
        assert load_config(cfg).bind.socket == "/run/meridiand.sock"

    def test_custom_log_level(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "storage_root": str(tmp_path / "storage"),
                    "log_level": "debug",
                }
            )
        )
        assert load_config(cfg).log_level == "debug"

    def test_missing_storage_root_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("log_level: info\n")
        with pytest.raises(ConfigLoadError, match="storage_root"):
            load_config(cfg)

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- item\n")
        with pytest.raises(ConfigLoadError, match="not a YAML mapping"):
            load_config(cfg)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoadError):
            load_config(tmp_path / "missing.yaml")

    def test_storage_root_expanduser(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("storage_root: ~/meridian-store\n")
        result = load_config(cfg)
        assert not str(result.storage_root).startswith("~")


# ---------------------------------------------------------------------------
# FileAuditLog
# ---------------------------------------------------------------------------


class TestFileAuditLog:
    def _entry(self, **overrides: object) -> AuditLogEntry:
        defaults = dict(
            level="error",
            event="test.event",
            code="test_code",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        defaults.update(overrides)
        return AuditLogEntry(**defaults)  # type: ignore[arg-type]

    def test_creates_audit_ndjson(self, storage_root: Path) -> None:
        log = FileAuditLog(storage_root)
        log.write(self._entry())
        assert (storage_root / "audit.ndjson").exists()

    def test_single_line_written(self, storage_root: Path) -> None:
        log = FileAuditLog(storage_root)
        log.write(self._entry())
        lines = (storage_root / "audit.ndjson").read_text().splitlines()
        assert len(lines) == 1

    def test_multiple_writes_append(self, storage_root: Path) -> None:
        log = FileAuditLog(storage_root)
        for _ in range(3):
            log.write(self._entry())
        lines = (storage_root / "audit.ndjson").read_text().splitlines()
        assert len(lines) == 3

    def test_level_field(self, storage_root: Path) -> None:
        log = FileAuditLog(storage_root)
        log.write(self._entry(level="warn"))
        record = json.loads((storage_root / "audit.ndjson").read_text().strip())
        assert record["level"] == "warn"

    def test_event_field(self, storage_root: Path) -> None:
        log = FileAuditLog(storage_root)
        log.write(self._entry(event="my.event"))
        record = json.loads((storage_root / "audit.ndjson").read_text().strip())
        assert record["event"] == "my.event"

    def test_code_field(self, storage_root: Path) -> None:
        log = FileAuditLog(storage_root)
        log.write(self._entry(code="my_code"))
        record = json.loads((storage_root / "audit.ndjson").read_text().strip())
        assert record["code"] == "my_code"

    def test_timestamp_field(self, storage_root: Path) -> None:
        log = FileAuditLog(storage_root)
        log.write(self._entry())
        record = json.loads((storage_root / "audit.ndjson").read_text().strip())
        assert record["timestamp"] == "2026-01-01T00:00:00+00:00"

    def test_detail_field_included(self, storage_root: Path) -> None:
        log = FileAuditLog(storage_root)
        log.write(self._entry(detail={"key": "val"}))
        record = json.loads((storage_root / "audit.ndjson").read_text().strip())
        assert record["detail"] == {"key": "val"}

    def test_no_detail_field_omitted(self, storage_root: Path) -> None:
        log = FileAuditLog(storage_root)
        log.write(self._entry())
        record = json.loads((storage_root / "audit.ndjson").read_text().strip())
        assert "detail" not in record

    def test_creates_storage_root_if_missing(self, tmp_path: Path) -> None:
        root = tmp_path / "new_storage"
        log = FileAuditLog(root)
        log.write(self._entry())
        assert (root / "audit.ndjson").exists()

    def test_file_mode_600(self, storage_root: Path) -> None:
        import stat

        log = FileAuditLog(storage_root)
        log.write(self._entry())
        mode = (storage_root / "audit.ndjson").stat().st_mode
        assert stat.S_IMODE(mode) == 0o600


# ---------------------------------------------------------------------------
# create_app — lifespan
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_ready_logged_on_startup(
        self, storage_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        audit_log = FileAuditLog(storage_root)
        app = create_app(audit_log)
        with caplog.at_level(logging.INFO, logger="meridiand"), TestClient(app):
            pass
        assert any("meridiand ready" in r.message for r in caplog.records)

    def test_error_handler_installed(self, storage_root: Path) -> None:
        from core_errors import CapabilityDeniedError

        audit_log = FileAuditLog(storage_root)
        app = create_app(audit_log)

        @app.get("/denied")
        def raise_denied() -> None:
            raise CapabilityDeniedError(message="no", timestamp="2026-01-01T00:00:00+00:00")

        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/denied").status_code == 403


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def _write_cfg(self, tmp_path: Path, **extra: object) -> Path:
        cfg = tmp_path / "config.yaml"
        data: dict[str, object] = {"storage_root": str(tmp_path / "storage")}
        data.update(extra)
        cfg.write_text(yaml.dump(data))
        return cfg

    def test_missing_config_returns_nonzero(self, tmp_path: Path) -> None:
        assert main(["--config", str(tmp_path / "missing.yaml")]) != 0

    def test_missing_config_prints_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        main(["--config", str(tmp_path / "missing.yaml")])
        assert "meridiand" in capsys.readouterr().err

    def test_invalid_config_returns_nonzero(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("log_level: info\n")
        assert main(["--config", str(cfg)]) != 0

    def test_invalid_config_prints_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("log_level: info\n")
        main(["--config", str(cfg)])
        assert "meridiand" in capsys.readouterr().err

    def test_successful_boot_returns_zero(self, tmp_path: Path) -> None:
        cfg = self._write_cfg(tmp_path)
        with patch("meridiand.__main__.uvicorn.run"):
            assert main(["--config", str(cfg)]) == 0

    def test_successful_boot_calls_uvicorn(self, tmp_path: Path) -> None:
        cfg = self._write_cfg(tmp_path)
        with patch("meridiand.__main__.uvicorn.run") as mock_run:
            main(["--config", str(cfg)])
        mock_run.assert_called_once()

    def test_tcp_bind_kwargs_passed_to_uvicorn(self, tmp_path: Path) -> None:
        # socket: null explicitly disables socket binding so TCP kwargs are forwarded.
        cfg = self._write_cfg(tmp_path, bind={"host": "0.0.0.0", "port": 9000, "socket": None})
        with patch("meridiand.__main__.uvicorn.run") as mock_run:
            main(["--config", str(cfg)])
        _, kwargs = mock_run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000
        assert "uds" not in kwargs

    def test_unix_socket_kwargs_passed_to_uvicorn(self, tmp_path: Path) -> None:
        socket_path = str(tmp_path / "meridiand.sock")
        cfg = self._write_cfg(tmp_path, bind={"socket": socket_path})
        with patch("meridiand.__main__.uvicorn.run") as mock_run:
            main(["--config", str(cfg)])
        _, kwargs = mock_run.call_args
        assert kwargs["uds"] == socket_path
        assert "host" not in kwargs
        assert "port" not in kwargs

    def test_log_level_passed_to_uvicorn(self, tmp_path: Path) -> None:
        cfg = self._write_cfg(tmp_path, log_level="debug")
        with patch("meridiand.__main__.uvicorn.run") as mock_run:
            main(["--config", str(cfg)])
        _, kwargs = mock_run.call_args
        assert kwargs["log_level"] == "debug"

    def test_uvicorn_error_returns_nonzero(self, tmp_path: Path) -> None:
        cfg = self._write_cfg(tmp_path)
        with patch("meridiand.__main__.uvicorn.run", side_effect=RuntimeError("boom")):
            assert main(["--config", str(cfg)]) != 0

    def test_uvicorn_error_prints_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = self._write_cfg(tmp_path)
        with patch("meridiand.__main__.uvicorn.run", side_effect=RuntimeError("boom")):
            main(["--config", str(cfg)])
        assert "meridiand" in capsys.readouterr().err

    def test_uvicorn_error_writes_audit_log(self, tmp_path: Path) -> None:
        cfg = self._write_cfg(tmp_path)
        with patch("meridiand.__main__.uvicorn.run", side_effect=RuntimeError("boom")):
            main(["--config", str(cfg)])
        audit_path = tmp_path / "storage" / "audit.ndjson"
        assert audit_path.exists()
        record = json.loads(audit_path.read_text().strip())
        assert record["event"] == "meridiand.startup_failed"
        assert record["code"] == "startup_failed"
        assert record["level"] == "error"

    def test_uvicorn_error_audit_detail_contains_message(self, tmp_path: Path) -> None:
        cfg = self._write_cfg(tmp_path)
        with patch("meridiand.__main__.uvicorn.run", side_effect=RuntimeError("boom")):
            main(["--config", str(cfg)])
        record = json.loads((tmp_path / "storage" / "audit.ndjson").read_text().strip())
        assert record["detail"]["message"] == "boom"
