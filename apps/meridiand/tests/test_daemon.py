"""
Daemon startup conformance suite.

Tests cover:
  - main() emits OTel span "daemon.start" on every invocation.
  - Span carries "daemon.bind_mode" attribute.
  - Span carries "daemon.socket" attribute when binding via socket.
  - Span carries "daemon.host" / "daemon.port" attributes.
  - Span contains "meridian.error.invocation" structured event with code "daemon_start".
  - Span contains "daemon.start" structured event with bind attributes.
  - Socket parent directory is created before binding.
  - Binding via TCP (socket=null) uses loopback :8888 by default.
  - main() returns 0 on successful startup.
  - uvicorn failure writes audit log entry with event "meridiand.startup_failed".
  - uvicorn failure sets span status to ERROR.
  - uvicorn failure emits "daemon.start.error" span event.
  - uvicorn failure returns 1.
  - uvicorn failure prints message to stderr.
"""

from __future__ import annotations

import json
from pathlib import Path

from opentelemetry.trace import StatusCode
import uvicorn
import yaml

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_cfg(tmp_path: Path, **extra: object) -> Path:
    """Write a minimal config YAML and return its path."""
    cfg = tmp_path / "config.yaml"
    data: dict[str, object] = {
        "storage_root": str(tmp_path / "storage"),
        # Use a socket path inside tmp_path so tests are hermetic.
        "bind": {"socket": str(tmp_path / "meridiand.sock")},
    }
    data.update(extra)
    cfg.write_text(yaml.dump(data))
    return cfg


def _write_tcp_cfg(tmp_path: Path) -> Path:
    """Config that uses TCP loopback binding (socket=null)."""
    cfg = tmp_path / "config.yaml"
    data: dict[str, object] = {
        "storage_root": str(tmp_path / "storage"),
        "bind": {"socket": None, "host": "127.0.0.1", "port": 8888},
    }
    cfg.write_text(yaml.dump(data))
    return cfg


def _read_audit(tmp_path: Path) -> list[dict]:
    audit_path = tmp_path / "storage" / "audit.ndjson"
    if not audit_path.exists():
        return []
    return [json.loads(line) for line in audit_path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# TestDaemonStartupSpan
# ---------------------------------------------------------------------------


class TestDaemonStartupSpan:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_emits_daemon_start_span(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

        from meridiand.__main__ import main

        assert main(["--config", str(cfg)]) == 0
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "daemon.start" in span_names

    def test_span_has_bind_mode_socket(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        assert span.attributes["daemon.bind_mode"] == "socket"

    def test_span_has_daemon_socket_attribute(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        assert span.attributes["daemon.socket"] == str(tmp_path / "meridiand.sock")

    def test_span_has_bind_mode_tcp(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_tcp_cfg(tmp_path)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        assert span.attributes["daemon.bind_mode"] == "tcp"

    def test_span_has_host_attribute(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_tcp_cfg(tmp_path)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        assert span.attributes["daemon.host"] == "127.0.0.1"

    def test_span_has_port_attribute(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_tcp_cfg(tmp_path)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        assert span.attributes["daemon.port"] == 8888

    def test_span_has_invocation_event(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_invocation_event_has_code_daemon_start(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        evt = next(e for e in span.events if e.name == "meridian.error.invocation")
        assert evt.attributes["code"] == "daemon_start"

    def test_span_has_daemon_start_event(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        event_names = [e.name for e in span.events]
        assert "daemon.start" in event_names

    def test_daemon_start_event_has_bind_mode(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        evt = next(e for e in span.events if e.name == "daemon.start")
        assert evt.attributes["daemon.bind_mode"] == "socket"


# ---------------------------------------------------------------------------
# TestDaemonSocketSetup
# ---------------------------------------------------------------------------


class TestDaemonSocketSetup:
    def test_socket_parent_dir_created(self, tmp_path: Path, monkeypatch) -> None:
        socket_path = tmp_path / "run" / "nested" / "meridiand.sock"
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "storage_root": str(tmp_path / "storage"),
                    "bind": {"socket": str(socket_path)},
                }
            )
        )
        captured: dict[str, object] = {}

        def _mock_run(app, **kwargs):
            captured["uds"] = kwargs.get("uds")

        monkeypatch.setattr(uvicorn, "run", _mock_run)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        assert socket_path.parent.exists()

    def test_uvicorn_receives_uds_kwarg(self, tmp_path: Path, monkeypatch) -> None:
        socket_path = tmp_path / "meridiand.sock"
        cfg = _write_cfg(tmp_path)
        captured: dict[str, object] = {}

        def _mock_run(app, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(uvicorn, "run", _mock_run)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        assert captured.get("uds") == str(socket_path)

    def test_tcp_binding_uses_loopback_8888(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_tcp_cfg(tmp_path)
        captured: dict[str, object] = {}

        def _mock_run(app, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(uvicorn, "run", _mock_run)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        assert captured.get("host") == "127.0.0.1"
        assert captured.get("port") == 8888

    def test_tcp_binding_has_no_uds_kwarg(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_tcp_cfg(tmp_path)
        captured: dict[str, object] = {}

        def _mock_run(app, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(uvicorn, "run", _mock_run)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        assert "uds" not in captured


# ---------------------------------------------------------------------------
# TestDaemonStartupFailure
# ---------------------------------------------------------------------------


class TestDaemonStartupFailure:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_failure_returns_nonzero(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)

        def _fail(*a, **kw):
            raise RuntimeError("bind failed")

        monkeypatch.setattr(uvicorn, "run", _fail)

        from meridiand.__main__ import main

        assert main(["--config", str(cfg)]) == 1

    def test_failure_writes_audit_log(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)

        def _fail(*a, **kw):
            raise RuntimeError("bind failed")

        monkeypatch.setattr(uvicorn, "run", _fail)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        entries = _read_audit(tmp_path)
        assert any(e["event"] == "meridiand.startup_failed" for e in entries)

    def test_failure_audit_level_is_error(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)

        def _fail(*a, **kw):
            raise RuntimeError("bind failed")

        monkeypatch.setattr(uvicorn, "run", _fail)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        entries = _read_audit(tmp_path)
        entry = next(e for e in entries if e["event"] == "meridiand.startup_failed")
        assert entry["level"] == "error"

    def test_failure_audit_detail_has_message(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)

        def _fail(*a, **kw):
            raise RuntimeError("bind failed")

        monkeypatch.setattr(uvicorn, "run", _fail)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        entries = _read_audit(tmp_path)
        entry = next(e for e in entries if e["event"] == "meridiand.startup_failed")
        assert "bind failed" in entry["detail"]["message"]

    def test_failure_sets_span_error_status(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)

        def _fail(*a, **kw):
            raise RuntimeError("bind failed")

        monkeypatch.setattr(uvicorn, "run", _fail)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        assert span.status.status_code == StatusCode.ERROR

    def test_failure_emits_daemon_start_error_event(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _write_cfg(tmp_path)

        def _fail(*a, **kw):
            raise RuntimeError("bind failed")

        monkeypatch.setattr(uvicorn, "run", _fail)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "daemon.start")
        event_names = [e.name for e in span.events]
        assert "daemon.start.error" in event_names

    def test_failure_prints_to_stderr(self, tmp_path: Path, monkeypatch, capsys) -> None:
        cfg = _write_cfg(tmp_path)

        def _fail(*a, **kw):
            raise RuntimeError("bind failed")

        monkeypatch.setattr(uvicorn, "run", _fail)

        from meridiand.__main__ import main

        main(["--config", str(cfg)])
        captured = capsys.readouterr()
        assert "bind failed" in captured.err
