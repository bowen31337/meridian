"""Tests covering the remaining small gaps across __main__, _audit, _resource,
_telemetry, files, sessions, workspace, and meridianconfig.
"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from meridian_cli.__main__ import cli, main
from meridian_cli._audit import write_audit
from meridian_cli._client import DaemonError
from meridian_cli._resource import _client
from meridian_cli._telemetry import record_invocation_event


# ---------------------------------------------------------------------------
# __main__.workspace_init + main + __main__ block
# ---------------------------------------------------------------------------


class TestWorkspaceInitCommand:
    def test_workspace_init_success(self, mock_client: MagicMock, tmp_path: Path) -> None:
        runner = CliRunner()
        with (
            patch("meridian_cli.__main__.client_from_env", return_value=mock_client),
            patch("meridian_cli.__main__.UvWorkspaceInitializer") as mock_init_cls,
        ):
            mock_init = MagicMock()
            mock_init_cls.return_value = mock_init
            result = runner.invoke(cli, ["workspace-init", "--root", str(tmp_path)])
        assert result.exit_code == 0
        mock_init.init.assert_called_once()

    def test_workspace_init_error_exits_nonzero(
        self, mock_client: MagicMock, tmp_path: Path
    ) -> None:
        from meridian_cli.workspace import WorkspaceError

        runner = CliRunner()
        with (
            patch("meridian_cli.__main__.client_from_env", return_value=mock_client),
            patch("meridian_cli.__main__.UvWorkspaceInitializer") as mock_init_cls,
        ):
            mock_init = MagicMock()
            mock_init.init.side_effect = WorkspaceError(code="nope", message="nope")
            mock_init_cls.return_value = mock_init
            result = runner.invoke(cli, ["workspace-init", "--root", str(tmp_path)])
        assert result.exit_code == 1


def test_main_function_invokes_cli(mock_client: MagicMock) -> None:
    """main() calls cli() — verify by patching cli and calling main()."""
    with patch("meridian_cli.__main__.cli") as mock_cli:
        main()
        mock_cli.assert_called_once()


def test_module_dunder_main_block() -> None:
    """Execute __main__.py with __name__='__main__' to cover the if-name-main block."""
    from meridian_cli import __main__ as main_mod

    src_path = Path(main_mod.__file__)
    code = compile(src_path.read_text(), str(src_path), "exec")
    called: list[int] = []

    # The exec'd code defines its own `main()` which calls cli().
    # Inject a no-op cli into the namespace so main() returns immediately.
    class _NoopCli:
        def __call__(self, *_a, **_k) -> None:
            called.append(1)

        def add_command(self, *_a, **_k) -> None:
            pass

        def command(self, *_a, **_k):
            def deco(f):
                return f

            return deco

    ns: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(src_path),
        "__package__": "meridian_cli",
        "__loader__": None,
    }
    # Replace click.group so cli() at module load time uses our noop
    import click as _click

    real_group = _click.group

    def _fake_group(*a, **kw):
        def deco(f):
            return _NoopCli()

        return deco

    _click.group = _fake_group  # type: ignore[assignment]
    try:
        exec(code, ns)
    finally:
        _click.group = real_group  # type: ignore[assignment]
    assert called == [1]


# ---------------------------------------------------------------------------
# _audit.write_audit — detail=None vs detail=dict (branch 20->22)
# ---------------------------------------------------------------------------


class TestAuditWrite:
    def test_write_audit_with_detail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("meridian_cli._audit.MERIDIAN_DIR", tmp_path)
        log_path = tmp_path / "audit.ndjson"
        monkeypatch.setattr("meridian_cli._audit.AUDIT_LOG", log_path)
        write_audit("info", "evt", {"k": "v"})
        line = log_path.read_text().strip()
        entry = json.loads(line)
        assert entry["detail"] == {"k": "v"}

    def test_write_audit_no_detail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("meridian_cli._audit.MERIDIAN_DIR", tmp_path)
        log_path = tmp_path / "audit.ndjson"
        monkeypatch.setattr("meridian_cli._audit.AUDIT_LOG", log_path)
        write_audit("info", "evt")
        line = log_path.read_text().strip()
        entry = json.loads(line)
        assert "detail" not in entry

    def test_write_audit_empty_detail_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("meridian_cli._audit.MERIDIAN_DIR", tmp_path)
        log_path = tmp_path / "audit.ndjson"
        monkeypatch.setattr("meridian_cli._audit.AUDIT_LOG", log_path)
        write_audit("info", "evt", {})
        line = log_path.read_text().strip()
        entry = json.loads(line)
        # falsy dict skipped
        assert "detail" not in entry


# ---------------------------------------------------------------------------
# _resource: _update with invalid JSON for --data (lines 121-123)
# ---------------------------------------------------------------------------


def test_update_invalid_json_exits(mock_client: MagicMock) -> None:
    runner = CliRunner()
    with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
        result = runner.invoke(cli, ["agents", "update", "id1", "--data", "not json {{{"])
    assert result.exit_code == 1
    assert "invalid JSON" in result.output


# ---------------------------------------------------------------------------
# _telemetry.record_invocation_event — branch 18->17 (non-primitive value skipped)
# ---------------------------------------------------------------------------


def test_record_invocation_event_skips_non_primitive() -> None:
    span = MagicMock()
    record_invocation_event(
        span,
        {
            "str": "s",
            "int": 1,
            "float": 1.5,
            "bool": True,
            "list": [1, 2],  # skipped
            "dict": {"x": 1},  # skipped
            "none": None,  # skipped
        },
    )
    span.add_event.assert_called_once()
    attrs = span.add_event.call_args[0][1]
    assert "list" not in attrs
    assert "dict" not in attrs
    assert "none" not in attrs
    assert attrs["str"] == "s"


# ---------------------------------------------------------------------------
# files.upload — full coverage
# ---------------------------------------------------------------------------


class TestFilesUpload:
    def test_upload_success_with_id(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"\x00\x01\x02hello")
        mock_client.request.return_value = {"id": "file-1", "name": "x.bin"}

        runner = CliRunner()
        with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
            result = runner.invoke(cli, ["files", "upload", str(f)])
        assert result.exit_code == 0
        # First call POST metadata, second call PUT content
        calls = mock_client.request.call_args_list
        assert len(calls) == 2
        assert calls[0][0] == ("POST", "/v1/x/files")
        assert calls[1][0] == ("PUT", "/v1/x/files/file-1/content")
        assert calls[1][1]["content"] == b"\x00\x01\x02hello"

    def test_upload_with_custom_name(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"data")
        mock_client.request.return_value = {"id": "f1"}
        runner = CliRunner()
        with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
            result = runner.invoke(cli, ["files", "upload", str(f), "--name", "renamed"])
        assert result.exit_code == 0
        first_body = mock_client.request.call_args_list[0][1]["json_body"]
        assert first_body["name"] == "renamed"

    def test_upload_metadata_no_id_skips_content_put(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"d")
        mock_client.request.return_value = {"no_id": True}
        runner = CliRunner()
        with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
            result = runner.invoke(cli, ["files", "upload", str(f)])
        assert result.exit_code == 0
        # only the POST call should have been made
        assert mock_client.request.call_count == 1

    def test_upload_daemon_error(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"d")
        mock_client.request.side_effect = DaemonError(code="x", message="boom")
        runner = CliRunner()
        with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
            result = runner.invoke(cli, ["files", "upload", str(f)])
        assert result.exit_code == 1
        assert "boom" in result.output

    def test_upload_no_result_no_echo(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"d")
        mock_client.request.return_value = None
        runner = CliRunner()
        with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
            result = runner.invoke(cli, ["files", "upload", str(f)])
        assert result.exit_code == 0
        # No JSON should be printed
        assert "{" not in result.output


# ---------------------------------------------------------------------------
# sessions.archive + sessions.restore (lines 15, 30)
# ---------------------------------------------------------------------------


class TestSessionsExtras:
    def test_sessions_archive(self, mock_client: MagicMock) -> None:
        mock_client.request.return_value = {"archived": True}
        runner = CliRunner()
        with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
            result = runner.invoke(cli, ["sessions", "archive", "s1"])
        assert result.exit_code == 0
        mock_client.request.assert_called_once_with(
            "POST", "/v1/x/sessions/s1/archive", json_body=None
        )

    def test_sessions_restore(self, mock_client: MagicMock) -> None:
        mock_client.request.return_value = {"restored": True}
        runner = CliRunner()
        with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
            result = runner.invoke(cli, ["sessions", "restore", "s1"])
        assert result.exit_code == 0
        mock_client.request.assert_called_once_with(
            "POST", "/v1/x/sessions/s1/restore", json_body=None
        )


# ---------------------------------------------------------------------------
# workspace.py line 56->58 branch  (audit when detail is None)
# ---------------------------------------------------------------------------


def test_workspace_audit_helper_no_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """workspace._audit with detail=None skips the detail key."""
    from meridian_cli import workspace as ws

    monkeypatch.setattr(ws, "AUDIT_DIR", tmp_path)
    log_path = tmp_path / "audit.ndjson"
    monkeypatch.setattr(ws, "AUDIT_LOG", log_path)
    ws._audit("info", "evt", None)
    entry = json.loads(log_path.read_text().strip())
    assert "detail" not in entry


# ---------------------------------------------------------------------------
# meridianconfig: socket=None branch (lines 37-39), bind port out of range (241),
# migrate path with unknown upgrade (321-322), bind log_level invalid, etc.
# ---------------------------------------------------------------------------


class TestMeridianconfigGaps:
    def test_bind_socket_none_returns_none(self) -> None:
        from meridian_cli.meridianconfig import _BindConfig

        c = _BindConfig(socket=None)
        assert c.socket is None

    def test_bind_socket_expands_home(self) -> None:
        from meridian_cli.meridianconfig import _BindConfig

        c = _BindConfig(socket="~/x.sock")
        assert "~" not in c.socket  # type: ignore[operator]

    def test_validate_invalid_port_yields_error(self, tmp_path: Path) -> None:
        from meridian_cli.meridianconfig import _CONFIG_VERSION

        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "version": _CONFIG_VERSION,
                    "storage_root": "/tmp/m",
                    "daemon": {
                        "log_level": "info",
                        "bind": {"host": "127.0.0.1", "port": 70000},  # out of range
                    },
                }
            )
        )
        runner = CliRunner()
        with patch("meridian_cli.__main__.client_from_env"):
            result = runner.invoke(cli, ["meridianconfig", "validate", "--config", str(cfg_path)])
        assert result.exit_code == 1
        assert "not in range" in result.output

    def test_migrate_no_upgrade_path(self, tmp_path: Path) -> None:
        """Patch _UPGRADES to be empty so migrate cannot proceed."""
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(yaml.dump({"version": 1}))
        runner = CliRunner()
        with (
            patch("meridian_cli.__main__.client_from_env"),
            patch("meridian_cli.meridianconfig._UPGRADES", {}),
        ):
            result = runner.invoke(cli, ["meridianconfig", "migrate", "--config", str(cfg_path)])
        assert result.exit_code == 1
        assert "no upgrade path" in result.output

    def test_daemon_workspace_root_expands_home(self) -> None:
        from meridian_cli.meridianconfig import _DaemonConfig

        c = _DaemonConfig(workspace_root=Path("~/meridian-test"))
        assert "~" not in str(c.workspace_root)

    def test_migrate_invalid_log_level(self, tmp_path: Path) -> None:
        """validate flags an unknown log level."""
        from meridian_cli.meridianconfig import _CONFIG_VERSION

        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "version": _CONFIG_VERSION,
                    "storage_root": "/tmp/m",
                    "daemon": {"log_level": "BOGUS"},
                }
            )
        )
        runner = CliRunner()
        with patch("meridian_cli.__main__.client_from_env"):
            result = runner.invoke(cli, ["meridianconfig", "validate", "--config", str(cfg_path)])
        assert result.exit_code == 1
        assert "log_level" in result.output
