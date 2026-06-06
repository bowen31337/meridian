"""Tests for `meridian imports openclaw` and `meridian imports hermes`.

Covers the file/dir branches, JSON parsing failures, DaemonError handling,
and audit-log + click.echo side effects.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from meridian_cli.__main__ import cli
from meridian_cli._client import DaemonError


def _invoke(args: list[str], mock_client: MagicMock) -> object:
    runner = CliRunner()
    with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
        return runner.invoke(cli, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# openclaw — file mode
# ---------------------------------------------------------------------------


class TestImportOpenclawFile:
    def test_success(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "ex.json"
        f.write_text(json.dumps({"records": [{"id": "a"}]}))
        mock_client.request.return_value = {"imported": 1}
        result = _invoke(["imports", "openclaw", str(f)], mock_client)
        assert result.exit_code == 0
        mock_client.request.assert_called_once_with(
            "POST",
            "/v1/x/imports/openclaw",
            json_body={"records": [{"id": "a"}]},
        )
        assert "imported" in result.output

    def test_read_error_exits(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "ex.json"
        f.write_text("ok")  # exists for click's --exists check
        # Patch Path.read_text on instances to raise OSError
        with patch.object(Path, "read_text", side_effect=OSError("no read")):
            result = _invoke(["imports", "openclaw", str(f)], mock_client)
        assert result.exit_code == 1
        assert "import_read_failed" in result.output

    def test_invalid_json_exits(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "ex.json"
        f.write_text("not json {{{")
        result = _invoke(["imports", "openclaw", str(f)], mock_client)
        assert result.exit_code == 1
        assert "import_invalid_json" in result.output

    def test_daemon_error_exits(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "ex.json"
        f.write_text(json.dumps({"records": []}))
        mock_client.request.side_effect = DaemonError(code="boom", message="kaboom")
        result = _invoke(["imports", "openclaw", str(f)], mock_client)
        assert result.exit_code == 1
        assert "[boom] kaboom" in result.output

    def test_no_result_skips_echo(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "ex.json"
        f.write_text(json.dumps({"records": []}))
        mock_client.request.return_value = None
        result = _invoke(["imports", "openclaw", str(f)], mock_client)
        assert result.exit_code == 0
        assert result.output.strip() == ""


# ---------------------------------------------------------------------------
# openclaw — directory mode
# ---------------------------------------------------------------------------


class TestImportOpenclawDir:
    def test_full_install(self, tmp_path: Path, mock_client: MagicMock) -> None:
        (tmp_path / "channels.json").write_text(json.dumps({"records": [1]}))
        (tmp_path / "sessions.json").write_text(json.dumps({"records": [2]}))
        (tmp_path / "tools.json").write_text(json.dumps({"records": [3]}))
        (tmp_path / "MEMORY.md").write_text("hello memory")
        mock_client.request.return_value = {"ok": True}

        result = _invoke(["imports", "openclaw", str(tmp_path)], mock_client)
        assert result.exit_code == 0
        called_args = mock_client.request.call_args
        assert called_args[0] == ("POST", "/v1/x/imports/openclaw/install")
        body = called_args[1]["json_body"]
        assert body["channels"] == [1]
        assert body["sessions"] == [2]
        assert body["tools"] == [3]
        assert body["memory"][0]["key"] == "MEMORY.md"
        assert body["memory"][0]["content"] == "hello memory"

    def test_missing_subsystem_files_default_to_empty(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        # only channels.json + MEMORY.md
        (tmp_path / "channels.json").write_text(json.dumps({"records": [1]}))
        mock_client.request.return_value = None
        result = _invoke(["imports", "openclaw", str(tmp_path)], mock_client)
        assert result.exit_code == 0
        body = mock_client.request.call_args[1]["json_body"]
        assert body["sessions"] == []
        assert body["tools"] == []
        assert body["memory"] == []

    def test_invalid_subsystem_json_exits(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        (tmp_path / "channels.json").write_text(json.dumps({"missing": "records"}))
        result = _invoke(["imports", "openclaw", str(tmp_path)], mock_client)
        assert result.exit_code == 1
        assert "import_invalid_json" in result.output

    def test_subsystem_json_parse_error_exits(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        (tmp_path / "channels.json").write_text("not json {{{")
        result = _invoke(["imports", "openclaw", str(tmp_path)], mock_client)
        assert result.exit_code == 1
        assert "import_invalid_json" in result.output

    def test_memory_read_error_exits(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        (tmp_path / "MEMORY.md").write_text("ok")
        real_read = Path.read_text

        def _selective(self: Path, *a, **k) -> str:
            if self.name == "MEMORY.md":
                raise OSError("denied")
            return real_read(self, *a, **k)

        with patch.object(Path, "read_text", _selective):
            result = _invoke(["imports", "openclaw", str(tmp_path)], mock_client)
        assert result.exit_code == 1
        assert "import_read_failed" in result.output

    def test_empty_dir_warns_but_still_posts(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        mock_client.request.return_value = None
        result = _invoke(["imports", "openclaw", str(tmp_path)], mock_client)
        assert result.exit_code == 0
        assert "warning: no OpenClaw subsystem files found" in result.output

    def test_daemon_error_exits(self, tmp_path: Path, mock_client: MagicMock) -> None:
        (tmp_path / "channels.json").write_text(json.dumps({"records": []}))
        mock_client.request.side_effect = DaemonError(code="db_fail", message="bad")
        result = _invoke(["imports", "openclaw", str(tmp_path)], mock_client)
        assert result.exit_code == 1
        assert "[db_fail] bad" in result.output

    def test_result_echoed(self, tmp_path: Path, mock_client: MagicMock) -> None:
        (tmp_path / "channels.json").write_text(json.dumps({"records": []}))
        mock_client.request.return_value = {"imported": 5}
        result = _invoke(["imports", "openclaw", str(tmp_path)], mock_client)
        assert result.exit_code == 0
        assert "imported" in result.output


# ---------------------------------------------------------------------------
# openclaw — neither file nor dir (e.g. broken symlink, fifo)
# ---------------------------------------------------------------------------


def test_openclaw_path_neither_file_nor_dir(
    tmp_path: Path, mock_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force is_file and is_dir to both return False (e.g. socket/fifo)."""
    f = tmp_path / "weird"
    f.write_text("")  # exists for click's check
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    result = _invoke(["imports", "openclaw", str(f)], mock_client)
    assert result.exit_code == 1
    assert "import_path_invalid" in result.output


# ---------------------------------------------------------------------------
# hermes — file mode
# ---------------------------------------------------------------------------


class TestImportHermesFile:
    def test_success(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "ex.json"
        f.write_text(json.dumps({"records": [{"id": "s1"}]}))
        mock_client.request.return_value = {"ok": True}
        result = _invoke(["imports", "hermes", str(f)], mock_client)
        assert result.exit_code == 0
        mock_client.request.assert_called_once_with(
            "POST",
            "/v1/x/imports/hermes",
            json_body={"records": [{"id": "s1"}]},
        )

    def test_read_error_exits(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "ex.json"
        f.write_text("ok")
        with patch.object(Path, "read_text", side_effect=OSError("no read")):
            result = _invoke(["imports", "hermes", str(f)], mock_client)
        assert result.exit_code == 1
        assert "import_read_failed" in result.output

    def test_invalid_json_exits(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "ex.json"
        f.write_text("not json {{{")
        result = _invoke(["imports", "hermes", str(f)], mock_client)
        assert result.exit_code == 1
        assert "import_invalid_json" in result.output

    def test_daemon_error_exits(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "ex.json"
        f.write_text(json.dumps({"records": []}))
        mock_client.request.side_effect = DaemonError(code="x", message="y")
        result = _invoke(["imports", "hermes", str(f)], mock_client)
        assert result.exit_code == 1
        assert "[x] y" in result.output

    def test_none_result_no_echo(self, tmp_path: Path, mock_client: MagicMock) -> None:
        f = tmp_path / "ex.json"
        f.write_text(json.dumps({"records": []}))
        mock_client.request.return_value = None
        result = _invoke(["imports", "hermes", str(f)], mock_client)
        assert result.exit_code == 0
        assert result.output.strip() == ""


# ---------------------------------------------------------------------------
# hermes — directory mode
# ---------------------------------------------------------------------------


class TestImportHermesDir:
    def test_full_install(self, tmp_path: Path, mock_client: MagicMock) -> None:
        for fname in (
            "skills.json",
            "environments.json",
            "providers.json",
            "sessions.json",
            "user_profiles.json",
            "cron.json",
            "acp_registry.json",
        ):
            (tmp_path / fname).write_text(json.dumps({"records": [{"x": fname}]}))
        mock_client.request.return_value = {"ok": True}

        result = _invoke(["imports", "hermes", str(tmp_path)], mock_client)
        assert result.exit_code == 0
        body = mock_client.request.call_args[1]["json_body"]
        assert body["skills"][0] == {"x": "skills.json"}
        assert body["cron"][0] == {"x": "cron.json"}

    def test_missing_files_default_empty(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        (tmp_path / "skills.json").write_text(json.dumps({"records": [1]}))
        mock_client.request.return_value = None
        result = _invoke(["imports", "hermes", str(tmp_path)], mock_client)
        assert result.exit_code == 0
        body = mock_client.request.call_args[1]["json_body"]
        assert body["skills"] == [1]
        assert body["environments"] == []

    def test_invalid_subsystem_json_exits(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        (tmp_path / "skills.json").write_text(json.dumps({"missing": "records"}))
        result = _invoke(["imports", "hermes", str(tmp_path)], mock_client)
        assert result.exit_code == 1
        assert "import_invalid_json" in result.output

    def test_empty_dir_warns(self, tmp_path: Path, mock_client: MagicMock) -> None:
        mock_client.request.return_value = None
        result = _invoke(["imports", "hermes", str(tmp_path)], mock_client)
        assert result.exit_code == 0
        assert "warning: no Hermes subsystem files found" in result.output

    def test_daemon_error_exits(self, tmp_path: Path, mock_client: MagicMock) -> None:
        (tmp_path / "skills.json").write_text(json.dumps({"records": []}))
        mock_client.request.side_effect = DaemonError(code="z", message="bad")
        result = _invoke(["imports", "hermes", str(tmp_path)], mock_client)
        assert result.exit_code == 1
        assert "[z] bad" in result.output

    def test_result_echoed(self, tmp_path: Path, mock_client: MagicMock) -> None:
        (tmp_path / "skills.json").write_text(json.dumps({"records": []}))
        mock_client.request.return_value = {"imported": 5}
        result = _invoke(["imports", "hermes", str(tmp_path)], mock_client)
        assert result.exit_code == 0
        assert "imported" in result.output


def test_hermes_path_neither_file_nor_dir(
    tmp_path: Path, mock_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "weird"
    f.write_text("")
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    result = _invoke(["imports", "hermes", str(f)], mock_client)
    assert result.exit_code == 1
    assert "import_path_invalid" in result.output
