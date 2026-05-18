"""Tests for all CRUD resource subcommands.

Each resource is tested for the same five invariants:
  1. list  — calls GET  /v1/x/<resource>
  2. get   — calls GET  /v1/x/<resource>/<id>
  3. create — calls POST /v1/x/<resource> with parsed JSON body
  4. update — calls PATCH /v1/x/<resource>/<id> with parsed JSON body
  5. delete — calls DELETE /v1/x/<resource>/<id>

Failure paths verify that the error is written to the audit log, printed to
stderr, and causes a non-zero exit code.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from meridian_cli.__main__ import cli
from meridian_cli._client import DaemonError

_RESOURCES = [
    "agents",
    "sessions",
    "skills",
    "environments",
    "channels",
    "vaults",
    "memory_stores",
    "user_profiles",
    "webhooks",
    "files",
    "hooks",
    "cron",
]

_SAMPLE_RESPONSE = {"id": "abc123", "name": "test"}
_SAMPLE_DATA = json.dumps({"name": "new-resource"})


def _invoke(args: list[str], mock_client: MagicMock) -> object:
    """Run the CLI with a pre-built mock client injected as ctx.obj."""
    runner = CliRunner()
    # Patch client_from_env so cli() sets ctx.obj = mock_client
    with patch("meridian_cli.__main__.client_from_env", return_value=mock_client):
        return runner.invoke(cli, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# Parametrized success-path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resource", _RESOURCES)
def test_list_calls_get(resource: str, mock_client: MagicMock) -> None:
    mock_client.request.return_value = [_SAMPLE_RESPONSE]
    result = _invoke([resource, "list"], mock_client)
    assert result.exit_code == 0
    mock_client.request.assert_called_once_with("GET", f"/v1/x/{resource}", json_body=None)


@pytest.mark.parametrize("resource", _RESOURCES)
def test_get_calls_get_with_id(resource: str, mock_client: MagicMock) -> None:
    result = _invoke([resource, "get", "abc123"], mock_client)
    assert result.exit_code == 0
    mock_client.request.assert_called_once_with("GET", f"/v1/x/{resource}/abc123", json_body=None)


@pytest.mark.parametrize("resource", _RESOURCES)
def test_create_calls_post_with_body(resource: str, mock_client: MagicMock) -> None:
    result = _invoke([resource, "create", "--data", _SAMPLE_DATA], mock_client)
    assert result.exit_code == 0
    mock_client.request.assert_called_once_with(
        "POST", f"/v1/x/{resource}", json_body={"name": "new-resource"}
    )


@pytest.mark.parametrize("resource", _RESOURCES)
def test_update_calls_patch_with_id_and_body(resource: str, mock_client: MagicMock) -> None:
    result = _invoke([resource, "update", "abc123", "--data", _SAMPLE_DATA], mock_client)
    assert result.exit_code == 0
    mock_client.request.assert_called_once_with(
        "PATCH", f"/v1/x/{resource}/abc123", json_body={"name": "new-resource"}
    )


@pytest.mark.parametrize("resource", _RESOURCES)
def test_delete_calls_delete_with_id(resource: str, mock_client: MagicMock) -> None:
    mock_client.request.return_value = None
    result = _invoke([resource, "delete", "abc123"], mock_client)
    assert result.exit_code == 0
    mock_client.request.assert_called_once_with("DELETE", f"/v1/x/{resource}/abc123", json_body=None)


# ---------------------------------------------------------------------------
# Failure-path tests — daemon error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resource", _RESOURCES)
def test_list_daemon_error_exits_nonzero(resource: str, mock_client: MagicMock, tmp_path: Path) -> None:
    mock_client.request.side_effect = DaemonError(code="daemon_unreachable", message="no daemon")
    audit_log = tmp_path / "audit.ndjson"

    with patch("meridian_cli._resource.write_audit") as mock_audit:
        result = _invoke([resource, "list"], mock_client)

    assert result.exit_code != 0
    assert "daemon_unreachable" in result.output or "no daemon" in result.output
    mock_audit.assert_any_call(
        "error",
        f"{resource}.list.failed",
        {"code": "daemon_unreachable", "message": "no daemon"},
    )


@pytest.mark.parametrize("resource", _RESOURCES)
def test_create_invalid_json_exits_nonzero(resource: str, mock_client: MagicMock) -> None:
    result = _invoke([resource, "create", "--data", "not-json"], mock_client)
    assert result.exit_code != 0
    mock_client.request.assert_not_called()


# ---------------------------------------------------------------------------
# Output format test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resource", _RESOURCES)
def test_get_outputs_json(resource: str, mock_client: MagicMock) -> None:
    mock_client.request.return_value = _SAMPLE_RESPONSE
    result = _invoke([resource, "get", "abc123"], mock_client)
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed == _SAMPLE_RESPONSE


# ---------------------------------------------------------------------------
# Audit log written on invocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resource", _RESOURCES)
def test_list_writes_invocation_audit(resource: str, mock_client: MagicMock) -> None:
    mock_client.request.return_value = []
    with patch("meridian_cli._resource.write_audit") as mock_audit:
        result = _invoke([resource, "list"], mock_client)
    assert result.exit_code == 0
    mock_audit.assert_any_call(
        "info",
        f"{resource}.list.invoked",
        {"resource": resource, "operation": "list"},
    )


# ---------------------------------------------------------------------------
# files upload command
# ---------------------------------------------------------------------------


def test_files_upload_sends_content(mock_client: MagicMock, tmp_path: Path) -> None:
    upload_file = tmp_path / "hello.txt"
    upload_file.write_bytes(b"hello world")
    mock_client.request.return_value = {"id": "file-1", "name": "hello.txt"}

    result = _invoke(["files", "upload", str(upload_file)], mock_client)

    assert result.exit_code == 0
    # First call creates the file record
    first_call = mock_client.request.call_args_list[0]
    assert first_call.args[0] == "POST"
    assert first_call.args[1] == "/v1/x/files"
    assert first_call.kwargs.get("json_body", {}).get("name") == "hello.txt"
    # Second call streams the content
    second_call = mock_client.request.call_args_list[1]
    assert second_call.args[0] == "PUT"
    assert second_call.kwargs.get("content") == b"hello world"


def test_files_upload_with_name_override(mock_client: MagicMock, tmp_path: Path) -> None:
    upload_file = tmp_path / "hello.txt"
    upload_file.write_bytes(b"data")
    mock_client.request.return_value = {"id": "file-2", "name": "custom.txt"}

    result = _invoke(["files", "upload", str(upload_file), "--name", "custom.txt"], mock_client)
    assert result.exit_code == 0
    first_call = mock_client.request.call_args_list[0]
    assert first_call.kwargs.get("json_body", {}).get("name") == "custom.txt"
