"""Tests for the uv workspace initializer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meridian_cli.workspace import UvWorkspaceInitializer, WorkspaceError

# ---------------------------------------------------------------------------
# OTel mock — controllable span; no SDK bootstrap needed.
# ---------------------------------------------------------------------------
_mock_span = MagicMock()


@pytest.fixture(autouse=True)
def _otel_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = lambda *_: _mock_span
    tracer.start_as_current_span.return_value.__exit__ = lambda *_: False
    monkeypatch.setattr("meridian_cli.workspace.get_tracer", lambda: tracer)
    _mock_span.reset_mock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_workspace_root(tmp_path: Path, members: list[str]) -> Path:
    for member in members:
        (tmp_path / member).mkdir(parents=True)
        (tmp_path / member / "pyproject.toml").write_text("[project]\nname = 'stub'\n")
    return tmp_path


_ALL_MEMBERS = [
    "apps/meridiand",
    "apps/meridian-cli",
    "packages/core-errors",
    "packages/knowledge-base-indexer",
    "packages/sdk-capabilities",
    "packages/sdk-channel",
    "packages/sdk-environment",
    "packages/sdk-provider",
    "packages/sdk-sandbox",
    "packages/sdk-tool",
    "packages/storage-blob",
    "packages/storage-event-log",
    "packages/storage-reposit",
    "packages/storage-repository",
    "packages/system-ulid",
]


# ---------------------------------------------------------------------------
# Tests: success path
# ---------------------------------------------------------------------------


def test_init_emits_invocation_event(tmp_path: Path) -> None:
    root = make_workspace_root(tmp_path, _ALL_MEMBERS)
    with patch("meridian_cli.workspace.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        UvWorkspaceInitializer(repo_root=root).init()

    _mock_span.add_event.assert_any_call(
        "meridian.cli.invocation",
        {"event.name": "workspace.invocation", "workspace.operation": "init", "workspace.root": str(root)},
    )


def test_init_emits_completed_event(tmp_path: Path) -> None:
    root = make_workspace_root(tmp_path, _ALL_MEMBERS)
    with patch("meridian_cli.workspace.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        UvWorkspaceInitializer(repo_root=root).init()

    _mock_span.add_event.assert_any_call("workspace.init.completed")


def test_init_writes_audit_log(tmp_path: Path) -> None:
    root = make_workspace_root(tmp_path, _ALL_MEMBERS)
    with patch("meridian_cli.workspace.subprocess.run") as mock_run, patch(
        "meridian_cli.workspace.AUDIT_DIR", tmp_path / ".meridian"
    ), patch("meridian_cli.workspace.AUDIT_LOG", tmp_path / ".meridian" / "workspace-audit.ndjson"):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        UvWorkspaceInitializer(repo_root=root).init()

    log = tmp_path / ".meridian" / "workspace-audit.ndjson"
    assert log.exists()
    lines = [line for line in log.read_text().splitlines() if line.strip()]
    assert any("workspace.init.ok" in line for line in lines)


# ---------------------------------------------------------------------------
# Tests: missing member failure
# ---------------------------------------------------------------------------


def test_init_raises_on_missing_member(tmp_path: Path) -> None:
    # Omit apps/meridiand
    members = [m for m in _ALL_MEMBERS if m != "apps/meridiand"]
    root = make_workspace_root(tmp_path, members)

    errors: list[WorkspaceError] = []
    with pytest.raises(WorkspaceError) as exc_info:
        UvWorkspaceInitializer(repo_root=root, on_error=errors.append).init()

    assert exc_info.value.code == "WORKSPACE_MISSING_MEMBERS"
    assert "apps/meridiand" in exc_info.value.message
    assert len(errors) == 1


def test_init_sets_span_to_error_on_missing_member(tmp_path: Path) -> None:
    members = [m for m in _ALL_MEMBERS if m != "apps/meridian-cli"]
    root = make_workspace_root(tmp_path, members)

    with pytest.raises(WorkspaceError):
        UvWorkspaceInitializer(repo_root=root).init()

    _mock_span.set_status.assert_called_once()
    call_args = _mock_span.set_status.call_args[0][0]
    from opentelemetry.trace import StatusCode

    assert call_args.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Tests: uv sync failure
# ---------------------------------------------------------------------------


def test_init_raises_on_uv_sync_failure(tmp_path: Path) -> None:
    root = make_workspace_root(tmp_path, _ALL_MEMBERS)
    with patch("meridian_cli.workspace.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="Resolution failed")
        with pytest.raises(WorkspaceError) as exc_info:
            UvWorkspaceInitializer(repo_root=root).init()

    assert exc_info.value.code == "WORKSPACE_SYNC_FAILED"
    assert "Resolution failed" in exc_info.value.message


def test_init_calls_on_error_on_uv_sync_failure(tmp_path: Path) -> None:
    root = make_workspace_root(tmp_path, _ALL_MEMBERS)
    errors: list[WorkspaceError] = []
    with patch("meridian_cli.workspace.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="lock mismatch")
        with pytest.raises(WorkspaceError):
            UvWorkspaceInitializer(repo_root=root, on_error=errors.append).init()

    assert len(errors) == 1
    assert errors[0].code == "WORKSPACE_SYNC_FAILED"
