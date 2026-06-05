"""
Filesystem jail test suite.

Covers:
  FilesystemEnforcer
    - workspace jail: paths outside $WORKSPACE are always denied.
    - symlink escape: symlinks resolving outside $WORKSPACE are rejected.
    - glob matching: read_globs / write_globs / delete_globs with *, **, ?.
    - agent allowlist: further constrains (intersection, no escalation).
    - assert_allowed: raises FilesystemViolation on denied path.
    - violation fields: operation, path, agent_id, environment_id, session_id.

  FilesystemGate
    - allowed: audit entry at level="info", span emitted, check passes.
    - denied: audit entry at level="error", span set to ERROR, raises FilesystemViolation.
    - FilesystemViolation carries operation, path, agent_id, environment_id, session_id.
    - span name is "fs.access".
    - span attributes include fs.path, fs.operation, environment.id, session.id, agent.id.
    - span ended on both success and failure paths.
    - audit detail contains path, operation, allowed, agent_id.
"""

from __future__ import annotations

import os
from pathlib import Path

from opentelemetry.trace import StatusCode
import pytest
from sdk_environment import (
    AgentFilesystemPolicy,
    AuditLogEntry,
    FilesystemEnforcer,
    FilesystemGate,
    FilesystemPolicy,
    FilesystemViolation,
)

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enforcer(
    workspace: str | os.PathLike[str],
    *,
    read_globs: tuple[str, ...] = (),
    write_globs: tuple[str, ...] = (),
    delete_globs: tuple[str, ...] = (),
    agent: AgentFilesystemPolicy | None = None,
) -> FilesystemEnforcer:
    return FilesystemEnforcer(
        workspace,
        FilesystemPolicy(
            read_globs=read_globs,
            write_globs=write_globs,
            delete_globs=delete_globs,
        ),
        agent,
    )


def _gate(
    enforcer: FilesystemEnforcer,
    audit: CapturingAuditLog,
    *,
    environment_id: str = "env1",
    session_id: str = "sess1",
    agent_id: str = "agent1",
) -> FilesystemGate:
    return FilesystemGate(
        enforcer,
        environment_id=environment_id,
        session_id=session_id,
        agent_id=agent_id,
        audit_log=audit,
    )


# ===========================================================================
# FilesystemEnforcer — workspace jail
# ===========================================================================


class TestEnforcerWorkspaceJail:
    def test_denies_path_outside_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        outside = tmp_path / "outside" / "secret.txt"
        e = _enforcer(ws, read_globs=(str(tmp_path / "**"),))
        assert e.is_allowed("read", str(outside)) is False

    def test_allows_path_inside_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        inside = ws / "file.txt"
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        assert e.is_allowed("read", str(inside)) is True

    def test_denies_dotdot_traversal_escape(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        escape = str(ws / ".." / "secret.txt")
        e = _enforcer(ws, read_globs=(str(tmp_path / "**"),))
        assert e.is_allowed("read", escape) is False

    def test_allows_workspace_root_itself(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws),))
        assert e.is_allowed("read", str(ws)) is True

    def test_denies_sibling_directory(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        sibling = tmp_path / "workspace2" / "file.txt"
        e = _enforcer(ws, read_globs=(str(tmp_path / "**"),))
        assert e.is_allowed("read", str(sibling)) is False


# ===========================================================================
# FilesystemEnforcer — symlink escape
# ===========================================================================


class TestEnforcerSymlinkEscape:
    def test_rejects_symlink_pointing_outside_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        link = ws / "escape"
        link.symlink_to(outside)
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        assert e.is_allowed("read", str(link / "secret.txt")) is False

    def test_allows_symlink_pointing_inside_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        real_dir = ws / "real"
        real_dir.mkdir()
        link = ws / "link"
        link.symlink_to(real_dir)
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        assert e.is_allowed("read", str(link / "file.txt")) is True

    def test_rejects_symlink_to_parent_escape(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        outside = tmp_path / "sibling"
        outside.mkdir()
        link = ws / "up"
        link.symlink_to(tmp_path)
        # ws/up resolves to tmp_path; ws/up/sibling resolves to tmp_path/sibling (outside ws)
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        assert e.is_allowed("read", str(link / "sibling" / "secret.txt")) is False


# ===========================================================================
# FilesystemEnforcer — glob matching
# ===========================================================================


class TestEnforcerGlobMatching:
    def test_star_matches_within_single_directory(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        pattern = str(ws / "*.py")
        e = _enforcer(ws, read_globs=(pattern,))
        assert e.is_allowed("read", str(ws / "main.py")) is True
        assert e.is_allowed("read", str(ws / "sub" / "main.py")) is False

    def test_doublestar_matches_nested_paths(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        # ws/**/subdir/*.py matches any depth but requires at least one directory component
        pattern = str(ws / "**" / "*.py")
        e = _enforcer(ws, read_globs=(pattern,))
        assert e.is_allowed("read", str(ws / "a" / "b" / "c.py")) is True
        assert e.is_allowed("read", str(ws / "src" / "main.py")) is True
        assert e.is_allowed("read", str(ws / "data.json")) is False

    def test_doublestar_alone_matches_everything_under_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        assert e.is_allowed("read", str(ws / "a" / "b" / "c.txt")) is True

    def test_question_mark_matches_single_char(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        pattern = str(ws / "file?.txt")
        e = _enforcer(ws, read_globs=(pattern,))
        assert e.is_allowed("read", str(ws / "file1.txt")) is True
        assert e.is_allowed("read", str(ws / "file12.txt")) is False

    def test_no_matching_glob_denies(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws / "*.py"),))
        assert e.is_allowed("read", str(ws / "data.json")) is False

    def test_empty_globs_denies_all(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=())
        assert e.is_allowed("read", str(ws / "anything.txt")) is False

    def test_multiple_globs_any_match_allows(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws / "*.py"), str(ws / "*.txt")))
        assert e.is_allowed("read", str(ws / "readme.txt")) is True
        assert e.is_allowed("read", str(ws / "main.py")) is True
        assert e.is_allowed("read", str(ws / "data.json")) is False

    def test_operations_use_separate_globs(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(
            ws,
            read_globs=(str(ws / "**"),),
            write_globs=(str(ws / "output" / "**"),),
            delete_globs=(),
        )
        path = str(ws / "src" / "main.py")
        assert e.is_allowed("read", path) is True
        assert e.is_allowed("write", path) is False
        assert e.is_allowed("delete", path) is False


# ===========================================================================
# FilesystemEnforcer — agent allowlist (intersection, no escalation)
# ===========================================================================


class TestEnforcerAgentPolicy:
    def test_agent_further_constrains_read(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        agent = AgentFilesystemPolicy(
            agent_id="a1",
            read_globs=(str(ws / "public" / "**"),),
        )
        e = _enforcer(ws, read_globs=(str(ws / "**"),), agent=agent)
        assert e.is_allowed("read", str(ws / "public" / "index.html")) is True
        assert e.is_allowed("read", str(ws / "private" / "key.pem")) is False

    def test_agent_cannot_escalate_beyond_env(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        agent = AgentFilesystemPolicy(
            agent_id="a1",
            read_globs=(str(ws / "**"),),
        )
        e = _enforcer(ws, read_globs=(str(ws / "public" / "**"),), agent=agent)
        assert e.is_allowed("read", str(ws / "private" / "key.pem")) is False

    def test_empty_agent_globs_do_not_further_restrict(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        agent = AgentFilesystemPolicy(agent_id="a1", read_globs=())
        e = _enforcer(ws, read_globs=(str(ws / "**"),), agent=agent)
        assert e.is_allowed("read", str(ws / "file.txt")) is True

    def test_no_agent_policy_uses_env_only(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        assert e.is_allowed("read", str(ws / "file.txt")) is True

    def test_agent_write_globs_constrain_write(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        agent = AgentFilesystemPolicy(
            agent_id="a1",
            write_globs=(str(ws / "tmp" / "**"),),
        )
        e = _enforcer(
            ws,
            write_globs=(str(ws / "**"),),
            agent=agent,
        )
        assert e.is_allowed("write", str(ws / "tmp" / "out.txt")) is True
        assert e.is_allowed("write", str(ws / "src" / "main.py")) is False


# ===========================================================================
# FilesystemEnforcer — assert_allowed
# ===========================================================================


class TestEnforcerAssertAllowed:
    def test_raises_filesystem_violation_when_denied(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=())
        with pytest.raises(FilesystemViolation) as exc_info:
            e.assert_allowed(
                "read", str(ws / "secret.txt"), environment_id="env1", session_id="sess1"
            )
        v = exc_info.value
        assert v.operation == "read"
        assert "secret.txt" in v.path
        assert v.environment_id == "env1"
        assert v.session_id == "sess1"

    def test_does_not_raise_when_allowed(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        e.assert_allowed("read", str(ws / "ok.txt"))  # no exception

    def test_violation_message_contains_operation_and_path(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, write_globs=())
        with pytest.raises(FilesystemViolation) as exc_info:
            e.assert_allowed("write", str(ws / "blocked.py"))
        assert "write" in str(exc_info.value)
        assert "blocked.py" in str(exc_info.value)

    def test_violation_carries_agent_id(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        # Agent allows only *.txt; path is .py → agent blocks it → violation carries agent_id
        agent = AgentFilesystemPolicy(agent_id="my-agent", read_globs=(str(ws / "*.txt"),))
        e = FilesystemEnforcer(ws, FilesystemPolicy(read_globs=(str(ws / "**"),)), agent)
        with pytest.raises(FilesystemViolation) as exc_info:
            e.assert_allowed("read", str(ws / "main.py"))
        assert exc_info.value.agent_id == "my-agent"

    def test_violation_for_symlink_escape_carries_path(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        link = ws / "escape"
        link.symlink_to(outside)
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        with pytest.raises(FilesystemViolation) as exc_info:
            e.assert_allowed("read", str(link / "file.txt"))
        assert "file.txt" in exc_info.value.path


# ===========================================================================
# FilesystemGate — allowed request
# ===========================================================================


class TestGateAllowed:
    def test_check_passes_on_allowed_path(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        g = _gate(e, audit_log)
        g.check("read", str(ws / "file.txt"))  # must not raise

    def test_audit_entry_level_info(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        g = _gate(e, audit_log)
        g.check("read", str(ws / "file.txt"))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].level == "info"

    def test_audit_entry_event_name(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        g = _gate(e, audit_log)
        g.check("read", str(ws / "file.txt"))
        assert audit_log.entries[0].event == "fs.access"

    def test_audit_entry_detail_allowed_true(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        path = str(ws / "file.txt")
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        g = _gate(e, audit_log)
        g.check("read", path)
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.detail is not None
        assert entry.detail["allowed"] is True
        assert entry.detail["path"] == path
        assert entry.detail["operation"] == "read"

    def test_span_name(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        g = _gate(e, audit_log)
        g.check("read", str(ws / "file.txt"))
        assert mock_fs_gate_span.name == "fs.access"

    def test_span_attributes(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        path = str(ws / "file.txt")
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        g = _gate(e, audit_log, environment_id="env1", session_id="sess1", agent_id="agent1")
        g.check("read", path)
        assert mock_fs_gate_span.attributes["fs.path"] == path
        assert mock_fs_gate_span.attributes["fs.operation"] == "read"
        assert mock_fs_gate_span.attributes["environment.id"] == "env1"
        assert mock_fs_gate_span.attributes["session.id"] == "sess1"
        assert mock_fs_gate_span.attributes["agent.id"] == "agent1"

    def test_span_ended_on_success(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        g = _gate(e, audit_log)
        g.check("read", str(ws / "file.txt"))
        assert mock_fs_gate_span.ended


# ===========================================================================
# FilesystemGate — denied request
# ===========================================================================


class TestGateDenied:
    def test_raises_filesystem_violation(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, write_globs=())
        g = _gate(e, audit_log)
        with pytest.raises(FilesystemViolation) as exc_info:
            g.check("write", str(ws / "blocked.py"))
        assert exc_info.value.operation == "write"
        assert "blocked.py" in exc_info.value.path

    def test_violation_carries_context(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, delete_globs=())
        g = _gate(e, audit_log, environment_id="env1", session_id="sess1", agent_id="agent1")
        with pytest.raises(FilesystemViolation) as exc_info:
            g.check("delete", str(ws / "file.txt"))
        v = exc_info.value
        assert v.environment_id == "env1"
        assert v.session_id == "sess1"
        assert v.agent_id == "agent1"

    def test_audit_entry_level_error(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=())
        g = _gate(e, audit_log)
        with pytest.raises(FilesystemViolation):
            g.check("read", str(ws / "file.txt"))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].level == "error"

    def test_audit_entry_detail_allowed_false(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        path = str(ws / "blocked.txt")
        e = _enforcer(ws, read_globs=())
        g = _gate(e, audit_log)
        with pytest.raises(FilesystemViolation):
            g.check("read", path)
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.detail is not None
        assert entry.detail["allowed"] is False
        assert entry.detail["path"] == path
        assert entry.detail["operation"] == "read"

    def test_span_set_to_error(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, write_globs=())
        g = _gate(e, audit_log)
        with pytest.raises(FilesystemViolation):
            g.check("write", str(ws / "file.txt"))
        assert mock_fs_gate_span.status is not None
        assert mock_fs_gate_span.status.status_code == StatusCode.ERROR

    def test_span_ended_on_denial(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        e = _enforcer(ws, read_globs=())
        g = _gate(e, audit_log)
        with pytest.raises(FilesystemViolation):
            g.check("read", str(ws / "file.txt"))
        assert mock_fs_gate_span.ended

    def test_violation_for_symlink_escape(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        link = ws / "escape"
        link.symlink_to(outside)
        e = _enforcer(ws, read_globs=(str(ws / "**"),))
        g = _gate(e, audit_log)
        with pytest.raises(FilesystemViolation):
            g.check("read", str(link / "secret.txt"))
        assert audit_log.entries[0].level == "error"


# ===========================================================================
# Integration: default-deny per env, agent allowlist further constrains
# ===========================================================================


class TestIntegrationEnvAndAgentIntersection:
    def test_allowed_by_both_env_and_agent(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        agent = AgentFilesystemPolicy(
            agent_id="a1",
            read_globs=(str(ws / "public" / "**"),),
        )
        e = FilesystemEnforcer(
            ws,
            FilesystemPolicy(read_globs=(str(ws / "public" / "**"), str(ws / "src" / "**"))),
            agent,
        )
        g = _gate(e, audit_log, agent_id="a1")
        g.check("read", str(ws / "public" / "index.html"))  # no exception
        assert audit_log.entries[0].level == "info"

    def test_allowed_by_env_but_not_agent_is_denied(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        agent = AgentFilesystemPolicy(
            agent_id="a1",
            read_globs=(str(ws / "public" / "**"),),
        )
        e = FilesystemEnforcer(
            ws,
            FilesystemPolicy(read_globs=(str(ws / "public" / "**"), str(ws / "src" / "**"))),
            agent,
        )
        g = _gate(e, audit_log, agent_id="a1")
        with pytest.raises(FilesystemViolation) as exc_info:
            g.check("read", str(ws / "src" / "main.py"))
        assert exc_info.value.agent_id == "a1"
        assert audit_log.entries[0].level == "error"

    def test_denied_by_env_even_if_agent_would_allow(
        self, tmp_path: Path, mock_fs_gate_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        agent = AgentFilesystemPolicy(
            agent_id="a1",
            read_globs=(str(ws / "**"),),
        )
        e = FilesystemEnforcer(
            ws,
            FilesystemPolicy(read_globs=(str(ws / "public" / "**"),)),
            agent,
        )
        g = _gate(e, audit_log, agent_id="a1")
        with pytest.raises(FilesystemViolation):
            g.check("read", str(ws / "src" / "main.py"))
