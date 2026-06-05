"""
Secret-ref substitution conformance suite.

Covers:
  - substitute_secret_refs: success paths (single, multiple, dedup, nested)
  - substitute_secret_refs: no refs → input returned unchanged
  - substitute_secret_refs: vault not found → SecretRefVaultNotFoundError + audit entry
  - substitute_secret_refs: secret not found → SecretRefNotFoundError + audit entry
  - substitute_secret_refs: corrupt secret file → SecretRefResolveError + audit entry
  - OTel span emitted with correct name and attributes on success and failure
  - refs list returned (not plaintext) — plaintext never in event log
  - Sandbox.execute integration: no storage_root → no substitution
  - Sandbox.execute integration: storage_root set, no refs → dispatches normally
  - Sandbox.execute integration: storage_root + refs → substituted input reaches dispatcher
  - Sandbox.execute integration: resolution failure → SandboxResult(is_error=True)
  - Sandbox.execute integration: on_error callback called on resolution failure
  - Sandbox.execute integration: dispatcher never called after resolution failure
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sdk_sandbox import (
    AuditLogEntry,
    ExecutionContext,
    InProcessHandler,
    RuntimeOptions,
    Sandbox,
    SandboxResult,
    ToolDefinition,
    ToolDispatcher,
)
from sdk_sandbox._secret_refs import (
    SecretRefNotFoundError,
    SecretRefResolveError,
    SecretRefVaultNotFoundError,
    substitute_secret_refs,
)

from .conftest import CapturingAuditLog, MockSpan, MockTracer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockSecretRefTracer(MockTracer):
    """Tracer that patches sdk_sandbox._secret_refs.get_tracer."""


@pytest.fixture()
def mock_secret_ref_tracer(monkeypatch: pytest.MonkeyPatch) -> MockSecretRefTracer:
    tracer = MockSecretRefTracer()
    monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def secret_ref_span(mock_secret_ref_tracer: MockSecretRefTracer) -> MockSpan:
    return mock_secret_ref_tracer.span


def write_vault(storage_root: Path, vault_id: str) -> None:
    """Create the vault metadata file so the vault is 'found'."""
    vault_dir = storage_root / "vaults"
    vault_dir.mkdir(parents=True, exist_ok=True)
    (vault_dir / f"{vault_id}.json").write_text(json.dumps({"id": vault_id}))


def write_secret(storage_root: Path, vault_id: str, name: str, value: str) -> None:
    """Write a secret record under the vault."""
    secret_dir = storage_root / "vaults" / vault_id / "secrets"
    secret_dir.mkdir(parents=True, exist_ok=True)
    (secret_dir / f"{name}.json").write_text(json.dumps({"value": value}))


# ---------------------------------------------------------------------------
# Unit: substitute_secret_refs — success paths
# ---------------------------------------------------------------------------


class TestSubstituteSuccess:
    def test_no_refs_returns_input_unchanged(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        payload: dict[str, Any] = {"msg": "hello", "count": 42}
        result, refs = substitute_secret_refs(
            payload,
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="test.tool",
            session_id="sess1",
        )
        assert result == payload
        assert refs == []

    def test_no_refs_no_audit_entries(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        substitute_secret_refs(
            {"x": "plain"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        assert audit_log.entries == []

    def test_single_ref_substituted(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "api_key", "supersecret")
        result, refs = substitute_secret_refs(
            {"token": "secret_ref://vault/v1/api_key"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        assert result == {"token": "supersecret"}

    def test_refs_returned_not_plaintext(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "api_key", "supersecret")
        _, refs = substitute_secret_refs(
            {"token": "secret_ref://vault/v1/api_key"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        assert refs == ["secret_ref://vault/v1/api_key"]
        assert "supersecret" not in refs

    def test_multiple_distinct_refs(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "key1", "val1")
        write_secret(tmp_path, "v1", "key2", "val2")
        result, refs = substitute_secret_refs(
            {"a": "secret_ref://vault/v1/key1", "b": "secret_ref://vault/v1/key2"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        assert result == {"a": "val1", "b": "val2"}
        assert set(refs) == {
            "secret_ref://vault/v1/key1",
            "secret_ref://vault/v1/key2",
        }

    def test_duplicate_refs_deduped_in_resolution(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "pw", "secret123")
        result, refs = substitute_secret_refs(
            {"x": "secret_ref://vault/v1/pw", "y": "secret_ref://vault/v1/pw"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        assert result == {"x": "secret123", "y": "secret123"}
        assert refs.count("secret_ref://vault/v1/pw") == 2  # collected, not deduped in output

    def test_nested_dict_ref_substituted(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "db_pass", "dbsecret")
        result, _ = substitute_secret_refs(
            {"db": {"password": "secret_ref://vault/v1/db_pass", "host": "localhost"}},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        assert result == {"db": {"password": "dbsecret", "host": "localhost"}}

    def test_list_ref_substituted(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "tok", "mytoken")
        result, _ = substitute_secret_refs(
            {"tokens": ["secret_ref://vault/v1/tok", "plain"]},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        assert result == {"tokens": ["mytoken", "plain"]}

    def test_non_ref_strings_untouched(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "k", "v")
        result, _ = substitute_secret_refs(
            {"ref": "secret_ref://vault/v1/k", "plain": "leave_me"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        assert result["plain"] == "leave_me"

    def test_span_emitted_on_success(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "k", "v")
        substitute_secret_refs(
            {"x": "secret_ref://vault/v1/k"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="test.tool",
            session_id="sess1",
        )
        assert secret_ref_span.name == "secret_ref.substitute"

    def test_span_attributes_on_success(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "k", "v")
        substitute_secret_refs(
            {"x": "secret_ref://vault/v1/k"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="test.tool",
            session_id="sess1",
        )
        assert secret_ref_span.attributes["tool.name"] == "test.tool"
        assert secret_ref_span.attributes["session.id"] == "sess1"
        assert secret_ref_span.attributes["ref_count"] == 1

    def test_substituted_event_on_span(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "k", "v")
        substitute_secret_refs(
            {"x": "secret_ref://vault/v1/k"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        event_names = [e[0] for e in secret_ref_span.events]
        assert "secret_ref.substituted" in event_names

    def test_substituted_event_refs_not_plaintext(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "k", "mysecretvalue")
        substitute_secret_refs(
            {"x": "secret_ref://vault/v1/k"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        ev = next(e for e in secret_ref_span.events if e[0] == "secret_ref.substituted")
        assert "mysecretvalue" not in str(ev[1])
        assert "secret_ref://vault/v1/k" in ev[1]["refs"]

    def test_no_audit_entries_on_success(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "k", "v")
        substitute_secret_refs(
            {"x": "secret_ref://vault/v1/k"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        assert audit_log.entries == []

    def test_secret_value_with_special_chars(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "tok", 'p@$$w0rd!#"')
        result, _ = substitute_secret_refs(
            {"pw": "secret_ref://vault/v1/tok"},
            storage_root=tmp_path,
            audit_log=audit_log,
            tool_name="t",
            session_id="s",
        )
        assert result["pw"] == 'p@$$w0rd!#"'


# ---------------------------------------------------------------------------
# Unit: substitute_secret_refs — failure paths
# ---------------------------------------------------------------------------


class TestSubstituteVaultNotFound:
    def test_raises_vault_not_found_error(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(SecretRefVaultNotFoundError) as exc_info:
            substitute_secret_refs(
                {"x": "secret_ref://vault/missing_vault/key"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        assert exc_info.value.code == "secret_ref_vault_not_found"

    def test_error_message_names_vault(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(SecretRefVaultNotFoundError) as exc_info:
            substitute_secret_refs(
                {"x": "secret_ref://vault/missing_vault/key"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        assert "missing_vault" in exc_info.value.message

    def test_audit_entry_written(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(SecretRefVaultNotFoundError):
            substitute_secret_refs(
                {"x": "secret_ref://vault/missing_vault/key"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "secret_ref.substitute.failed"

    def test_audit_entry_detail(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(SecretRefVaultNotFoundError):
            substitute_secret_refs(
                {"x": "secret_ref://vault/mv/key"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        detail = audit_log.entries[0].detail
        assert detail is not None
        assert detail["code"] == "secret_ref_vault_not_found"
        assert detail["ref"] == "secret_ref://vault/mv/key"

    def test_span_marked_error(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        from opentelemetry.trace import StatusCode

        with pytest.raises(SecretRefVaultNotFoundError):
            substitute_secret_refs(
                {"x": "secret_ref://vault/mv/key"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        assert secret_ref_span.status is not None
        assert secret_ref_span.status.status_code == StatusCode.ERROR

    def test_failed_event_on_span(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(SecretRefVaultNotFoundError):
            substitute_secret_refs(
                {"x": "secret_ref://vault/mv/key"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        event_names = [e[0] for e in secret_ref_span.events]
        assert "secret_ref.substitute.failed" in event_names

    def test_failed_event_ref_not_plaintext(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(SecretRefVaultNotFoundError):
            substitute_secret_refs(
                {"x": "secret_ref://vault/mv/key"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        ev = next(e for e in secret_ref_span.events if e[0] == "secret_ref.substitute.failed")
        assert ev[1]["ref"] == "secret_ref://vault/mv/key"


class TestSubstituteSecretNotFound:
    def test_raises_secret_not_found_error(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        with pytest.raises(SecretRefNotFoundError) as exc_info:
            substitute_secret_refs(
                {"x": "secret_ref://vault/v1/missing_secret"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        assert exc_info.value.code == "secret_ref_not_found"

    def test_error_message_names_secret(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        with pytest.raises(SecretRefNotFoundError) as exc_info:
            substitute_secret_refs(
                {"x": "secret_ref://vault/v1/missing_secret"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        assert "missing_secret" in exc_info.value.message

    def test_audit_entry_written(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        with pytest.raises(SecretRefNotFoundError):
            substitute_secret_refs(
                {"x": "secret_ref://vault/v1/missing_secret"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        assert len(audit_log.entries) == 1
        entry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "secret_ref.substitute.failed"

    def test_audit_entry_detail(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        ref = "secret_ref://vault/v1/ms"
        with pytest.raises(SecretRefNotFoundError):
            substitute_secret_refs(
                {"x": ref},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        detail = audit_log.entries[0].detail
        assert detail is not None
        assert detail["code"] == "secret_ref_not_found"
        assert detail["ref"] == ref

    def test_span_marked_error(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        from opentelemetry.trace import StatusCode

        write_vault(tmp_path, "v1")
        with pytest.raises(SecretRefNotFoundError):
            substitute_secret_refs(
                {"x": "secret_ref://vault/v1/ms"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        assert secret_ref_span.status is not None
        assert secret_ref_span.status.status_code == StatusCode.ERROR


class TestSubstituteReadFailure:
    def test_raises_secret_ref_resolve_error(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Corrupt JSON in a secret file raises a read-failed error."""
        write_vault(tmp_path, "v1")
        secret_dir = tmp_path / "vaults" / "v1" / "secrets"
        secret_dir.mkdir(parents=True, exist_ok=True)
        (secret_dir / "bad.json").write_text("not json {{")
        with pytest.raises(SecretRefResolveError) as exc_info:
            substitute_secret_refs(
                {"x": "secret_ref://vault/v1/bad"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        assert exc_info.value.code == "secret_ref_read_failed"

    def test_audit_entry_written_on_read_failure(
        self, tmp_path: Path, secret_ref_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        write_vault(tmp_path, "v1")
        secret_dir = tmp_path / "vaults" / "v1" / "secrets"
        secret_dir.mkdir(parents=True, exist_ok=True)
        (secret_dir / "bad.json").write_text("{{")
        with pytest.raises(SecretRefResolveError):
            substitute_secret_refs(
                {"x": "secret_ref://vault/v1/bad"},
                storage_root=tmp_path,
                audit_log=audit_log,
                tool_name="t",
                session_id="s",
            )
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].event == "secret_ref.substitute.failed"


# ---------------------------------------------------------------------------
# Integration: Sandbox.execute with secret_ref substitution
# ---------------------------------------------------------------------------


ECHO_TOOL = ToolDefinition(
    name="test.echo",
    description="Echo input",
    input_schema={"type": "object"},
    handler=InProcessHandler(),
)

CTX = ExecutionContext(session_id="sess1", workspace="/tmp")


class CapturingDispatcher(ToolDispatcher):
    kind = "in_process"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def dispatch(
        self, tool: ToolDefinition, input: dict, context: ExecutionContext
    ) -> SandboxResult:
        self.calls.append(input)
        return SandboxResult(content="ok", duration_ms=1.0)


def make_sandbox(dispatcher: CapturingDispatcher) -> Sandbox:
    sb = Sandbox()
    sb.register_dispatcher(dispatcher)
    sb.register_tool(ECHO_TOOL)
    return sb


class TestSandboxSecretRefNoStorageRoot:
    async def test_no_storage_root_dispatches_ref_string_unchanged(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When storage_root is not set, secret_ref strings are passed as-is."""
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log)
        await sb.execute(
            "test.echo",
            {"token": "secret_ref://vault/v1/api_key"},
            CTX,
            opts,
        )
        assert len(dispatcher.calls) == 1
        assert dispatcher.calls[0]["token"] == "secret_ref://vault/v1/api_key"


class TestSandboxSecretRefSuccess:
    async def test_refs_substituted_before_dispatch(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        write_vault(tmp_path, "vault1")
        write_secret(tmp_path, "vault1", "api_key", "resolved_token")
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        result = await sb.execute(
            "test.echo",
            {"token": "secret_ref://vault/vault1/api_key"},
            CTX,
            opts,
        )
        assert result.is_error is False
        assert len(dispatcher.calls) == 1
        assert dispatcher.calls[0]["token"] == "resolved_token"

    async def test_no_refs_in_input_dispatches_normally(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        result = await sb.execute("test.echo", {"msg": "plain"}, CTX, opts)
        assert result.is_error is False
        assert dispatcher.calls[0]["msg"] == "plain"

    async def test_no_audit_entries_on_successful_substitution(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "k", "val")
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        await sb.execute("test.echo", {"x": "secret_ref://vault/v1/k"}, CTX, opts)
        assert audit_log.entries == []

    async def test_plaintext_not_in_span_events(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The secret value must never appear in any OTel span event."""
        secret_tracer = MockTracer()
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: secret_tracer)
        write_vault(tmp_path, "v1")
        write_secret(tmp_path, "v1", "pw", "TOPLEVEL_SECRET_VALUE")
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        await sb.execute("test.echo", {"x": "secret_ref://vault/v1/pw"}, CTX, opts)
        all_event_text = str(secret_tracer.span.events)
        assert "TOPLEVEL_SECRET_VALUE" not in all_event_text


class TestSandboxSecretRefFailure:
    async def test_vault_not_found_returns_is_error(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        result = await sb.execute(
            "test.echo",
            {"token": "secret_ref://vault/no_vault/key"},
            CTX,
            opts,
        )
        assert isinstance(result, SandboxResult)
        assert result.is_error is True

    async def test_vault_not_found_error_code(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        result = await sb.execute(
            "test.echo",
            {"token": "secret_ref://vault/no_vault/key"},
            CTX,
            opts,
        )
        assert result.error_code == "secret_ref_vault_not_found"

    async def test_secret_not_found_returns_is_error(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        write_vault(tmp_path, "v1")
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        result = await sb.execute(
            "test.echo",
            {"token": "secret_ref://vault/v1/no_secret"},
            CTX,
            opts,
        )
        assert result.is_error is True
        assert result.error_code == "secret_ref_not_found"

    async def test_error_message_surfaced_to_caller(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        result = await sb.execute(
            "test.echo",
            {"token": "secret_ref://vault/no_vault/key"},
            CTX,
            opts,
        )
        assert result.error_message is not None
        assert len(result.error_message) > 0

    async def test_dispatcher_not_called_on_ref_failure(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        await sb.execute(
            "test.echo",
            {"token": "secret_ref://vault/no_vault/key"},
            CTX,
            opts,
        )
        assert dispatcher.calls == []

    async def test_on_error_callback_called_on_ref_failure(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sdk_sandbox import SandboxFailure

        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        errors: list[SandboxFailure] = []
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(
            audit_log=audit_log,
            storage_root=tmp_path,
            on_error=lambda e: errors.append(e),
        )
        await sb.execute(
            "test.echo",
            {"token": "secret_ref://vault/no_vault/key"},
            CTX,
            opts,
        )
        assert len(errors) == 1
        assert errors[0].code == "secret_ref_vault_not_found"

    async def test_audit_entry_written_on_ref_failure(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        await sb.execute(
            "test.echo",
            {"token": "secret_ref://vault/no_vault/key"},
            CTX,
            opts,
        )
        assert any(e.event == "secret_ref.substitute.failed" for e in audit_log.entries)

    async def test_span_ended_on_ref_failure(
        self,
        tmp_path: Path,
        mock_span: MockSpan,
        audit_log: CapturingAuditLog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sdk_sandbox._secret_refs.get_tracer", lambda: MockTracer())
        dispatcher = CapturingDispatcher()
        sb = make_sandbox(dispatcher)
        opts = RuntimeOptions(audit_log=audit_log, storage_root=tmp_path)
        await sb.execute(
            "test.echo",
            {"token": "secret_ref://vault/no_vault/key"},
            CTX,
            opts,
        )
        assert mock_span.ended
