"""
Hook stdin redaction conformance suite.

Tests cover:
  - redact_vault_refs: vault ref without secret.read declaration is left unsubstituted.
  - redact_vault_refs: plaintext secret value does NOT reach hook stdin when undeclared (Risk R7 CI assertion).
  - redact_vault_refs: vault ref with secret.read declaration is resolved to plaintext.
  - redact_vault_refs: non-vault-ref strings are passed through unchanged.
  - redact_vault_refs: non-string scalar values (int, bool, None) pass through unchanged.
  - redact_vault_refs: nested dict vault refs are redacted/resolved recursively.
  - redact_vault_refs: nested list vault refs are redacted/resolved recursively.
  - redact_vault_refs: multiple refs — only declared ones are resolved, others unsubstituted.
  - redact_vault_refs: OTel span "hook.stdin.redact" emitted on every invocation.
  - redact_vault_refs: span carries "hook.stdin.allowed_key_count" attribute.
  - redact_vault_refs: span has structured invocation event with code "hook_stdin_redact".
  - redact_vault_refs: span status is ERROR on failure.
  - redact_vault_refs: writes audit log entry with event "hook.stdin.redact.failed" on failure.
  - redact_vault_refs: audit entry level is "error" on failure.
  - redact_vault_refs: audit entry code is "hook_stdin_redaction_failed" on failure.
  - redact_vault_refs: audit detail has "message" on failure.
  - redact_vault_refs: raises HookStdinRedactionError on failure.
  - redact_vault_refs: HookStdinRedactionError.code is "hook_stdin_redaction_failed".
  - redact_vault_refs: no audit log does not crash (NoopAuditLog fallback).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
from opentelemetry.trace import StatusCode

from meridiand._hook_stdin_redaction import HookStdinRedactionError, redact_vault_refs
from meridiand._secret_ref import SecretRefResolver
from meridiand._vault_backend_os_keychain import OsKeychainVaultBackend

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MemoryKeyring:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self._store.pop((service, username), None)


class _CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


def _make_backend() -> OsKeychainVaultBackend:
    return OsKeychainVaultBackend(_keyring=_MemoryKeyring())


def _write_vault(storage_root: Path, vault_id: str) -> None:
    vaults_dir = storage_root / "vaults"
    vaults_dir.mkdir(parents=True, exist_ok=True)
    (vaults_dir / f"{vault_id}.json").write_text(
        json.dumps({"id": vault_id, "name": "test-vault", "backend": "os_keychain"})
    )


def _store_secret(backend: OsKeychainVaultBackend, vault_id: str, key: str, value: str) -> None:
    backend.store_secret(vault_id, key, value, "2026-01-01T00:00:00+00:00")


def _make_resolver(
    storage_root: Path,
    backend: OsKeychainVaultBackend | None = None,
) -> SecretRefResolver:
    return SecretRefResolver(
        storage_root=storage_root,
        os_keychain_backend=backend,
    )


def _ref(vault_id: str, key: str) -> str:
    return f"secret_ref://vault/{vault_id}/{key}"


# ---------------------------------------------------------------------------
# TestRedactionBasics
# ---------------------------------------------------------------------------


class TestRedactionBasics:
    def test_vault_ref_without_declaration_is_unsubstituted(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_a")
        _store_secret(backend, "vault_a", "api_key", "super_secret_value")
        resolver = _make_resolver(storage_root, backend)

        ref = _ref("vault_a", "api_key")
        result = redact_vault_refs({"key": ref}, allowed_keys=frozenset(), resolver=resolver)

        assert result["key"] == ref

    def test_r7_no_vault_plaintext_in_unallowed_stdin(self, storage_root: Path) -> None:
        # Risk R7: plaintext secret value must never reach hook stdin when
        # the hook has not declared secret.read[key].
        backend = _make_backend()
        _write_vault(storage_root, "vault_r7")
        plaintext = "plaintext_secret_r7_canary"
        _store_secret(backend, "vault_r7", "api_key", plaintext)
        resolver = _make_resolver(storage_root, backend)

        payload = {"args": {"token": _ref("vault_r7", "api_key")}}
        result = redact_vault_refs(payload, allowed_keys=frozenset(), resolver=resolver)

        stdin_json = json.dumps(result)
        assert plaintext not in stdin_json

    def test_vault_ref_with_declaration_is_resolved(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_b")
        _store_secret(backend, "vault_b", "db_pass", "resolved_value")
        resolver = _make_resolver(storage_root, backend)

        result = redact_vault_refs(
            {"pw": _ref("vault_b", "db_pass")},
            allowed_keys=frozenset({"db_pass"}),
            resolver=resolver,
        )

        assert result["pw"] == "resolved_value"

    def test_non_vault_ref_string_passes_through(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        result = redact_vault_refs(
            {"msg": "hello world"},
            allowed_keys=frozenset(),
            resolver=resolver,
        )
        assert result["msg"] == "hello world"

    def test_non_string_int_passes_through(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        result = redact_vault_refs({"n": 42}, allowed_keys=frozenset(), resolver=resolver)
        assert result["n"] == 42

    def test_non_string_bool_passes_through(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        result = redact_vault_refs({"flag": True}, allowed_keys=frozenset(), resolver=resolver)
        assert result["flag"] is True

    def test_none_value_passes_through(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        result = redact_vault_refs({"x": None}, allowed_keys=frozenset(), resolver=resolver)
        assert result["x"] is None


# ---------------------------------------------------------------------------
# TestRedactionRecursive
# ---------------------------------------------------------------------------


class TestRedactionRecursive:
    def test_nested_dict_vault_ref_is_unsubstituted(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_n")
        _store_secret(backend, "vault_n", "tok", "secret_tok")
        resolver = _make_resolver(storage_root, backend)

        ref = _ref("vault_n", "tok")
        result = redact_vault_refs(
            {"outer": {"inner": ref}},
            allowed_keys=frozenset(),
            resolver=resolver,
        )

        assert result["outer"]["inner"] == ref
        assert "secret_tok" not in json.dumps(result)

    def test_nested_list_vault_ref_is_unsubstituted(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_lst")
        _store_secret(backend, "vault_lst", "k", "list_secret")
        resolver = _make_resolver(storage_root, backend)

        ref = _ref("vault_lst", "k")
        result = redact_vault_refs(
            {"items": [ref, "plain"]},
            allowed_keys=frozenset(),
            resolver=resolver,
        )

        assert result["items"][0] == ref
        assert result["items"][1] == "plain"
        assert "list_secret" not in json.dumps(result)

    def test_nested_dict_vault_ref_is_resolved_when_declared(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_nd")
        _store_secret(backend, "vault_nd", "cred", "deep_secret")
        resolver = _make_resolver(storage_root, backend)

        result = redact_vault_refs(
            {"config": {"auth": _ref("vault_nd", "cred")}},
            allowed_keys=frozenset({"cred"}),
            resolver=resolver,
        )

        assert result["config"]["auth"] == "deep_secret"

    def test_multiple_refs_only_declared_resolved(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_m")
        _store_secret(backend, "vault_m", "allowed_key", "plaintext_allowed")
        _store_secret(backend, "vault_m", "denied_key", "plaintext_denied")
        resolver = _make_resolver(storage_root, backend)

        ref_allowed = _ref("vault_m", "allowed_key")
        ref_denied = _ref("vault_m", "denied_key")

        result = redact_vault_refs(
            {"a": ref_allowed, "b": ref_denied},
            allowed_keys=frozenset({"allowed_key"}),
            resolver=resolver,
        )

        assert result["a"] == "plaintext_allowed"
        assert result["b"] == ref_denied
        assert "plaintext_denied" not in json.dumps(result)


# ---------------------------------------------------------------------------
# TestRedactionOtel
# ---------------------------------------------------------------------------


class TestRedactionOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_emits_hook_stdin_redact_span(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        redact_vault_refs({}, allowed_keys=frozenset(), resolver=resolver)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "hook.stdin.redact" in span_names

    def test_span_has_allowed_key_count_attribute(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        redact_vault_refs({}, allowed_keys=frozenset({"k1", "k2"}), resolver=resolver)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "hook.stdin.redact")
        assert span.attributes["hook.stdin.allowed_key_count"] == 2

    def test_span_has_zero_allowed_key_count_when_empty(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        redact_vault_refs({}, allowed_keys=frozenset(), resolver=resolver)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "hook.stdin.redact")
        assert span.attributes["hook.stdin.allowed_key_count"] == 0

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        redact_vault_refs({}, allowed_keys=frozenset(), resolver=resolver)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "hook.stdin.redact")
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_invocation_event_code(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        redact_vault_refs({}, allowed_keys=frozenset(), resolver=resolver)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "hook.stdin.redact")
        evt = next(e for e in span.events if e.name == "meridian.error.invocation")
        assert evt.attributes["code"] == "hook_stdin_redact"

    def test_failure_span_status_is_error(self, storage_root: Path) -> None:
        # Trigger failure by using a resolver with no backend — resolution of an
        # allowed key will fail when the vault file is missing.
        backend = _make_backend()
        resolver = _make_resolver(storage_root, backend)
        # No vault file written — vault not found → resolver raises → redaction fails
        with pytest.raises(HookStdinRedactionError):
            redact_vault_refs(
                {"k": _ref("vault_missing", "key")},
                allowed_keys=frozenset({"key"}),
                resolver=resolver,
            )
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "hook.stdin.redact")
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# TestRedactionAuditLog
# ---------------------------------------------------------------------------


class TestRedactionAuditLog:
    def _failing_resolver(self, storage_root: Path) -> SecretRefResolver:
        return _make_resolver(storage_root, _make_backend())

    def test_resolution_failure_writes_audit(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = self._failing_resolver(storage_root)
        with pytest.raises(HookStdinRedactionError):
            redact_vault_refs(
                {"k": _ref("vault_ghost", "key")},
                allowed_keys=frozenset({"key"}),
                resolver=resolver,
                audit_log=audit,
            )
        assert any(e.event == "hook.stdin.redact.failed" for e in audit.entries)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = self._failing_resolver(storage_root)
        with pytest.raises(HookStdinRedactionError):
            redact_vault_refs(
                {"k": _ref("vault_ghost", "key")},
                allowed_keys=frozenset({"key"}),
                resolver=resolver,
                audit_log=audit,
            )
        entry = next(e for e in audit.entries if e.event == "hook.stdin.redact.failed")
        assert entry.level == "error"

    def test_failure_audit_code(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = self._failing_resolver(storage_root)
        with pytest.raises(HookStdinRedactionError):
            redact_vault_refs(
                {"k": _ref("vault_ghost", "key")},
                allowed_keys=frozenset({"key"}),
                resolver=resolver,
                audit_log=audit,
            )
        entry = next(e for e in audit.entries if e.event == "hook.stdin.redact.failed")
        assert entry.code == "hook_stdin_redaction_failed"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = self._failing_resolver(storage_root)
        with pytest.raises(HookStdinRedactionError):
            redact_vault_refs(
                {"k": _ref("vault_ghost", "key")},
                allowed_keys=frozenset({"key"}),
                resolver=resolver,
                audit_log=audit,
            )
        entry = next(e for e in audit.entries if e.event == "hook.stdin.redact.failed")
        assert entry.detail is not None
        assert len(entry.detail["message"]) > 0

    def test_no_audit_log_does_not_crash(self, storage_root: Path) -> None:
        resolver = self._failing_resolver(storage_root)
        with pytest.raises(HookStdinRedactionError):
            redact_vault_refs(
                {"k": _ref("vault_ghost", "key")},
                allowed_keys=frozenset({"key"}),
                resolver=resolver,
            )

    def test_noop_audit_log_accepted(self, storage_root: Path) -> None:
        resolver = self._failing_resolver(storage_root)
        with pytest.raises(HookStdinRedactionError):
            redact_vault_refs(
                {"k": _ref("vault_ghost", "key")},
                allowed_keys=frozenset({"key"}),
                resolver=resolver,
                audit_log=NoopAuditLog(),
            )


# ---------------------------------------------------------------------------
# TestRedactionErrorType
# ---------------------------------------------------------------------------


class TestRedactionErrorType:
    def test_raises_hook_stdin_redaction_error_on_failure(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root, _make_backend())
        with pytest.raises(HookStdinRedactionError):
            redact_vault_refs(
                {"k": _ref("vault_missing", "key")},
                allowed_keys=frozenset({"key"}),
                resolver=resolver,
            )

    def test_error_code_is_hook_stdin_redaction_failed(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root, _make_backend())
        with pytest.raises(HookStdinRedactionError) as exc_info:
            redact_vault_refs(
                {"k": _ref("vault_missing", "key")},
                allowed_keys=frozenset({"key"}),
                resolver=resolver,
            )
        assert exc_info.value.code == "hook_stdin_redaction_failed"

    def test_error_message_is_non_empty(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root, _make_backend())
        with pytest.raises(HookStdinRedactionError) as exc_info:
            redact_vault_refs(
                {"k": _ref("vault_missing", "key")},
                allowed_keys=frozenset({"key"}),
                resolver=resolver,
            )
        assert len(exc_info.value.message) > 0
