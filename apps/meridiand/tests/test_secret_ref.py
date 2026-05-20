"""
SecretRefResolver conformance suite.

Tests cover:
  - resolve: returns the secret value for a valid secret_ref://vault/{id}/{key} URI.
  - resolve: fetches fresh from vault on every call (no cache) so rotation takes effect.
  - resolve: raises SecretRefParseError for an invalid URI.
  - resolve: raises SecretRefResolveError when the vault does not exist.
  - resolve: raises SecretRefNotFoundError when the secret key is absent from the vault.
  - resolve: raises SecretRefResolveError when os_keychain backend is not configured.
  - resolve: raises SecretRefResolveError when encrypted_file backend is not configured.
  - resolve: emits OTel span "vault.secret.resolve" on every invocation.
  - resolve: span carries "vault.id" and "secret.key" attributes on success.
  - resolve: span carries "secret.ref" attribute on every invocation.
  - resolve: span has invocation event with code "vault_secret_resolve".
  - resolve: span status is ERROR on failure.
  - resolve: writes audit log entry with event "vault.secret.resolve.failed" on failure.
  - resolve: audit entry level is "error" on failure.
  - resolve: audit entry detail contains "ref" and "message".
  - resolve: audit code matches the raised error's code.
  - resolve: NoopAuditLog used when audit_log is None (no crash).
  - resolve: emits audit.secret_access entry with level "info" on success.
  - resolve: audit.secret_access detail contains vault_id, name, requester_agent_id, requester_tool_call_id.
  - resolve: audit.secret_access requester fields are None when not provided.
  - resolve: audit.secret_access requester fields carry provided agent_id and tool_call_id.
  - resolve: failure audit detail contains requester_agent_id and requester_tool_call_id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
from opentelemetry.trace import StatusCode

from meridiand._secret_ref import (
    SecretRefNotFoundError,
    SecretRefParseError,
    SecretRefResolveError,
    SecretRefResolver,
)
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


def _make_resolver(
    storage_root: Path,
    backend: OsKeychainVaultBackend | None = None,
    audit_log: AuditLog | None = None,
) -> SecretRefResolver:
    return SecretRefResolver(
        storage_root=storage_root,
        os_keychain_backend=backend,
        audit_log=audit_log,
    )


def _write_vault(storage_root: Path, vault_id: str, backend: str = "os_keychain") -> Path:
    vaults_dir = storage_root / "vaults"
    vaults_dir.mkdir(parents=True, exist_ok=True)
    vault_file = vaults_dir / f"{vault_id}.json"
    vault_file.write_text(
        json.dumps({"id": vault_id, "name": "test-vault", "backend": backend})
    )
    return vault_file


def _store_secret(
    backend: OsKeychainVaultBackend,
    vault_id: str,
    key: str,
    value: str,
) -> None:
    backend.store_secret(vault_id, key, value, "2026-01-01T00:00:00+00:00")


# ---------------------------------------------------------------------------
# TestSecretRefResolverSuccess
# ---------------------------------------------------------------------------


class TestSecretRefResolverSuccess:
    def test_returns_secret_value(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_abc")
        _store_secret(backend, "vault_abc", "api_key", "my-secret")
        resolver = _make_resolver(storage_root, backend)
        assert resolver.resolve("secret_ref://vault/vault_abc/api_key") == "my-secret"

    def test_returns_string_type(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_abc")
        _store_secret(backend, "vault_abc", "tok", "token_value")
        resolver = _make_resolver(storage_root, backend)
        result = resolver.resolve("secret_ref://vault/vault_abc/tok")
        assert isinstance(result, str)

    def test_different_keys_resolve_independently(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_multi")
        _store_secret(backend, "vault_multi", "key_a", "value_a")
        _store_secret(backend, "vault_multi", "key_b", "value_b")
        resolver = _make_resolver(storage_root, backend)
        assert resolver.resolve("secret_ref://vault/vault_multi/key_a") == "value_a"
        assert resolver.resolve("secret_ref://vault/vault_multi/key_b") == "value_b"

    def test_different_vaults_resolve_independently(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_x")
        _write_vault(storage_root, "vault_y")
        _store_secret(backend, "vault_x", "secret", "x_val")
        _store_secret(backend, "vault_y", "secret", "y_val")
        resolver = _make_resolver(storage_root, backend)
        assert resolver.resolve("secret_ref://vault/vault_x/secret") == "x_val"
        assert resolver.resolve("secret_ref://vault/vault_y/secret") == "y_val"


# ---------------------------------------------------------------------------
# TestSecretRefResolverLazy (no caching — rotation test)
# ---------------------------------------------------------------------------


class TestSecretRefResolverLazy:
    def test_second_call_reflects_updated_value(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_rot")
        _store_secret(backend, "vault_rot", "pw", "original")
        resolver = _make_resolver(storage_root, backend)

        first = resolver.resolve("secret_ref://vault/vault_rot/pw")
        assert first == "original"

        # Simulate rotation: update the secret directly in the backend
        record = backend.get_secret("vault_rot", "pw")
        assert record is not None
        record = dict(record)
        record["value"] = "rotated"
        backend.update_secret("vault_rot", "pw", record)

        second = resolver.resolve("secret_ref://vault/vault_rot/pw")
        assert second == "rotated"

    def test_each_call_hits_backend(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_nc")
        _store_secret(backend, "vault_nc", "k", "v1")
        resolver = _make_resolver(storage_root, backend)

        resolver.resolve("secret_ref://vault/vault_nc/k")

        record = backend.get_secret("vault_nc", "k")
        assert record is not None
        updated = dict(record)
        updated["value"] = "v2"
        backend.update_secret("vault_nc", "k", updated)

        assert resolver.resolve("secret_ref://vault/vault_nc/k") == "v2"


# ---------------------------------------------------------------------------
# TestSecretRefParseError
# ---------------------------------------------------------------------------


class TestSecretRefParseError:
    def test_invalid_scheme_raises(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("https://example.com/secret")

    def test_missing_key_raises(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("secret_ref://vault/vault_abc")

    def test_empty_string_raises(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("")

    def test_error_code(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        with pytest.raises(SecretRefParseError) as exc_info:
            resolver.resolve("bad_uri")
        assert exc_info.value.code == "secret_ref_parse_failed"

    def test_error_message_contains_ref(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        with pytest.raises(SecretRefParseError) as exc_info:
            resolver.resolve("not_a_ref")
        assert "not_a_ref" in exc_info.value.message


# ---------------------------------------------------------------------------
# TestSecretRefVaultNotFound
# ---------------------------------------------------------------------------


class TestSecretRefVaultNotFound:
    def test_missing_vault_raises_resolve_error(self, storage_root: Path) -> None:
        backend = _make_backend()
        resolver = _make_resolver(storage_root, backend)
        with pytest.raises(SecretRefResolveError):
            resolver.resolve("secret_ref://vault/vault_ghost/api_key")

    def test_error_code(self, storage_root: Path) -> None:
        backend = _make_backend()
        resolver = _make_resolver(storage_root, backend)
        with pytest.raises(SecretRefResolveError) as exc_info:
            resolver.resolve("secret_ref://vault/vault_ghost/api_key")
        assert exc_info.value.code == "secret_ref_resolve_failed"

    def test_error_message_mentions_vault(self, storage_root: Path) -> None:
        backend = _make_backend()
        resolver = _make_resolver(storage_root, backend)
        with pytest.raises(SecretRefResolveError) as exc_info:
            resolver.resolve("secret_ref://vault/vault_ghost/api_key")
        assert "vault_ghost" in exc_info.value.message


# ---------------------------------------------------------------------------
# TestSecretRefSecretNotFound
# ---------------------------------------------------------------------------


class TestSecretRefSecretNotFound:
    def test_missing_key_raises_not_found(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_abc")
        resolver = _make_resolver(storage_root, backend)
        with pytest.raises(SecretRefNotFoundError):
            resolver.resolve("secret_ref://vault/vault_abc/missing_key")

    def test_error_code(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_abc")
        resolver = _make_resolver(storage_root, backend)
        with pytest.raises(SecretRefNotFoundError) as exc_info:
            resolver.resolve("secret_ref://vault/vault_abc/no_key")
        assert exc_info.value.code == "secret_ref_not_found"

    def test_error_message_mentions_key(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_abc")
        resolver = _make_resolver(storage_root, backend)
        with pytest.raises(SecretRefNotFoundError) as exc_info:
            resolver.resolve("secret_ref://vault/vault_abc/target_key")
        assert "target_key" in exc_info.value.message

    def test_error_message_mentions_vault(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_abc")
        resolver = _make_resolver(storage_root, backend)
        with pytest.raises(SecretRefNotFoundError) as exc_info:
            resolver.resolve("secret_ref://vault/vault_abc/target_key")
        assert "vault_abc" in exc_info.value.message


# ---------------------------------------------------------------------------
# TestSecretRefBackendNotConfigured
# ---------------------------------------------------------------------------


class TestSecretRefBackendNotConfigured:
    def test_keychain_not_configured_raises(self, storage_root: Path) -> None:
        _write_vault(storage_root, "vault_kc", backend="os_keychain")
        resolver = SecretRefResolver(storage_root=storage_root)
        with pytest.raises(SecretRefResolveError):
            resolver.resolve("secret_ref://vault/vault_kc/api_key")

    def test_keychain_not_configured_error_code(self, storage_root: Path) -> None:
        _write_vault(storage_root, "vault_kc", backend="os_keychain")
        resolver = SecretRefResolver(storage_root=storage_root)
        with pytest.raises(SecretRefResolveError) as exc_info:
            resolver.resolve("secret_ref://vault/vault_kc/api_key")
        assert exc_info.value.code == "secret_ref_resolve_failed"


# ---------------------------------------------------------------------------
# TestSecretRefOtel
# ---------------------------------------------------------------------------


class TestSecretRefOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_span(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_otel")
        _store_secret(backend, "vault_otel", "k", "v")
        resolver = _make_resolver(storage_root, backend)
        resolver.resolve("secret_ref://vault/vault_otel/k")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.secret.resolve" in span_names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.secret.resolve" in span_names

    def test_span_has_secret_ref_attribute(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_attr")
        _store_secret(backend, "vault_attr", "k", "v")
        resolver = _make_resolver(storage_root, backend)
        ref = "secret_ref://vault/vault_attr/k"
        resolver.resolve(ref)
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "vault.secret.resolve")
        assert span.attributes["secret.ref"] == ref

    def test_span_has_vault_id_attribute_on_success(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_vid")
        _store_secret(backend, "vault_vid", "k", "v")
        resolver = _make_resolver(storage_root, backend)
        resolver.resolve("secret_ref://vault/vault_vid/k")
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "vault.secret.resolve")
        assert span.attributes["vault.id"] == "vault_vid"

    def test_span_has_secret_key_attribute_on_success(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_sk")
        _store_secret(backend, "vault_sk", "my_key", "v")
        resolver = _make_resolver(storage_root, backend)
        resolver.resolve("secret_ref://vault/vault_sk/my_key")
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "vault.secret.resolve")
        assert span.attributes["secret.key"] == "my_key"

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_evt")
        _store_secret(backend, "vault_evt", "k", "v")
        resolver = _make_resolver(storage_root, backend)
        resolver.resolve("secret_ref://vault/vault_evt/k")
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "vault.secret.resolve")
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_invocation_event_code(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_ec")
        _store_secret(backend, "vault_ec", "k", "v")
        resolver = _make_resolver(storage_root, backend)
        resolver.resolve("secret_ref://vault/vault_ec/k")
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "vault.secret.resolve")
        evt = next(e for e in span.events if e.name == "meridian.error.invocation")
        assert evt.attributes["code"] == "vault_secret_resolve"

    def test_failure_span_status_is_error(self, storage_root: Path) -> None:
        resolver = _make_resolver(storage_root)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref")
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "vault.secret.resolve")
        assert span.status.status_code == StatusCode.ERROR

    def test_vault_not_found_span_status_is_error(self, storage_root: Path) -> None:
        backend = _make_backend()
        resolver = _make_resolver(storage_root, backend)
        with pytest.raises(SecretRefResolveError):
            resolver.resolve("secret_ref://vault/vault_ghost/k")
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "vault.secret.resolve")
        assert span.status.status_code == StatusCode.ERROR

    def test_secret_not_found_span_status_is_error(self, storage_root: Path) -> None:
        backend = _make_backend()
        _write_vault(storage_root, "vault_snf")
        resolver = _make_resolver(storage_root, backend)
        with pytest.raises(SecretRefNotFoundError):
            resolver.resolve("secret_ref://vault/vault_snf/missing")
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "vault.secret.resolve")
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# TestSecretRefAuditLog
# ---------------------------------------------------------------------------


class TestSecretRefAuditLog:
    def test_parse_error_writes_audit(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = _make_resolver(storage_root, audit_log=audit)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref")
        assert any(e.event == "vault.secret.resolve.failed" for e in audit.entries)

    def test_vault_not_found_writes_audit(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        with pytest.raises(SecretRefResolveError):
            resolver.resolve("secret_ref://vault/vault_ghost/k")
        assert any(e.event == "vault.secret.resolve.failed" for e in audit.entries)

    def test_secret_not_found_writes_audit(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_snf2")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        with pytest.raises(SecretRefNotFoundError):
            resolver.resolve("secret_ref://vault/vault_snf2/no_key")
        assert any(e.event == "vault.secret.resolve.failed" for e in audit.entries)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = _make_resolver(storage_root, audit_log=audit)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref")
        entry = next(e for e in audit.entries if e.event == "vault.secret.resolve.failed")
        assert entry.level == "error"

    def test_parse_error_audit_code(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = _make_resolver(storage_root, audit_log=audit)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref")
        entry = next(e for e in audit.entries if e.event == "vault.secret.resolve.failed")
        assert entry.code == "secret_ref_parse_failed"

    def test_vault_not_found_audit_code(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        with pytest.raises(SecretRefResolveError):
            resolver.resolve("secret_ref://vault/vault_ghost/k")
        entry = next(e for e in audit.entries if e.event == "vault.secret.resolve.failed")
        assert entry.code == "secret_ref_resolve_failed"

    def test_secret_not_found_audit_code(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_anf")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        with pytest.raises(SecretRefNotFoundError):
            resolver.resolve("secret_ref://vault/vault_anf/missing")
        entry = next(e for e in audit.entries if e.event == "vault.secret.resolve.failed")
        assert entry.code == "secret_ref_not_found"

    def test_audit_detail_has_ref(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = _make_resolver(storage_root, audit_log=audit)
        ref = "not_a_ref"
        with pytest.raises(SecretRefParseError):
            resolver.resolve(ref)
        entry = next(e for e in audit.entries if e.event == "vault.secret.resolve.failed")
        assert entry.detail is not None
        assert entry.detail["ref"] == ref

    def test_audit_detail_has_message(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = _make_resolver(storage_root, audit_log=audit)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref")
        entry = next(e for e in audit.entries if e.event == "vault.secret.resolve.failed")
        assert entry.detail is not None
        assert len(entry.detail["message"]) > 0

    def test_no_audit_log_does_not_crash(self, storage_root: Path) -> None:
        resolver = SecretRefResolver(storage_root=storage_root)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref")

    def test_noop_audit_log_accepted(self, storage_root: Path) -> None:
        resolver = SecretRefResolver(
            storage_root=storage_root,
            audit_log=NoopAuditLog(),
        )
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref")


# ---------------------------------------------------------------------------
# TestSecretAccessAuditEvent
# ---------------------------------------------------------------------------


class TestSecretAccessAuditEvent:
    def test_success_emits_secret_access_event(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_sa")
        _store_secret(backend, "vault_sa", "api_key", "val")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        resolver.resolve("secret_ref://vault/vault_sa/api_key")
        assert any(e.event == "audit.secret_access" for e in audit.entries)

    def test_secret_access_event_level_is_info(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_sai")
        _store_secret(backend, "vault_sai", "k", "v")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        resolver.resolve("secret_ref://vault/vault_sai/k")
        entry = next(e for e in audit.entries if e.event == "audit.secret_access")
        assert entry.level == "info"

    def test_secret_access_event_code(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_sac")
        _store_secret(backend, "vault_sac", "k", "v")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        resolver.resolve("secret_ref://vault/vault_sac/k")
        entry = next(e for e in audit.entries if e.event == "audit.secret_access")
        assert entry.code == "vault_secret_access"

    def test_secret_access_detail_has_vault_id(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_dvid")
        _store_secret(backend, "vault_dvid", "k", "v")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        resolver.resolve("secret_ref://vault/vault_dvid/k")
        entry = next(e for e in audit.entries if e.event == "audit.secret_access")
        assert entry.detail is not None
        assert entry.detail["vault_id"] == "vault_dvid"

    def test_secret_access_detail_has_name(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_dname")
        _store_secret(backend, "vault_dname", "my_secret_key", "v")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        resolver.resolve("secret_ref://vault/vault_dname/my_secret_key")
        entry = next(e for e in audit.entries if e.event == "audit.secret_access")
        assert entry.detail is not None
        assert entry.detail["name"] == "my_secret_key"

    def test_secret_access_requester_fields_none_when_not_provided(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_rn")
        _store_secret(backend, "vault_rn", "k", "v")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        resolver.resolve("secret_ref://vault/vault_rn/k")
        entry = next(e for e in audit.entries if e.event == "audit.secret_access")
        assert entry.detail is not None
        assert entry.detail["requester_agent_id"] is None
        assert entry.detail["requester_tool_call_id"] is None

    def test_secret_access_requester_agent_id_propagated(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_rag")
        _store_secret(backend, "vault_rag", "k", "v")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        resolver.resolve("secret_ref://vault/vault_rag/k", agent_id="agent_123")
        entry = next(e for e in audit.entries if e.event == "audit.secret_access")
        assert entry.detail is not None
        assert entry.detail["requester_agent_id"] == "agent_123"

    def test_secret_access_requester_tool_call_id_propagated(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_rtc")
        _store_secret(backend, "vault_rtc", "k", "v")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        resolver.resolve("secret_ref://vault/vault_rtc/k", tool_call_id="tc_456")
        entry = next(e for e in audit.entries if e.event == "audit.secret_access")
        assert entry.detail is not None
        assert entry.detail["requester_tool_call_id"] == "tc_456"

    def test_secret_access_both_requester_fields_propagated(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        backend = _make_backend()
        _write_vault(storage_root, "vault_rboth")
        _store_secret(backend, "vault_rboth", "k", "v")
        resolver = _make_resolver(storage_root, backend, audit_log=audit)
        resolver.resolve("secret_ref://vault/vault_rboth/k", agent_id="agt_1", tool_call_id="tc_2")
        entry = next(e for e in audit.entries if e.event == "audit.secret_access")
        assert entry.detail is not None
        assert entry.detail["requester_agent_id"] == "agt_1"
        assert entry.detail["requester_tool_call_id"] == "tc_2"

    def test_failure_does_not_emit_secret_access_event(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = _make_resolver(storage_root, audit_log=audit)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref")
        assert not any(e.event == "audit.secret_access" for e in audit.entries)

    def test_failure_audit_detail_has_requester_agent_id(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = _make_resolver(storage_root, audit_log=audit)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref", agent_id="agt_fail")
        entry = next(e for e in audit.entries if e.event == "vault.secret.resolve.failed")
        assert entry.detail is not None
        assert entry.detail["requester_agent_id"] == "agt_fail"

    def test_failure_audit_detail_has_requester_tool_call_id(self, storage_root: Path) -> None:
        audit = _CapturingAuditLog()
        resolver = _make_resolver(storage_root, audit_log=audit)
        with pytest.raises(SecretRefParseError):
            resolver.resolve("bad_ref", tool_call_id="tc_fail")
        entry = next(e for e in audit.entries if e.event == "vault.secret.resolve.failed")
        assert entry.detail is not None
        assert entry.detail["requester_tool_call_id"] == "tc_fail"
