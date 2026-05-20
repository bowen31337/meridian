"""
Vault leak soak CI test conformance suite.

Tests cover:
  - POST /v1/x/ci/vault-leak-soak-run returns 200 when no leaks found.
  - Response body has run_id, status, canary_count, leak_count, leaks fields.
  - run_id has "vault_soak_" prefix.
  - status is "passed" on success.
  - canary_count equals CANARY_COUNT when no override supplied.
  - leak_count is 0 on success.
  - leaks list is empty on success.
  - Returns 422 with code "vault_leak_soak_failed" when a file leak is detected.
  - Error message mentions the leak count.
  - Error message mentions the source type.
  - Canary written to a subdirectory of storage_root is also detected.
  - On failure: audit log entry "vault.leak.soak.run.failed" written.
  - On failure: audit entry level is "error".
  - On failure: audit detail has run_id.
  - On failure: audit detail has leak_count.
  - On failure: audit detail has first_leak_source.
  - On success: audit log entry "vault.leak.soak.ran" written.
  - On success: audit entry level is "info".
  - On success: audit detail has run_id.
  - On success: audit detail has canary_count.
  - On success: audit detail has leak_count equal to 0.
  - OTel span "vault.leak.soak.run" emitted on success.
  - OTel span "vault.leak.soak.run" emitted on failure.
  - OTel span set to ERROR status on failure.
  - OTel span has non-error status on success.
  - Span carries vault.leak.soak.canary_count attribute.
  - Span carries vault.leak.soak.leak_count attribute.
  - Hook stdin capture file created under soak_captures/{run_id}/.
  - Hook stdin capture file does not contain canary plaintext.
  - Event log capture file created under soak_captures/{run_id}/.
  - Event log capture file does not contain canary plaintext.
  - Log capture file created under soak_captures/{run_id}/.
  - Log capture file does not contain canary plaintext.
  - _scan_text returns an excerpt when canary found in text.
  - _scan_text returns None when canary not in text.
  - _scan_text excerpt is short (bounded, not the whole text).
  - _scan_storage_root returns empty list when storage_root does not exist.
  - _scan_storage_root returns empty list when no canary in any file.
  - _scan_storage_root returns leak record when canary found in file.
  - _scan_storage_root leak record source is "file".
  - _scan_storage_root leak record location is the relative path.
  - _scan_storage_root leak record has excerpt field.
  - _scan_storage_root scans files in nested subdirectories.
  - VaultLeakSoakError has http_status 422.
  - VaultLeakSoakError has code "vault_leak_soak_failed".
  - CANARY_COUNT is 4.
  - create_app wires the vault leak soak router when storage_root is supplied.
  - create_app omits the vault leak soak route when storage_root is None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core_errors import AuditLog, AuditLogEntry, HandlerOptions, install_error_handler
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._vault_leak_soak import (
    CANARY_COUNT,
    VaultLeakSoakError,
    _scan_storage_root,
    _scan_text,
    make_vault_leak_soak_router,
)

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KNOWN_CANARY = "vault_soak_test_canary_known_plaintext_xyz"


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_client(
    storage_root: Path,
    *,
    canary_override: list[str] | None = None,
) -> TestClient:
    audit = FileAuditLog(storage_root)
    router = make_vault_leak_soak_router(
        audit_log=audit,
        storage_root=storage_root,
        _canary_override=canary_override,
    )
    app = FastAPI()
    app.include_router(router)
    install_error_handler(app, HandlerOptions(audit_log=audit))
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Unit: _scan_text
# ---------------------------------------------------------------------------


class TestScanText:
    def test_returns_excerpt_when_canary_found(self) -> None:
        result = _scan_text("hello canary_val world", "canary_val")
        assert result is not None
        assert "canary_val" in result

    def test_returns_none_when_canary_absent(self) -> None:
        assert _scan_text("hello world", "not_here") is None

    def test_empty_text_returns_none(self) -> None:
        assert _scan_text("", "canary") is None

    def test_excerpt_is_bounded(self) -> None:
        # Surrounding context is capped at 20 chars on each side.
        long_prefix = "x" * 100
        long_suffix = "y" * 100
        canary = "CANARY"
        text = long_prefix + canary + long_suffix
        excerpt = _scan_text(text, canary)
        assert excerpt is not None
        assert len(excerpt) <= 20 + len(canary) + 20

    def test_canary_at_start_of_text(self) -> None:
        result = _scan_text("CANARY rest of text", "CANARY")
        assert result is not None
        assert "CANARY" in result

    def test_canary_at_end_of_text(self) -> None:
        result = _scan_text("text then CANARY", "CANARY")
        assert result is not None
        assert "CANARY" in result


# ---------------------------------------------------------------------------
# Unit: _scan_storage_root
# ---------------------------------------------------------------------------


class TestScanStorageRoot:
    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        assert _scan_storage_root(missing, ["canary"]) == []

    def test_no_canary_in_files_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("safe content only")
        assert _scan_storage_root(tmp_path, ["canary_xyz"]) == []

    def test_canary_in_file_returns_leak(self, tmp_path: Path) -> None:
        (tmp_path / "log.txt").write_text(f"prefix {_KNOWN_CANARY} suffix")
        leaks = _scan_storage_root(tmp_path, [_KNOWN_CANARY])
        assert len(leaks) == 1

    def test_leak_source_is_file(self, tmp_path: Path) -> None:
        (tmp_path / "log.txt").write_text(_KNOWN_CANARY)
        leaks = _scan_storage_root(tmp_path, [_KNOWN_CANARY])
        assert leaks[0]["source"] == "file"

    def test_leak_location_is_relative_path(self, tmp_path: Path) -> None:
        (tmp_path / "log.txt").write_text(_KNOWN_CANARY)
        leaks = _scan_storage_root(tmp_path, [_KNOWN_CANARY])
        assert leaks[0]["location"] == "log.txt"

    def test_leak_has_excerpt_field(self, tmp_path: Path) -> None:
        (tmp_path / "log.txt").write_text(_KNOWN_CANARY)
        leaks = _scan_storage_root(tmp_path, [_KNOWN_CANARY])
        assert "excerpt" in leaks[0]
        assert _KNOWN_CANARY in leaks[0]["excerpt"]

    def test_canary_in_nested_subdir(self, tmp_path: Path) -> None:
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        (sub / "deep.log").write_text(_KNOWN_CANARY)
        leaks = _scan_storage_root(tmp_path, [_KNOWN_CANARY])
        assert len(leaks) == 1
        assert "a/b/deep.log" in leaks[0]["location"] or "a" + "/" in leaks[0]["location"]

    def test_multiple_canaries_detected(self, tmp_path: Path) -> None:
        (tmp_path / "f1.txt").write_text("canary_a here")
        (tmp_path / "f2.txt").write_text("canary_b here")
        leaks = _scan_storage_root(tmp_path, ["canary_a", "canary_b"])
        assert len(leaks) == 2

    def test_only_matching_canary_reported(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("canary_present here")
        leaks = _scan_storage_root(tmp_path, ["canary_present", "canary_absent"])
        assert len(leaks) == 1
        assert "canary_present" in leaks[0]["excerpt"]


# ---------------------------------------------------------------------------
# Unit: VaultLeakSoakError
# ---------------------------------------------------------------------------


class TestVaultLeakSoakError:
    def test_http_status_is_422(self) -> None:
        err = VaultLeakSoakError(message="leak", timestamp="2026-01-01T00:00:00+00:00")
        assert err.http_status() == 422

    def test_code_is_vault_leak_soak_failed(self) -> None:
        err = VaultLeakSoakError(message="leak", timestamp="2026-01-01T00:00:00+00:00")
        assert err.code == "vault_leak_soak_failed"


# ---------------------------------------------------------------------------
# Unit: CANARY_COUNT
# ---------------------------------------------------------------------------


class TestCanaryCount:
    def test_canary_count_is_4(self) -> None:
        assert CANARY_COUNT == 4


# ---------------------------------------------------------------------------
# Endpoint: happy path
# ---------------------------------------------------------------------------


class TestVaultLeakSoakHappyPath:
    def test_no_leaks_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/ci/vault-leak-soak-run")
        assert resp.status_code == 200

    def test_run_id_has_vault_soak_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        assert body["run_id"].startswith("vault_soak_")

    def test_status_is_passed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        assert body["status"] == "passed"

    def test_canary_count_equals_default_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        assert body["canary_count"] == CANARY_COUNT

    def test_leak_count_is_zero(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        assert body["leak_count"] == 0

    def test_leaks_list_is_empty(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        assert body["leaks"] == []

    def test_response_has_all_expected_fields(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        for field in ("run_id", "status", "canary_count", "leak_count", "leaks"):
            assert field in body

    def test_canary_count_equals_override_length(self, storage_root: Path) -> None:
        overrides = [
            "override_canary_alpha_unique_99",
            "override_canary_beta_unique_99",
            "override_canary_gamma_unique_99",
        ]
        client = _make_client(storage_root, canary_override=overrides)
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        assert body["canary_count"] == 3


# ---------------------------------------------------------------------------
# Endpoint: failure detection
# ---------------------------------------------------------------------------


class TestVaultLeakSoakFailureDetection:
    def test_file_leak_returns_422(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        resp = client.post("/v1/x/ci/vault-leak-soak-run")
        assert resp.status_code == 422

    def test_file_leak_error_code(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        assert body["error"]["code"] == "vault_leak_soak_failed"

    def test_error_message_mentions_leak_count(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        assert "1" in body["error"]["message"]

    def test_error_message_mentions_source(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        assert "file" in body["error"]["message"]

    def test_leak_in_subdirectory_detected(self, storage_root: Path) -> None:
        sub = storage_root / "events" / "2026" / "01"
        sub.mkdir(parents=True)
        (sub / "session.ndjson").write_text(f'{{"value": "{_KNOWN_CANARY}"}}')
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        resp = client.post("/v1/x/ci/vault-leak-soak-run")
        assert resp.status_code == 422

    def test_leak_in_log_file_detected(self, storage_root: Path) -> None:
        (storage_root / "meridiand.log").write_text(
            f"2026-01-01 INFO secret resolved: {_KNOWN_CANARY}\n"
        )
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        resp = client.post("/v1/x/ci/vault-leak-soak-run")
        assert resp.status_code == 422

    def test_no_false_positive_without_canary(self, storage_root: Path) -> None:
        (storage_root / "safe.txt").write_text("secret_ref://vault/v1/key1 safe content")
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        resp = client.post("/v1/x/ci/vault-leak-soak-run")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Endpoint: exercises write safe capture files
# ---------------------------------------------------------------------------


class TestVaultLeakSoakExercises:
    def test_hook_stdin_capture_file_created(self, storage_root: Path) -> None:
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        run_id = body["run_id"]
        capture = storage_root / "soak_captures" / run_id / "hook_stdin_result.json"
        assert capture.exists()

    def test_hook_stdin_capture_has_no_canary_plaintext(self, storage_root: Path) -> None:
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        run_id = body["run_id"]
        capture = storage_root / "soak_captures" / run_id / "hook_stdin_result.json"
        assert _KNOWN_CANARY not in capture.read_text()

    def test_event_log_capture_file_created(self, storage_root: Path) -> None:
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        run_id = body["run_id"]
        capture = storage_root / "soak_captures" / run_id / "event_log_entries.ndjson"
        assert capture.exists()

    def test_event_log_capture_has_no_canary_plaintext(self, storage_root: Path) -> None:
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        run_id = body["run_id"]
        capture = storage_root / "soak_captures" / run_id / "event_log_entries.ndjson"
        assert _KNOWN_CANARY not in capture.read_text()

    def test_log_capture_file_created(self, storage_root: Path) -> None:
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        run_id = body["run_id"]
        capture = storage_root / "soak_captures" / run_id / "log_capture.txt"
        assert capture.exists()

    def test_log_capture_has_no_canary_plaintext(self, storage_root: Path) -> None:
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        run_id = body["run_id"]
        capture = storage_root / "soak_captures" / run_id / "log_capture.txt"
        assert _KNOWN_CANARY not in capture.read_text()

    def test_hook_stdin_capture_has_vault_ref_uri(self, storage_root: Path) -> None:
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        run_id = body["run_id"]
        capture = storage_root / "soak_captures" / run_id / "hook_stdin_result.json"
        content = capture.read_text()
        assert "secret_ref://vault/" in content

    def test_event_log_capture_has_vault_ref_uri(self, storage_root: Path) -> None:
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        body = client.post("/v1/x/ci/vault-leak-soak-run").json()
        run_id = body["run_id"]
        capture = storage_root / "soak_captures" / run_id / "event_log_entries.ndjson"
        content = capture.read_text()
        assert "secret_ref://vault/" in content


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestVaultLeakSoakAudit:
    def test_success_writes_ran_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/ci/vault-leak-soak-run")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.leak.soak.ran" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/ci/vault-leak-soak-run")
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "vault.leak.soak.ran")
        assert record["level"] == "info"

    def test_success_audit_detail_has_run_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/ci/vault-leak-soak-run")
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "vault.leak.soak.ran")
        assert "run_id" in record["detail"] and record["detail"]["run_id"]

    def test_success_audit_detail_has_canary_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/ci/vault-leak-soak-run")
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "vault.leak.soak.ran")
        assert "canary_count" in record["detail"]
        assert record["detail"]["canary_count"] == CANARY_COUNT

    def test_success_audit_detail_has_leak_count_zero(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/ci/vault-leak-soak-run")
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "vault.leak.soak.ran")
        assert record["detail"]["leak_count"] == 0

    def test_failure_writes_failed_audit_entry(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        client.post("/v1/x/ci/vault-leak-soak-run")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.leak.soak.run.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        client.post("/v1/x/ci/vault-leak-soak-run")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "vault.leak.soak.run.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_run_id(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        client.post("/v1/x/ci/vault-leak-soak-run")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "vault.leak.soak.run.failed"
        )
        assert "run_id" in record["detail"] and record["detail"]["run_id"]

    def test_failure_audit_detail_has_leak_count(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        client.post("/v1/x/ci/vault-leak-soak-run")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "vault.leak.soak.run.failed"
        )
        assert record["detail"]["leak_count"] >= 1

    def test_failure_audit_detail_has_first_leak_source(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        client.post("/v1/x/ci/vault-leak-soak-run")
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "vault.leak.soak.run.failed"
        )
        assert record["detail"]["first_leak_source"] == "file"


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestVaultLeakSoakOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _get_soak_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("vault.leak.soak.run")

    def test_success_emits_vault_leak_soak_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/ci/vault-leak-soak-run")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.leak.soak.run" in span_names

    def test_failure_emits_vault_leak_soak_span(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        client.post("/v1/x/ci/vault-leak-soak-run")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.leak.soak.run" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        client.post("/v1/x/ci/vault-leak-soak-run")
        span = self._get_soak_span()
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_non_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_client(storage_root)
        client.post("/v1/x/ci/vault-leak-soak-run")
        span = self._get_soak_span()
        assert span is not None
        assert span.status.status_code != StatusCode.ERROR

    def test_span_has_canary_count_attribute(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/ci/vault-leak-soak-run")
        span = self._get_soak_span()
        assert span is not None
        assert span.attributes["vault.leak.soak.canary_count"] == CANARY_COUNT

    def test_span_has_leak_count_attribute_zero_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/ci/vault-leak-soak-run")
        span = self._get_soak_span()
        assert span is not None
        assert span.attributes["vault.leak.soak.leak_count"] == 0

    def test_span_has_leak_count_attribute_nonzero_on_failure(self, storage_root: Path) -> None:
        (storage_root / "leaked.txt").write_text(_KNOWN_CANARY)
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        client.post("/v1/x/ci/vault-leak-soak-run")
        span = self._get_soak_span()
        assert span is not None
        assert span.attributes["vault.leak.soak.leak_count"] >= 1

    def test_span_attributes_do_not_contain_canary_plaintext(self, storage_root: Path) -> None:
        client = _make_client(storage_root, canary_override=[_KNOWN_CANARY])
        client.post("/v1/x/ci/vault-leak-soak-run")
        span = self._get_soak_span()
        assert span is not None
        for val in span.attributes.values():
            assert _KNOWN_CANARY not in str(val)


# ---------------------------------------------------------------------------
# create_app integration
# ---------------------------------------------------------------------------


class TestCreateAppIntegration:
    def test_route_present_when_storage_root_set(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/ci/vault-leak-soak-run")
        assert resp.status_code != 404

    def test_route_absent_when_no_storage_root(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/ci/vault-leak-soak-run")
        assert resp.status_code == 404
