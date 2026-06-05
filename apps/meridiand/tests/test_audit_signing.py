"""
Ed25519 audit log signing conformance suite.

Tests cover:
  - DaemonSigningKey: generates audit_signing.key and audit_signing.pub on first init.
  - DaemonSigningKey: private key file has mode 0o600.
  - DaemonSigningKey: public key file is written alongside the private key.
  - DaemonSigningKey: second init loads the same key (sign output is stable).
  - DaemonSigningKey: sign() returns a valid base64-encoded string.
  - DaemonSigningKey: signature verifies against the stored public key.
  - DaemonSigningKey: public_key_path returns the path to audit_signing.pub.
  - FileAuditLog (unsigned): write() produces a line with no "sig" field.
  - FileAuditLog (signed): write() appends a "sig" field to each NDJSON line.
  - FileAuditLog (signed): "sig" field is a valid Ed25519 signature.
  - FileAuditLog (signed): signed payload is the canonical JSON (sort_keys=True) of the
    record without the "sig" field.
  - FileAuditLog (signed): entry with detail signs correctly.
  - FileAuditLog (signed): entry without detail signs correctly.
  - FileAuditLog (signed): sign failure writes unsigned audit.sign.failed entry to the log.
  - FileAuditLog (signed): sign failure raises AuditSignFailedError.
  - AuditSignFailedError: has code "audit_sign_failed".
  - AuditSignFailedError: http_status() returns 500.
  - AuditSigningConfig: enabled defaults to False.
  - MeridianConfig: has audit_signing field with AuditSigningConfig default.
  - MeridianConfig: audit_signing.enabled can be set to True via YAML.
  - init_services: FileAuditLog has no signing key when audit_signing.enabled is False.
  - init_services: FileAuditLog has a signing key when audit_signing.enabled is True.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import stat
from unittest.mock import patch

from core_errors import AuditLogEntry
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from meridiand._audit import AuditSignFailedError, FileAuditLog
from meridiand._config import AuditSigningConfig, MeridianConfig
from meridiand._services import init_services
from meridiand._signing import DaemonSigningKey
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    *,
    level: str = "info",
    event: str = "test.event",
    code: str = "test_code",
    timestamp: str = "2026-01-01T00:00:00+00:00",
    detail: dict | None = None,
) -> AuditLogEntry:
    return AuditLogEntry(
        level=level,  # type: ignore[arg-type]
        event=event,
        code=code,
        timestamp=timestamp,
        detail=detail,
    )


def _load_pub_key(pub_path: Path) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(pub_path.read_bytes())


def _read_ndjson_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# DaemonSigningKey
# ---------------------------------------------------------------------------


def test_signing_key_generates_key_files(tmp_path):
    DaemonSigningKey(tmp_path)
    assert (tmp_path / "audit_signing.key").exists()
    assert (tmp_path / "audit_signing.pub").exists()


def test_signing_key_private_file_permissions(tmp_path):
    DaemonSigningKey(tmp_path)
    mode = stat.S_IMODE((tmp_path / "audit_signing.key").stat().st_mode)
    assert mode == 0o600


def test_signing_key_public_file_written(tmp_path):
    key = DaemonSigningKey(tmp_path)
    pub_bytes = (tmp_path / "audit_signing.pub").read_bytes()
    assert len(pub_bytes) == 32
    assert key.public_key_path == tmp_path / "audit_signing.pub"


def test_signing_key_loads_existing_key(tmp_path):
    key1 = DaemonSigningKey(tmp_path)
    data = b"hello world"
    sig1 = key1.sign(data)

    key2 = DaemonSigningKey(tmp_path)
    sig2 = key2.sign(data)

    assert sig1 == sig2


def test_signing_key_sign_returns_base64(tmp_path):
    key = DaemonSigningKey(tmp_path)
    sig = key.sign(b"test payload")
    decoded = base64.b64decode(sig)
    assert len(decoded) == 64


def test_signing_key_signature_verifiable(tmp_path):
    key = DaemonSigningKey(tmp_path)
    data = b"audit entry bytes"
    sig = key.sign(data)

    pub_key = _load_pub_key(key.public_key_path)
    # verify() raises InvalidSignature on failure; no return value on success
    pub_key.verify(base64.b64decode(sig), data)


def test_signing_key_public_key_path_property(tmp_path):
    key = DaemonSigningKey(tmp_path)
    assert key.public_key_path == tmp_path / "audit_signing.pub"


# ---------------------------------------------------------------------------
# FileAuditLog — unsigned (no signing key)
# ---------------------------------------------------------------------------


def test_file_audit_log_unsigned_no_sig_field(tmp_path):
    log = FileAuditLog(tmp_path)
    log.write(_make_entry())
    lines = _read_ndjson_lines(tmp_path / "audit.ndjson")
    assert len(lines) == 1
    assert "sig" not in lines[0]


def test_file_audit_log_unsigned_writes_fields_correctly(tmp_path):
    log = FileAuditLog(tmp_path)
    log.write(_make_entry(event="foo.bar", code="foo_bar", level="error"))
    lines = _read_ndjson_lines(tmp_path / "audit.ndjson")
    assert lines[0]["event"] == "foo.bar"
    assert lines[0]["code"] == "foo_bar"
    assert lines[0]["level"] == "error"


# ---------------------------------------------------------------------------
# FileAuditLog — signed
# ---------------------------------------------------------------------------


def test_file_audit_log_signed_has_sig_field(tmp_path):
    signing_key = DaemonSigningKey(tmp_path)
    log = FileAuditLog(tmp_path, signing_key=signing_key)
    log.write(_make_entry())
    lines = _read_ndjson_lines(tmp_path / "audit.ndjson")
    assert "sig" in lines[0]


def test_file_audit_log_signed_sig_is_valid_ed25519(tmp_path):
    signing_key = DaemonSigningKey(tmp_path)
    log = FileAuditLog(tmp_path, signing_key=signing_key)
    log.write(_make_entry())
    lines = _read_ndjson_lines(tmp_path / "audit.ndjson")

    pub_key = _load_pub_key(signing_key.public_key_path)
    line = lines[0]
    sig_bytes = base64.b64decode(line["sig"])
    # Reconstruct the canonical payload: record without "sig", keys sorted
    payload_record = {k: v for k, v in line.items() if k != "sig"}
    payload = json.dumps(payload_record, separators=(",", ":"), sort_keys=True).encode()
    pub_key.verify(sig_bytes, payload)  # raises InvalidSignature if wrong


def test_file_audit_log_signed_payload_is_canonical_sorted_no_sig(tmp_path):
    signing_key = DaemonSigningKey(tmp_path)
    log = FileAuditLog(tmp_path, signing_key=signing_key)
    entry = _make_entry(
        level="warn",
        event="something.happened",
        code="something_happened",
        timestamp="2026-05-20T12:00:00+00:00",
    )
    log.write(entry)
    lines = _read_ndjson_lines(tmp_path / "audit.ndjson")
    line = lines[0]

    expected_payload = json.dumps(
        {k: v for k, v in line.items() if k != "sig"},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()

    pub_key = _load_pub_key(signing_key.public_key_path)
    pub_key.verify(base64.b64decode(line["sig"]), expected_payload)


def test_file_audit_log_signed_with_detail(tmp_path):
    signing_key = DaemonSigningKey(tmp_path)
    log = FileAuditLog(tmp_path, signing_key=signing_key)
    log.write(_make_entry(detail={"path": "/v1/skills", "method": "POST"}))
    lines = _read_ndjson_lines(tmp_path / "audit.ndjson")
    line = lines[0]

    assert line["detail"] == {"path": "/v1/skills", "method": "POST"}
    pub_key = _load_pub_key(signing_key.public_key_path)
    payload = json.dumps(
        {k: v for k, v in line.items() if k != "sig"},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    pub_key.verify(base64.b64decode(line["sig"]), payload)


def test_file_audit_log_signed_without_detail(tmp_path):
    signing_key = DaemonSigningKey(tmp_path)
    log = FileAuditLog(tmp_path, signing_key=signing_key)
    log.write(_make_entry(detail=None))
    lines = _read_ndjson_lines(tmp_path / "audit.ndjson")
    line = lines[0]

    assert "detail" not in line
    pub_key = _load_pub_key(signing_key.public_key_path)
    payload = json.dumps(
        {k: v for k, v in line.items() if k != "sig"},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    pub_key.verify(base64.b64decode(line["sig"]), payload)


def test_file_audit_log_sign_failure_writes_failure_entry(tmp_path):
    signing_key = DaemonSigningKey(tmp_path)
    log = FileAuditLog(tmp_path, signing_key=signing_key)

    with (
        patch.object(signing_key, "sign", side_effect=RuntimeError("key exploded")),
        pytest.raises(AuditSignFailedError),
    ):
        log.write(_make_entry())

    lines = _read_ndjson_lines(tmp_path / "audit.ndjson")
    assert len(lines) == 1
    failure = lines[0]
    assert failure["event"] == "audit.sign.failed"
    assert failure["code"] == "audit_sign_failed"
    assert failure["level"] == "error"
    assert "sig" not in failure
    assert "key exploded" in failure["detail"]["message"]


def test_file_audit_log_sign_failure_raises_audit_sign_failed_error(tmp_path):
    signing_key = DaemonSigningKey(tmp_path)
    log = FileAuditLog(tmp_path, signing_key=signing_key)

    with (
        patch.object(signing_key, "sign", side_effect=ValueError("bad key")),
        pytest.raises(AuditSignFailedError) as exc_info,
    ):
        log.write(_make_entry())

    assert exc_info.value.code == "audit_sign_failed"


# ---------------------------------------------------------------------------
# AuditSignFailedError
# ---------------------------------------------------------------------------


def test_audit_sign_failed_error_code():
    err = AuditSignFailedError(message="boom", timestamp="2026-01-01T00:00:00+00:00")
    assert err.code == "audit_sign_failed"


def test_audit_sign_failed_error_http_status():
    err = AuditSignFailedError(message="boom", timestamp="2026-01-01T00:00:00+00:00")
    assert err.http_status() == 500


# ---------------------------------------------------------------------------
# AuditSigningConfig
# ---------------------------------------------------------------------------


def test_audit_signing_config_defaults_disabled():
    cfg = AuditSigningConfig()
    assert cfg.enabled is False


def test_audit_signing_config_can_be_enabled():
    cfg = AuditSigningConfig(enabled=True)
    assert cfg.enabled is True


# ---------------------------------------------------------------------------
# MeridianConfig
# ---------------------------------------------------------------------------


def test_meridian_config_has_audit_signing_field(tmp_path):
    cfg = MeridianConfig(storage_root=tmp_path)
    assert isinstance(cfg.audit_signing, AuditSigningConfig)
    assert cfg.audit_signing.enabled is False


def test_meridian_config_audit_signing_enabled_via_model(tmp_path):
    cfg = MeridianConfig(storage_root=tmp_path, audit_signing=AuditSigningConfig(enabled=True))
    assert cfg.audit_signing.enabled is True


# ---------------------------------------------------------------------------
# init_services
# ---------------------------------------------------------------------------


def test_init_services_no_signing_key_when_disabled(tmp_path):
    cfg = MeridianConfig(storage_root=tmp_path)
    services = init_services(cfg)
    assert services.audit_log._signing_key is None


def test_init_services_creates_signing_key_when_enabled(tmp_path):
    cfg = MeridianConfig(
        storage_root=tmp_path,
        audit_signing=AuditSigningConfig(enabled=True),
    )
    services = init_services(cfg)
    assert services.audit_log._signing_key is not None
    assert isinstance(services.audit_log._signing_key, DaemonSigningKey)
    assert (tmp_path / "audit_signing.key").exists()
