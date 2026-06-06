"""Sweep tests covering tiny gaps across multiple meridiand modules:
http_status methods, _now helpers, custom-error re-raise paths,
DaemonSigningKey concurrent-write branch.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from meridiand._cluster_extraction import ClusterExtractionError
from meridiand._hook_stdin_redaction import HookStdinRedactionError
from meridiand._logging import LoggingConfigError, _now as logging_now
from meridiand._pagination import CursorDecodeError, _now as pagination_now
from meridiand._signing import DaemonSigningKey


# ---------------------------------------------------------------------------
# http_status overrides
# ---------------------------------------------------------------------------


class TestHttpStatus:
    def test_cluster_extraction_error(self) -> None:
        err = ClusterExtractionError(message="m", timestamp="t", cause=None)
        assert err.http_status() == 500

    def test_hook_stdin_redaction_error(self) -> None:
        err = HookStdinRedactionError(message="m", timestamp="t", cause=None)
        assert err.http_status() == 500

    def test_logging_config_error(self) -> None:
        err = LoggingConfigError(message="m", timestamp="t")
        assert err.http_status() == 500

    def test_cursor_decode_error(self) -> None:
        err = CursorDecodeError(message="m", timestamp="t")
        assert err.http_status() == 400


# ---------------------------------------------------------------------------
# _now helpers — exercise the module-level helper
# ---------------------------------------------------------------------------


def test_pagination_now_returns_iso_string() -> None:
    s = pagination_now()
    assert isinstance(s, str)
    assert "T" in s


def test_logging_now_returns_iso_string() -> None:
    s = logging_now()
    assert isinstance(s, str)
    assert "T" in s


# ---------------------------------------------------------------------------
# DaemonSigningKey — concurrent FileExistsError branch
# ---------------------------------------------------------------------------


def test_signing_key_concurrent_creation_falls_back_to_existing(tmp_path: Path) -> None:
    """When os.open raises FileExistsError mid-create, key is loaded from disk."""
    # First instance creates the key normally
    key1 = DaemonSigningKey(tmp_path)
    key_bytes = (tmp_path / "audit_signing.key").read_bytes()

    real_open = os.open

    # Force a FileExistsError on the FIRST open attempt for audit_signing.key,
    # then fall through to the existing file (which key1 already wrote).
    # We need to simulate a race: signal that the key path doesn't exist
    # initially (so we take the "else" branch), then have os.open raise EEXIST.
    target = tmp_path / "audit_signing.key"
    raise_count: list[int] = []

    def _fake_open(path: str, flags: int, mode: int = 0o777) -> int:
        if str(path) == str(target) and (flags & os.O_EXCL) and not raise_count:
            raise_count.append(1)
            raise FileExistsError(17, "exists", str(path))
        return real_open(path, flags, mode)

    # Pre-condition: simulate the race by claiming "key_path.exists()" is False at probe
    with (
        patch.object(Path, "exists", lambda self: False if str(self) == str(target) else True),
        patch("meridiand._signing.os.open", side_effect=_fake_open),
    ):
        # The .key file is on disk (from key1) but our exists() override says no,
        # so we take the generate branch, then EEXIST forces fallback to read.
        key2 = DaemonSigningKey(tmp_path)

    # Verify the fallback reloaded the existing key — both signers should produce
    # signatures from the same private key.
    sig1 = key1._private_key.sign(b"x")
    sig2 = key2._private_key.sign(b"x")
    # Ed25519 is deterministic for the same private key + same message
    assert sig1 == sig2
    assert raise_count == [1]
    assert (tmp_path / "audit_signing.key").read_bytes() == key_bytes


# ---------------------------------------------------------------------------
# Budget exceeded / soft budget exceeded / user_can_continue — re-raise paths
# ---------------------------------------------------------------------------


def test_budget_exceeded_error_reraised_when_already_typed(tmp_path: Path) -> None:
    """If the handler raises BudgetExceededSessionError, the outer except re-raises it."""
    from fastapi.testclient import TestClient

    from meridiand._app import create_app
    from meridiand._audit import FileAuditLog
    from meridiand._budget_exceeded import BudgetExceededSessionError
    from storage_event_log import LocalEventLogWriter

    audit = FileAuditLog(tmp_path)
    writer = LocalEventLogWriter(tmp_path)
    err_to_raise = BudgetExceededSessionError(
        message="pre-typed", timestamp=pagination_now(), cause=None
    )

    async def _boom(*_a: object, **_k: object) -> None:
        raise err_to_raise

    with patch.object(writer, "append", _boom):
        app = create_app(audit, storage_root=tmp_path, event_log=writer)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/x/sessions/s1/budget-exceeded",
            json={"dimension": "tokens", "limit": 100, "actual": 200},
        )
    # The re-raised typed error gets surfaced through MeridianError handling
    assert resp.status_code == 422


def test_soft_budget_exceeded_error_reraised_when_already_typed(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from meridiand._app import create_app
    from meridiand._audit import FileAuditLog
    from meridiand._soft_budget_exceeded import SoftBudgetExceededSessionError
    from storage_event_log import LocalEventLogWriter

    audit = FileAuditLog(tmp_path)
    writer = LocalEventLogWriter(tmp_path)
    err_to_raise = SoftBudgetExceededSessionError(
        message="pre-typed", timestamp=pagination_now(), cause=None
    )

    async def _boom(*_a: object, **_k: object) -> None:
        raise err_to_raise

    with patch.object(writer, "append", _boom):
        app = create_app(audit, storage_root=tmp_path, event_log=writer)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/x/sessions/s1/soft-budget-exceeded",
            json={"dimension": "tokens", "limit": 100, "actual": 200},
        )
    assert resp.status_code == 422


def test_user_can_continue_error_reraised_when_already_typed(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from meridiand._app import create_app
    from meridiand._audit import FileAuditLog
    from meridiand._user_can_continue import UserCanContinueError
    from storage_event_log import LocalEventLogWriter

    audit = FileAuditLog(tmp_path)
    writer = LocalEventLogWriter(tmp_path)
    err_to_raise = UserCanContinueError(
        message="pre-typed", timestamp=pagination_now(), cause=None
    )

    async def _boom(*_a: object, **_k: object) -> None:
        raise err_to_raise

    with patch.object(writer, "append", _boom):
        app = create_app(audit, storage_root=tmp_path, event_log=writer)
        client = TestClient(app, raise_server_exceptions=False)
        # Hit the user_can_continue route. Look for the route shape:
        resp = client.post(
            "/v1/x/sessions/s1/user-can-continue",
            json={"prompt": "ok?"},
        )
    assert resp.status_code in {422, 500}
