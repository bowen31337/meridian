"""
System audit middleware conformance suite.

Tests cover:
  - Non-HTTP scope (websocket) is passed through without audit.
  - Requests to non-monitored routes are passed through without audit.
  - POST /v1/skills success (2xx) writes info-level audit entry with code capability_skill_create.
  - POST /v1/skills failure (4xx/5xx) writes error-level audit entry with code
    capability_skill_create_failed.
  - POST /v1/skills/install success writes info-level audit entry with code
    capability_skill_install.
  - POST /v1/skills/install failure writes error-level audit entry with code
    capability_skill_install_failed.
  - POST /v1/vaults/{id}/secrets success writes info-level audit entry with code
    vault_secret_store.
  - POST /v1/vaults/{id}/secrets failure writes error-level audit entry with code
    vault_secret_store_failed.
  - GET /v1/vaults/{id}/secrets/{name}/meta success writes info-level entry with code
    vault_secret_meta.
  - GET /v1/vaults/{id}/secrets/{name}/meta failure writes error-level entry with code
    vault_secret_meta_failed.
  - POST /v1/channels/{id}/pair success writes info-level audit entry with code channel_pair.
  - POST /v1/channels/{id}/pair failure writes error-level audit entry with code
    channel_pair_failed.
  - POST /v1/agents/{id}/skills/{id}/approve success writes info-level entry with code
    skill_activation_approve.
  - POST /v1/agents/{id}/skills/{id}/approve failure writes error-level entry with code
    skill_activation_approve_failed.
  - POST /v1/environments success writes info-level audit entry with code environment_create.
  - POST /v1/environments failure writes error-level audit entry with code
    environment_create_failed.
  - PATCH /v1/environments/{id} success writes info-level entry with code environment_update.
  - PATCH /v1/environments/{id} failure writes error-level entry with code
    environment_update_failed.
  - DELETE /v1/environments/{id} success writes info-level entry with code environment_delete.
  - DELETE /v1/environments/{id} failure writes error-level entry with code
    environment_delete_failed.
  - Audit entry detail includes path and method.
  - Uncaught exception from inner app writes error audit entry and re-raises.
  - Audit write failure when no response sent surfaces 500 error to caller.
  - Audit write failure after response sent writes system_audit.write.failed entry.
  - SystemAuditMiddleware is registered in create_app.
  - E2E: POST /v1/skills writes audit entry to audit.ndjson.
  - E2E: POST /v1/environments failure writes error audit entry to audit.ndjson.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._system_audit_middleware import SystemAuditMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


class _FailingAuditLog(AuditLog):
    """Raises on every write call."""

    def write(self, entry: AuditLogEntry) -> None:
        raise RuntimeError("audit write failed")


class _FailOnceThenCapture(AuditLog):
    """Fails on the first write, captures subsequent writes."""

    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []
        self._failed = False

    def write(self, entry: AuditLogEntry) -> None:
        if not self._failed:
            self._failed = True
            raise RuntimeError("first write failed")
        self.entries.append(entry)


async def _invoke(
    middleware: SystemAuditMiddleware,
    *,
    method: str = "GET",
    path: str = "/",
    status: int = 200,
    scope_type: str = "http",
    raise_exc: Exception | None = None,
) -> tuple[int | None, bytes]:
    scope: dict[str, Any] = {
        "type": scope_type,
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8888),
    }
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg: dict[str, Any]) -> None:
        messages.append(msg)

    async def inner_app(s: Any, r: Any, send_fn: Any) -> None:
        if raise_exc is not None:
            raise raise_exc
        await send_fn({"type": "http.response.start", "status": status, "headers": []})
        await send_fn({"type": "http.response.body", "body": b"{}", "more_body": False})

    mw = SystemAuditMiddleware(inner_app, audit_log=middleware._audit_log)

    with contextlib.suppress(Exception):
        await mw(scope, receive, send)

    captured_status = next(
        (m["status"] for m in messages if m.get("type") == "http.response.start"), None
    )
    body = next(
        (m.get("body", b"") for m in messages if m.get("type") == "http.response.body"), b""
    )
    return captured_status, body


def _make_middleware(audit_log: AuditLog | None = None) -> SystemAuditMiddleware:
    async def _handler(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    return SystemAuditMiddleware(_handler, audit_log=audit_log or NoopAuditLog())


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _read_audit_log(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# TestNonHttpPassthrough
# ---------------------------------------------------------------------------


class TestNonHttpPassthrough:
    async def test_websocket_scope_passes_through(self) -> None:
        audit = _CapturingAuditLog()
        forwarded: list[Any] = []

        async def capture(scope: Any, receive: Any, send: Any) -> None:
            forwarded.append(scope)

        mw = SystemAuditMiddleware(capture, audit_log=audit)
        scope = {"type": "websocket", "headers": [], "client": None}

        async def receive() -> dict[str, Any]:
            return {}

        async def send(msg: Any) -> None:
            pass

        await mw(scope, receive, send)
        assert forwarded == [scope]
        assert not audit.entries

    async def test_lifespan_scope_passes_through(self) -> None:
        audit = _CapturingAuditLog()
        forwarded: list[Any] = []

        async def capture(scope: Any, receive: Any, send: Any) -> None:
            forwarded.append(scope)

        mw = SystemAuditMiddleware(capture, audit_log=audit)
        scope = {"type": "lifespan"}

        async def receive() -> dict[str, Any]:
            return {}

        async def send(msg: Any) -> None:
            pass

        await mw(scope, receive, send)
        assert len(forwarded) == 1
        assert not audit.entries


# ---------------------------------------------------------------------------
# TestNonMonitoredRoutes
# ---------------------------------------------------------------------------


class TestNonMonitoredRoutes:
    async def test_get_skills_not_monitored(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="GET", path="/v1/skills")
        assert not audit.entries

    async def test_get_vaults_not_monitored(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="GET", path="/v1/vaults")
        assert not audit.entries

    async def test_get_environments_not_monitored(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="GET", path="/v1/environments")
        assert not audit.entries

    async def test_post_agents_not_monitored(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/agents")
        assert not audit.entries

    async def test_post_vaults_not_monitored(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/vaults")
        assert not audit.entries

    async def test_unknown_path_not_monitored(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/unknown/route")
        assert not audit.entries


# ---------------------------------------------------------------------------
# TestCapabilityDecisions
# ---------------------------------------------------------------------------


class TestCapabilityDecisions:
    async def test_post_skills_success_writes_info_entry(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills", status=201)
        assert any(e.level == "info" for e in audit.entries)

    async def test_post_skills_success_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills", status=201)
        assert any(e.code == "capability_skill_create" for e in audit.entries)

    async def test_post_skills_success_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills", status=201)
        assert any(e.event == "capability.decision.skill.created" for e in audit.entries)

    async def test_post_skills_failure_writes_error_entry(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills", status=422)
        assert any(e.level == "error" for e in audit.entries)

    async def test_post_skills_failure_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills", status=422)
        assert any(e.code == "capability_skill_create_failed" for e in audit.entries)

    async def test_post_skills_failure_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills", status=500)
        assert any(e.event == "capability.decision.skill.create.failed" for e in audit.entries)

    async def test_post_skills_install_success_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills/install", status=201)
        assert any(e.code == "capability_skill_install" for e in audit.entries)

    async def test_post_skills_install_failure_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills/install", status=422)
        assert any(e.code == "capability_skill_install_failed" for e in audit.entries)

    async def test_post_skills_install_failure_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills/install", status=422)
        assert any(e.event == "capability.decision.skill.install.failed" for e in audit.entries)

    async def test_post_skills_detail_has_path_and_method(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills", status=201)
        entry = next(e for e in audit.entries if e.code == "capability_skill_create")
        assert entry.detail is not None
        assert entry.detail["path"] == "/v1/skills"
        assert entry.detail["method"] == "POST"


# ---------------------------------------------------------------------------
# TestVaultAccesses
# ---------------------------------------------------------------------------


class TestVaultAccesses:
    async def test_post_vault_secrets_success_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/vaults/vault_abc/secrets", status=201)
        assert any(e.code == "vault_secret_store" for e in audit.entries)

    async def test_post_vault_secrets_success_level(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/vaults/vault_abc/secrets", status=201)
        assert any(e.level == "info" for e in audit.entries)

    async def test_post_vault_secrets_success_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/vaults/vault_abc/secrets", status=201)
        assert any(e.event == "vault.access.secret.stored" for e in audit.entries)

    async def test_post_vault_secrets_failure_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/vaults/vault_abc/secrets", status=409)
        assert any(e.code == "vault_secret_store_failed" for e in audit.entries)

    async def test_post_vault_secrets_failure_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/vaults/vault_abc/secrets", status=404)
        assert any(e.event == "vault.access.secret.store.failed" for e in audit.entries)

    async def test_get_vault_secret_meta_success_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="GET", path="/v1/vaults/vault_abc/secrets/my_key/meta", status=200)
        assert any(e.code == "vault_secret_meta" for e in audit.entries)

    async def test_get_vault_secret_meta_success_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="GET", path="/v1/vaults/vault_abc/secrets/my_key/meta", status=200)
        assert any(e.event == "vault.access.secret.meta.read" for e in audit.entries)

    async def test_get_vault_secret_meta_failure_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="GET", path="/v1/vaults/vault_abc/secrets/my_key/meta", status=404)
        assert any(e.code == "vault_secret_meta_failed" for e in audit.entries)

    async def test_get_vault_secret_meta_failure_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="GET", path="/v1/vaults/vault_abc/secrets/my_key/meta", status=404)
        assert any(e.event == "vault.access.secret.meta.failed" for e in audit.entries)

    async def test_vault_id_with_underscores_matched(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/vaults/vault_abc123def/secrets", status=201)
        assert any(e.code == "vault_secret_store" for e in audit.entries)


# ---------------------------------------------------------------------------
# TestChannelPairings
# ---------------------------------------------------------------------------


class TestChannelPairings:
    async def test_post_channel_pair_success_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/channels/ch_abc/pair", status=201)
        assert any(e.code == "channel_pair" for e in audit.entries)

    async def test_post_channel_pair_success_level(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/channels/ch_abc/pair", status=201)
        assert any(e.level == "info" for e in audit.entries)

    async def test_post_channel_pair_success_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/channels/ch_abc/pair", status=201)
        assert any(e.event == "channel.pairing.issued" for e in audit.entries)

    async def test_post_channel_pair_failure_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/channels/ch_abc/pair", status=404)
        assert any(e.code == "channel_pair_failed" for e in audit.entries)

    async def test_post_channel_pair_failure_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/channels/ch_abc/pair", status=404)
        assert any(e.event == "channel.pairing.failed" for e in audit.entries)

    async def test_post_channel_pair_detail_has_path(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/channels/ch_xyz/pair", status=201)
        entry = next(e for e in audit.entries if e.code == "channel_pair")
        assert entry.detail is not None
        assert entry.detail["path"] == "/v1/channels/ch_xyz/pair"


# ---------------------------------------------------------------------------
# TestSkillPromotions
# ---------------------------------------------------------------------------


class TestSkillPromotions:
    async def test_post_approve_success_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(
            mw,
            method="POST",
            path="/v1/agents/agent_abc/skills/skill_xyz/approve",
            status=200,
        )
        assert any(e.code == "skill_activation_approve" for e in audit.entries)

    async def test_post_approve_success_level(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(
            mw,
            method="POST",
            path="/v1/agents/agent_abc/skills/skill_xyz/approve",
            status=200,
        )
        assert any(e.level == "info" for e in audit.entries)

    async def test_post_approve_success_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(
            mw,
            method="POST",
            path="/v1/agents/agent_abc/skills/skill_xyz/approve",
            status=200,
        )
        assert any(e.event == "skill.promotion.approved" for e in audit.entries)

    async def test_post_approve_failure_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(
            mw,
            method="POST",
            path="/v1/agents/agent_abc/skills/skill_xyz/approve",
            status=409,
        )
        assert any(e.code == "skill_activation_approve_failed" for e in audit.entries)

    async def test_post_approve_failure_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(
            mw,
            method="POST",
            path="/v1/agents/agent_abc/skills/skill_xyz/approve",
            status=404,
        )
        assert any(e.event == "skill.promotion.approve.failed" for e in audit.entries)

    async def test_post_approve_detail_has_method(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(
            mw,
            method="POST",
            path="/v1/agents/agent_abc/skills/skill_xyz/approve",
            status=200,
        )
        entry = next(e for e in audit.entries if e.code == "skill_activation_approve")
        assert entry.detail is not None
        assert entry.detail["method"] == "POST"


# ---------------------------------------------------------------------------
# TestEnvironmentChanges
# ---------------------------------------------------------------------------


class TestEnvironmentChanges:
    async def test_post_environments_success_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/environments", status=201)
        assert any(e.code == "environment_create" for e in audit.entries)

    async def test_post_environments_success_level(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/environments", status=201)
        assert any(e.level == "info" for e in audit.entries)

    async def test_post_environments_success_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/environments", status=201)
        assert any(e.event == "environment.change.created" for e in audit.entries)

    async def test_post_environments_failure_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/environments", status=422)
        assert any(e.code == "environment_create_failed" for e in audit.entries)

    async def test_post_environments_failure_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/environments", status=500)
        assert any(e.event == "environment.change.create.failed" for e in audit.entries)

    async def test_patch_environment_success_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="PATCH", path="/v1/environments/env_abc", status=200)
        assert any(e.code == "environment_update" for e in audit.entries)

    async def test_patch_environment_success_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="PATCH", path="/v1/environments/env_abc", status=200)
        assert any(e.event == "environment.change.updated" for e in audit.entries)

    async def test_patch_environment_failure_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="PATCH", path="/v1/environments/env_abc", status=409)
        assert any(e.code == "environment_update_failed" for e in audit.entries)

    async def test_patch_environment_failure_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="PATCH", path="/v1/environments/env_abc", status=404)
        assert any(e.event == "environment.change.update.failed" for e in audit.entries)

    async def test_delete_environment_success_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="DELETE", path="/v1/environments/env_abc", status=204)
        assert any(e.code == "environment_delete" for e in audit.entries)

    async def test_delete_environment_success_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="DELETE", path="/v1/environments/env_abc", status=204)
        assert any(e.event == "environment.change.deleted" for e in audit.entries)

    async def test_delete_environment_failure_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="DELETE", path="/v1/environments/env_abc", status=404)
        assert any(e.code == "environment_delete_failed" for e in audit.entries)

    async def test_delete_environment_failure_event(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(mw, method="DELETE", path="/v1/environments/env_abc", status=409)
        assert any(e.event == "environment.change.delete.failed" for e in audit.entries)

    async def test_patch_environments_list_not_monitored(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        # PATCH /v1/environments (no ID) is not monitored — only PATCH /v1/environments/{id}
        await _invoke(mw, method="PATCH", path="/v1/environments", status=200)
        assert not audit.entries


# ---------------------------------------------------------------------------
# TestExceptionHandling
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    async def test_exception_writes_error_audit_entry(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(
            mw,
            method="POST",
            path="/v1/skills",
            raise_exc=RuntimeError("something broke"),
        )
        assert any(e.level == "error" for e in audit.entries)

    async def test_exception_writes_failure_code(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(
            mw,
            method="POST",
            path="/v1/skills",
            raise_exc=RuntimeError("something broke"),
        )
        assert any(e.code == "capability_skill_create_failed" for e in audit.entries)

    async def test_exception_audit_detail_has_error_message(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(
            mw,
            method="POST",
            path="/v1/skills",
            raise_exc=RuntimeError("something broke"),
        )
        entry = next(e for e in audit.entries if e.code == "capability_skill_create_failed")
        assert entry.detail is not None
        assert "something broke" in entry.detail.get("error", "")

    async def test_exception_from_non_monitored_route_no_audit(self) -> None:
        audit = _CapturingAuditLog()
        mw = _make_middleware(audit)
        await _invoke(
            mw,
            method="GET",
            path="/v1/sessions",
            raise_exc=RuntimeError("boom"),
        )
        assert not audit.entries


# ---------------------------------------------------------------------------
# TestAuditWriteFailure
# ---------------------------------------------------------------------------


class TestAuditWriteFailure:
    async def test_audit_write_failure_after_response_writes_failed_entry(self) -> None:
        audit = _FailOnceThenCapture()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills", status=201)
        assert any(e.code == "system_audit_write_failed" for e in audit.entries)

    async def test_audit_write_failure_after_response_failed_entry_has_original_event(
        self,
    ) -> None:
        audit = _FailOnceThenCapture()
        mw = _make_middleware(audit)
        await _invoke(mw, method="POST", path="/v1/skills", status=201)
        entry = next(e for e in audit.entries if e.code == "system_audit_write_failed")
        assert entry.detail is not None
        assert entry.detail["original_event"] == "capability.decision.skill.created"

    async def test_audit_write_failure_before_response_surfaces_500(self) -> None:
        audit = _FailingAuditLog()

        async def failing_inner(scope: Any, receive: Any, send: Any) -> None:
            raise RuntimeError("inner failure")

        mw = SystemAuditMiddleware(failing_inner, audit_log=audit)
        scope: dict[str, Any] = {
            "type": "http",
            "method": "POST",
            "path": "/v1/skills",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8888),
        }
        messages: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(msg: dict[str, Any]) -> None:
            messages.append(msg)

        await mw(scope, receive, send)
        status = next(
            (m["status"] for m in messages if m.get("type") == "http.response.start"), None
        )
        assert status == 500

    async def test_audit_write_failure_before_response_error_code(self) -> None:
        audit = _FailingAuditLog()

        async def failing_inner(scope: Any, receive: Any, send: Any) -> None:
            raise RuntimeError("inner failure")

        mw = SystemAuditMiddleware(failing_inner, audit_log=audit)
        scope: dict[str, Any] = {
            "type": "http",
            "method": "POST",
            "path": "/v1/skills",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8888),
        }
        messages: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(msg: dict[str, Any]) -> None:
            messages.append(msg)

        await mw(scope, receive, send)
        body = next(
            (m.get("body", b"") for m in messages if m.get("type") == "http.response.body"), b""
        )
        data = json.loads(body)
        assert data["error"]["code"] == "system_audit_write_failed"


# ---------------------------------------------------------------------------
# TestMiddlewareRegistration
# ---------------------------------------------------------------------------


class TestMiddlewareRegistration:
    def test_system_audit_middleware_registered_in_create_app(self) -> None:
        app = create_app(NoopAuditLog())
        assert any(m.cls is SystemAuditMiddleware for m in app.user_middleware)

    def test_system_audit_middleware_is_inside_gzip(self) -> None:
        from fastapi.middleware.gzip import GZipMiddleware

        app = create_app(NoopAuditLog())
        middleware_classes = [m.cls for m in app.user_middleware]
        gzip_idx = middleware_classes.index(GZipMiddleware)
        audit_idx = middleware_classes.index(SystemAuditMiddleware)
        # user_middleware is outermost-first; higher index = closer to routes (more inner).
        # SystemAuditMiddleware is added before GZipMiddleware so it sits inside GZip.
        assert audit_idx > gzip_idx


# ---------------------------------------------------------------------------
# E2E: audit entries appear in audit.ndjson via the full app stack
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_post_skills_writes_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/skills",
            json={
                "name": "my-skill",
                "description": "A test skill",
                "instructions": "Do the thing",
                "tools": [{"name": "bash", "description": "Run shell commands"}],
            },
        )
        entries = _read_audit_log(storage_root)
        assert any(e.get("code") == "capability_skill_create" for e in entries)

    def test_post_skills_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/skills",
            json={
                "name": "my-skill",
                "description": "A test skill",
                "instructions": "Do the thing",
                "tools": [{"name": "bash", "description": "Run shell commands"}],
            },
        )
        entries = _read_audit_log(storage_root)
        entry = next(e for e in entries if e.get("code") == "capability_skill_create")
        assert entry["level"] == "info"

    def test_post_environments_failure_writes_error_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/environments",
            json={"name": "", "backend": "docker"},
        )
        entries = _read_audit_log(storage_root)
        assert any(e.get("code") == "environment_create_failed" for e in entries)

    def test_post_environments_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/environments",
            json={"name": "", "backend": "docker"},
        )
        entries = _read_audit_log(storage_root)
        entry = next((e for e in entries if e.get("code") == "environment_create_failed"), None)
        assert entry is not None
        assert entry["level"] == "error"

    def test_post_channel_pair_not_found_writes_error_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels/nonexistent_ch/pair", json={})
        entries = _read_audit_log(storage_root)
        assert any(e.get("code") == "channel_pair_failed" for e in entries)
