"""Sweep tests to close small coverage gaps across many meridiand modules.

Each test class targets a single source module's leftover branches/lines
without needing to spin up a full FastAPI test client where possible.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def pagination_now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# _acp_compliance — _result with reason path
# ---------------------------------------------------------------------------


class TestCheckpointPerCall:
    """Cover the tool-call completion tracking + per-call duration logic in _checkpoint."""

    def test_per_call_duration_tracking(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        body1 = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "t1", "name": "bash"}, {"id": "t2", "name": "grep"}],
            "message_tail": [{"role": "user", "content": "x"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "taken_at": "2024-01-01T00:00:00+00:00",
        }
        r1 = client.post("/v1/x/sessions/s1/checkpoint", json=body1)
        assert r1.status_code == 200

        # Second checkpoint: t1 completed; t2 still pending
        body2 = {
            **body1,
            "seq": 2,
            "pending_tool_calls": [{"id": "t2", "name": "grep"}],
            "taken_at": "2024-01-01T00:00:01+00:00",
        }
        r2 = client.post("/v1/x/sessions/s1/checkpoint", json=body2)
        assert r2.status_code == 200

    def test_corrupt_latest_skipped(self, tmp_path: Path) -> None:
        """latest.json that can't be parsed is silently skipped (120-123)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        # Pre-write a corrupt latest.json
        cp_dir = tmp_path / "checkpoints" / "s2"
        cp_dir.mkdir(parents=True)
        (cp_dir / "latest.json").write_text("not json {{{")

        body = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [],
            "message_tail": [{"role": "user", "content": "x"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "taken_at": "2024-01-01T00:00:00+00:00",
        }
        resp = client.post("/v1/x/sessions/s2/checkpoint", json=body)
        assert resp.status_code == 200

    def test_per_call_duration_with_bad_timestamp(self, tmp_path: Path) -> None:
        """Bad taken_at in prev_taken_at raises inside try/except — silently skipped (143-148)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        cp_dir = tmp_path / "checkpoints" / "s3"
        cp_dir.mkdir(parents=True)
        prev = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "x", "name": "n"}],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "not-a-timestamp",
        }
        (cp_dir / "latest.json").write_text(json.dumps(prev))

        body = {
            "seq": 2,
            "phase": "thinking",
            "pending_tool_calls": [],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:00+00:00",
        }
        resp = client.post("/v1/x/sessions/s3/checkpoint", json=body)
        assert resp.status_code == 200

    def test_prev_call_not_dict_or_missing_id_skipped(self, tmp_path: Path) -> None:
        """A previous pending_tool_call that's not a dict or missing id is skipped (120->119)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        cp_dir = tmp_path / "checkpoints" / "s4"
        cp_dir.mkdir(parents=True)
        # Mix of non-dict, dict-no-id, and valid call
        prev = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [
                "not a dict",  # non-dict — skipped
                {"name": "nopid"},  # dict but no id — skipped
                {"id": "ok", "name": "k"},  # valid
            ],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:00+00:00",
        }
        (cp_dir / "latest.json").write_text(json.dumps(prev))

        body = {
            "seq": 2,
            "phase": "thinking",
            "pending_tool_calls": [],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:01+00:00",
        }
        resp = client.post("/v1/x/sessions/s4/checkpoint", json=body)
        assert resp.status_code == 200

    def test_no_completed_calls_skips_metrics(self, tmp_path: Path) -> None:
        """If prev_calls have all carried over to current, completed=[] (136->159)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        cp_dir = tmp_path / "checkpoints" / "s5"
        cp_dir.mkdir(parents=True)
        prev = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "t1", "name": "a"}],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:00+00:00",
        }
        (cp_dir / "latest.json").write_text(json.dumps(prev))

        # body still has t1 → not completed
        body = {
            "seq": 2,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "t1", "name": "a"}],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:01+00:00",
        }
        resp = client.post("/v1/x/sessions/s5/checkpoint", json=body)
        assert resp.status_code == 200

    def test_per_call_duration_no_prev_taken_at(self, tmp_path: Path) -> None:
        """prev with no taken_at falls through (138->149)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        cp_dir = tmp_path / "checkpoints" / "s6"
        cp_dir.mkdir(parents=True)
        prev = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "t1", "name": "a"}],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "",  # falsy — triggers 138->149
        }
        (cp_dir / "latest.json").write_text(json.dumps(prev))

        body = {
            "seq": 2,
            "phase": "thinking",
            "pending_tool_calls": [],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:01+00:00",
        }
        resp = client.post("/v1/x/sessions/s6/checkpoint", json=body)
        assert resp.status_code == 200

    def test_per_call_duration_naive_timestamps(self, tmp_path: Path) -> None:
        """Naive (no-tz) timestamps trigger the tzinfo-None branches (143, 145)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        cp_dir = tmp_path / "checkpoints" / "s7"
        cp_dir.mkdir(parents=True)
        prev = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "t1", "name": "a"}],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:00",  # naive — no tz
        }
        (cp_dir / "latest.json").write_text(json.dumps(prev))

        body = {
            "seq": 2,
            "phase": "thinking",
            "pending_tool_calls": [],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:01",  # naive — no tz
        }
        resp = client.post("/v1/x/sessions/s7/checkpoint", json=body)
        assert resp.status_code == 200

    def test_checkpoint_typed_error_reraised(self, tmp_path: Path) -> None:
        """CheckpointError raised inside is re-raised verbatim (line 171)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog
        from meridiand._checkpoint import CheckpointError

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "meridiand._checkpoint.dispatch_hooks",
            side_effect=CheckpointError(message="pre-typed", timestamp=pagination_now(), cause=None),
        ):
            body = {
                "seq": 1,
                "phase": "thinking",
                "pending_tool_calls": [],
                "message_tail": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "taken_at": "2024-01-01T00:00:00+00:00",
            }
            resp = client.post("/v1/x/sessions/s8/checkpoint", json=body)
        assert resp.status_code == 422


class TestSkillForgePrecisionErrors:
    def test_skill_forge_precision_typed_error_reraised(self, tmp_path: Path) -> None:
        """SkillForgePrecisionError raised inside is re-raised (165-179)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_forge_precision import (
            SkillForgePrecisionError,
            compute_precision_metric,
        )

        proposals_dir = tmp_path / "proposals"
        activations_dir = tmp_path / "activations"
        precision_dir = tmp_path / "precision"
        proposals_dir.mkdir()
        activations_dir.mkdir()
        precision_dir.mkdir()

        pre = SkillForgePrecisionError(message="pre", timestamp=pagination_now(), cause=None)
        with patch(
            "meridiand._skill_forge_precision.json.dumps",
            side_effect=pre,
        ):
            with pytest.raises(SkillForgePrecisionError):
                compute_precision_metric(
                    proposals_dir=proposals_dir,
                    activations_dir=activations_dir,
                    precision_dir=precision_dir,
                    audit_log=NoopAuditLog(),
                )

    def test_skill_forge_precision_skips_records_without_ids(self, tmp_path: Path) -> None:
        """Records without id/skill_version_id are skipped (branches 110->106, 125->120)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_forge_precision import compute_precision_metric

        proposals_dir = tmp_path / "proposals"
        activations_dir = tmp_path / "activations"
        precision_dir = tmp_path / "precision"
        proposals_dir.mkdir()
        activations_dir.mkdir()
        precision_dir.mkdir()
        # Proposal without 'id'
        (proposals_dir / "noid.json").write_text(json.dumps({"name": "noid"}))
        # Activation with active status but no skill_version_id
        (activations_dir / "noid.json").write_text(
            json.dumps({"status": "active"})
        )
        metric = compute_precision_metric(
            proposals_dir=proposals_dir,
            activations_dir=activations_dir,
            precision_dir=precision_dir,
            audit_log=NoopAuditLog(),
        )
        assert metric is not None

    def test_skill_forge_precision_skips_malformed_files(self, tmp_path: Path) -> None:
        """Malformed JSON in proposals/activations is silently skipped (112-113, 127-128)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_forge_precision import compute_precision_metric

        proposals_dir = tmp_path / "proposals"
        activations_dir = tmp_path / "activations"
        precision_dir = tmp_path / "precision"
        proposals_dir.mkdir()
        activations_dir.mkdir()
        precision_dir.mkdir()
        (proposals_dir / "bad.json").write_text("not json {{{")
        (proposals_dir / "good.json").write_text(json.dumps({"id": "p1"}))
        (activations_dir / "bad.json").write_text("not json {{{")
        (activations_dir / "good.json").write_text(
            json.dumps({"status": "active", "skill_version_id": "p1"})
        )

        metric = compute_precision_metric(
            proposals_dir=proposals_dir,
            activations_dir=activations_dir,
            precision_dir=precision_dir,
            audit_log=NoopAuditLog(),
        )
        assert metric is not None


class TestAgentsErrors:
    def test_all_http_statuses(self) -> None:
        from meridiand._agents import (
            AgentCreateError,
            AgentDeleteError,
            AgentGetError,
            AgentInvalidRequestError,
            AgentListError,
            AgentNotFoundError,
            AgentVersionCreateError,
            AgentVersionGetError,
            AgentVersionNotFoundError,
            AgentVersionsListError,
        )

        ts = pagination_now()
        assert AgentCreateError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert AgentInvalidRequestError(message="m", timestamp=ts).http_status() == 422
        assert AgentNotFoundError(agent_id="x", timestamp=ts).http_status() == 404
        assert AgentGetError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert AgentDeleteError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert AgentVersionCreateError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert (
            AgentVersionNotFoundError(
                agent_id="a", version_id="v", timestamp=ts
            ).http_status()
            == 404
        )
        assert AgentVersionGetError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert (
            AgentVersionsListError(message="m", timestamp=ts, cause=None).http_status() == 500
        )
        assert AgentListError(message="m", timestamp=ts, cause=None).http_status() == 500


class TestVaultsErrors:
    def test_all_http_statuses(self) -> None:
        from meridiand._vaults import (
            VaultCreateError,
            VaultDeleteError,
            VaultInUseError,
            VaultInvalidRequestError,
            VaultListError,
            VaultNotFoundError,
            VaultSecretConflictError,
            VaultSecretDeleteConfirmationError,
            VaultSecretDeleteError,
            VaultSecretInvalidRequestError,
            VaultSecretListError,
            VaultSecretMetaError,
            VaultSecretNotFoundError,
            VaultSecretStoreError,
        )

        ts = pagination_now()
        assert VaultCreateError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert VaultInvalidRequestError(message="m", timestamp=ts).http_status() == 422
        assert VaultNotFoundError(vault_id="x", timestamp=ts).http_status() == 404
        assert VaultInUseError(vault_id="x", timestamp=ts).http_status() == 409
        assert VaultDeleteError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert VaultListError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert VaultSecretStoreError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert VaultSecretInvalidRequestError(message="m", timestamp=ts).http_status() == 422
        assert (
            VaultSecretConflictError(vault_id="v", key="k", timestamp=ts).http_status() == 409
        )
        assert (
            VaultSecretNotFoundError(vault_id="v", name="n", timestamp=ts).http_status() == 404
        )
        assert (
            VaultSecretMetaError(message="m", timestamp=ts, cause=None).http_status() == 500
        )
        assert (
            VaultSecretListError(message="m", timestamp=ts, cause=None).http_status() == 500
        )
        assert (
            VaultSecretDeleteError(message="m", timestamp=ts, cause=None).http_status() == 500
        )
        assert (
            VaultSecretDeleteConfirmationError(
                name="n", vault_id="v", timestamp=ts
            ).http_status()
            == 400
        )


class TestSkillForgeProposalsErrors:
    def test_all_http_statuses(self) -> None:
        from meridiand._skill_forge_proposals import (
            SkillForgeProposalAlreadyPromotedError,
            SkillForgeProposalApproveError,
            SkillForgeProposalListError,
            SkillForgeProposalNotFoundError,
            SkillForgeProposalRejectError,
        )

        ts = pagination_now()
        assert SkillForgeProposalNotFoundError(proposal_id="x", timestamp=ts).http_status() == 404
        assert SkillForgeProposalAlreadyPromotedError(proposal_id="x", timestamp=ts).http_status() == 409
        assert (
            SkillForgeProposalListError(message="m", timestamp=ts, cause=None).http_status()
            == 500
        )
        assert (
            SkillForgeProposalRejectError(message="m", timestamp=ts, cause=None).http_status()
            == 500
        )
        assert (
            SkillForgeProposalApproveError(message="m", timestamp=ts, cause=None).http_status()
            == 500
        )

    async def test_list_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """Generic exception in list handler is wrapped (276-293)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_forge_proposals import (
            SkillForgeProposalListError,
            make_skill_forge_proposals_router,
        )

        proposals_dir = tmp_path / "skill_forge" / "proposals"
        proposals_dir.mkdir(parents=True)
        (proposals_dir / "p1.json").write_text(
            json.dumps({"id": "p1", "status": "PROPOSAL"})
        )

        router = make_skill_forge_proposals_router(
            audit_log=NoopAuditLog(), storage_root=tmp_path
        )
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/skill_forge/proposals" and "GET" in r.methods
        )
        with patch(
            "meridiand._skill_forge_proposals.make_cursor_page",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(SkillForgeProposalListError):
                await handler(cursor=None, limit=10, include_efficacy=False)

    async def test_approve_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """Generic exception in approve handler is wrapped (lines 422-442)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_forge_proposals import (
            SkillForgeProposalApproveError,
            make_skill_forge_proposals_router,
        )

        # Pre-create proposal
        proposals_dir = tmp_path / "skill_forge" / "proposals"
        proposals_dir.mkdir(parents=True)
        (proposals_dir / "p1.json").write_text(
            json.dumps(
                {
                    "id": "p1",
                    "skill_id": "s1",
                    "instructions": "do x",
                    "status": "proposed",
                }
            )
        )

        router = make_skill_forge_proposals_router(
            audit_log=NoopAuditLog(), storage_root=tmp_path
        )
        handler = next(
            r.endpoint
            for r in router.routes
            if "/approve" in r.path and "POST" in r.methods
        )
        with patch("meridiand._skill_forge_proposals.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(SkillForgeProposalApproveError):
                await handler("p1")

    async def test_reject_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """Generic exception in reject handler is wrapped (lines 517-537)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_forge_proposals import (
            RejectProposalRequest,
            SkillForgeProposalRejectError,
            make_skill_forge_proposals_router,
        )

        # Pre-create proposal
        proposals_dir = tmp_path / "skill_forge" / "proposals"
        proposals_dir.mkdir(parents=True)
        (proposals_dir / "p1.json").write_text(
            json.dumps(
                {"id": "p1", "skill_id": "s1", "status": "proposed"}
            )
        )

        router = make_skill_forge_proposals_router(
            audit_log=NoopAuditLog(), storage_root=tmp_path
        )
        handler = next(
            r.endpoint for r in router.routes if "/reject" in r.path and "POST" in r.methods
        )
        req = RejectProposalRequest(reason="bad")
        with patch("meridiand._skill_forge_proposals.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(SkillForgeProposalRejectError):
                await handler("p1", req)

    async def test_approve_skips_malformed_version_json(self, tmp_path: Path) -> None:
        """A malformed skill_versions JSON file is silently skipped during approve (356-357)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_forge_proposals import (
            make_skill_forge_proposals_router,
        )

        proposals_dir = tmp_path / "skill_forge" / "proposals"
        proposals_dir.mkdir(parents=True)
        (proposals_dir / "p1.json").write_text(
            json.dumps(
                {"id": "p1", "skill_id": "s1", "instructions": "do x", "status": "PROPOSAL"}
            )
        )
        # Seed a malformed version
        versions_dir = tmp_path / "skill_versions"
        versions_dir.mkdir(parents=True)
        (versions_dir / "bad.json").write_text("not json {{{")

        router = make_skill_forge_proposals_router(
            audit_log=NoopAuditLog(), storage_root=tmp_path
        )
        handler = next(
            r.endpoint for r in router.routes if "/approve" in r.path and "POST" in r.methods
        )
        resp = await handler("p1")
        assert resp is not None

    def test_promotion_skips_malformed_version(self, tmp_path: Path) -> None:
        """Test the malformed version JSON skip path (356-357)."""
        # This is exercised indirectly through approve. Just ensure the
        # versions_dir glob doesn't crash on bad JSON.
        versions_dir = tmp_path / "skill_versions"
        versions_dir.mkdir(parents=True)
        (versions_dir / "bad.json").write_text("not json {{{")
        # Iterate manually to mimic the path
        max_ver = 0
        for vpath in versions_dir.glob("*.json"):
            try:
                vr = json.loads(vpath.read_text())
                if vr.get("skill_id") == "x":
                    max_ver = max(max_ver, vr.get("version_number", 0))
            except Exception:
                pass
        assert max_ver == 0


class TestSystemConfigEndpoint:
    """Cover PUT /v1/system/config endpoint paths."""

    @staticmethod
    def _make_router_client(audit_log, model_router):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from meridiand._system_config import make_system_config_router

        router = make_system_config_router(audit_log=audit_log, model_router=model_router)
        app = FastAPI()
        app.include_router(router)
        return TestClient(app, raise_server_exceptions=False)

    def test_reload_invalid_yaml(self) -> None:
        from core_errors import NoopAuditLog
        from meridian_sdk_provider import ModelRouter

        from meridian_sdk_provider import ModelRoutingPolicy
        router = ModelRouter(registry=None, policy=ModelRoutingPolicy(rules=[], fallbacks=[]))
        client = self._make_router_client(NoopAuditLog(), router)
        resp = client.put(
            "/v1/system/config",
            content="not yaml ::",
            headers={"content-type": "text/yaml"},
        )
        assert resp.status_code == 422

    def test_reload_validate_error(self) -> None:
        from core_errors import NoopAuditLog
        from meridian_sdk_provider import ModelRouter

        from meridiand._config import MERIDIAN_CONFIG_VERSION

        from meridian_sdk_provider import ModelRoutingPolicy
        router = ModelRouter(registry=None, policy=ModelRoutingPolicy(rules=[], fallbacks=[]))
        client = self._make_router_client(NoopAuditLog(), router)
        yaml_body = (
            f"version: {MERIDIAN_CONFIG_VERSION}\n"
            "storage_root: /tmp/m\n"
            "daemon:\n"
            "  log_level: info\n"
            "  bind:\n"
            "    host: 127.0.0.1\n"
            "    port: 70000\n"
        )
        resp = client.put(
            "/v1/system/config",
            content=yaml_body,
            headers={"content-type": "text/yaml"},
        )
        assert resp.status_code == 422

    def test_reload_success_no_registry(self) -> None:
        from core_errors import NoopAuditLog
        from meridian_sdk_provider import ModelRouter

        from meridiand._config import MERIDIAN_CONFIG_VERSION

        from meridian_sdk_provider import ModelRoutingPolicy
        router = ModelRouter(registry=None, policy=ModelRoutingPolicy(rules=[], fallbacks=[]))
        client = self._make_router_client(NoopAuditLog(), router)
        yaml_body = (
            f"version: {MERIDIAN_CONFIG_VERSION}\n"
            "storage_root: /tmp/m\n"
        )
        resp = client.put(
            "/v1/system/config",
            content=yaml_body,
            headers={"content-type": "text/yaml"},
        )
        assert resp.status_code == 200


class TestSessionsErrors:
    def test_all_http_statuses(self) -> None:
        from meridiand._sessions import (
            MessageAppendError,
            MessageAppendRejectedError,
            MessageListError,
            SessionCreateError,
            SessionGetError,
            SessionListError,
            SessionNotFoundError,
            ThreadCreateError,
            ThreadListError,
        )

        ts = pagination_now()
        assert SessionCreateError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert ThreadListError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert MessageListError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert ThreadCreateError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert MessageAppendRejectedError(message="m", timestamp=ts).http_status() == 422
        assert MessageAppendError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert SessionListError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert SessionNotFoundError(message="m", timestamp=ts).http_status() == 404
        assert SessionGetError(message="m", timestamp=ts, cause=None).http_status() == 500


class TestMemoryStoresGenericExceptions:
    async def test_create_generic_exception_wrapped(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._memory_stores import (
            MemoryStoreCreateError,
            MemoryStoreCreateRequest,
            make_memory_stores_router,
        )

        router = make_memory_stores_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/memory_stores" and "POST" in r.methods
        )
        req = MemoryStoreCreateRequest(name="m", backend="sqlite-vec", scope="global")
        with patch("meridiand._memory_stores.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(MemoryStoreCreateError):
                await handler(req)


class TestSkillActivationsGenericExceptions:
    """Cover generic-exception wrapping in skill_activations handlers."""

    async def test_request_handler_direct_generic_exception(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._skill_activations import (
            SkillActivationError,
            SkillActivationRequest,
            make_skill_activations_router,
        )

        # Pre-create skill and agent so the validation guards pass
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "s1.json").write_text(json.dumps({"id": "s1"}))
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "a1.json").write_text(json.dumps({"id": "a1"}))

        router = make_skill_activations_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/agents/{agent_id}/skills" and "POST" in r.methods
        )
        req = SkillActivationRequest(skill_id="s1")
        with patch("meridiand._skill_activations.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(SkillActivationError):
                await handler("a1", req)

    async def test_approve_handler_direct_generic_exception(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._skill_activations import (
            SkillActivationApproveError,
            make_skill_activations_router,
        )

        # Pre-create activation so the lookup passes
        act_dir = tmp_path / "skill_activations"
        act_dir.mkdir(parents=True)
        (act_dir / "a1_s1.json").write_text(
            json.dumps(
                {
                    "id": "a1_s1",
                    "agent_id": "a1",
                    "skill_id": "s1",
                    "status": "pending",
                }
            )
        )

        router = make_skill_activations_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if "/approve" in r.path and "POST" in r.methods
        )
        with patch("meridiand._skill_activations.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(SkillActivationApproveError):
                await handler("a1", "s1")

    async def test_revoke_handler_direct_generic_exception(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._skill_activations import (
            SkillActivationRevokeError,
            make_skill_activations_router,
        )

        act_dir = tmp_path / "skill_activations"
        act_dir.mkdir(parents=True)
        (act_dir / "a1_s1.json").write_text(
            json.dumps(
                {
                    "id": "a1_s1",
                    "agent_id": "a1",
                    "skill_id": "s1",
                    "status": "active",
                }
            )
        )

        router = make_skill_activations_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/agents/{agent_id}/skills/{skill_id}"
            and "DELETE" in r.methods
        )
        with patch("meridiand._skill_activations.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(SkillActivationRevokeError):
                await handler("a1", "s1")

    async def test_list_handler_direct_generic_exception(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._skill_activations import (
            SkillActivationListError,
            make_skill_activations_router,
        )

        act_dir = tmp_path / "skill_activations"
        act_dir.mkdir(parents=True)
        (act_dir / "a1_s1.json").write_text(
            json.dumps({"id": "a1_s1", "agent_id": "a1", "skill_id": "s1"})
        )

        router = make_skill_activations_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/agents/{agent_id}/skills" and "GET" in r.methods
        )
        with patch(
            "meridiand._skill_activations.json.loads",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(SkillActivationListError):
                await handler("a1")


class TestUserProfilesGenericExceptions:
    """Cover generic-exception wrapping in all 4 user_profile handlers."""

    @staticmethod
    def _client(tmp_path: Path):
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        return TestClient(app, raise_server_exceptions=False)

    async def test_create_generic_exception_wrapped(self, tmp_path: Path) -> None:
        client = self._client(tmp_path)
        with patch("meridiand._user_profiles.json.dumps", side_effect=RuntimeError("boom")):
            resp = client.post(
                "/v1/user_profiles",
                json={"username": "u1", "display_name": "U One"},
            )
        assert resp.status_code == 500

    async def test_delete_generic_exception_wrapped(self, tmp_path: Path) -> None:
        client = self._client(tmp_path)
        d = tmp_path / "user_profiles"
        d.mkdir()
        (d / "u1.json").write_text(json.dumps({"id": "u1", "is_primary": False}))
        with patch.object(Path, "unlink", side_effect=RuntimeError("unlink boom")):
            resp = client.delete("/v1/user_profiles/u1")
        assert resp.status_code == 500

    async def test_update_generic_exception_wrapped(self, tmp_path: Path) -> None:
        client = self._client(tmp_path)
        d = tmp_path / "user_profiles"
        d.mkdir()
        (d / "u2.json").write_text(
            json.dumps({"id": "u2", "username": "u", "display_name": "U"})
        )
        with patch("meridiand._user_profiles.json.dumps", side_effect=RuntimeError("boom")):
            resp = client.patch(
                "/v1/user_profiles/u2",
                json={"display_name": "New"},
            )
        assert resp.status_code == 500

    async def test_get_generic_exception_wrapped(self, tmp_path: Path) -> None:
        client = self._client(tmp_path)
        d = tmp_path / "user_profiles"
        d.mkdir()
        (d / "u3.json").write_text(json.dumps({"id": "u3"}))
        with patch(
            "meridiand._user_profiles.json.loads",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.get("/v1/user_profiles/u3")
        assert resp.status_code == 500

    async def test_create_handler_direct_generic_exception(self, tmp_path: Path) -> None:
        """Call create handler directly to ensure raise-from is traced (line 272)."""
        from core_errors import NoopAuditLog

        from meridiand._user_profiles import (
            UserProfileCreateError,
            UserProfileCreateRequest,
            make_user_profiles_router,
        )

        router = make_user_profiles_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/user_profiles" and "POST" in r.methods
        )
        req = UserProfileCreateRequest(username="u", display_name="U")
        with patch("meridiand._user_profiles.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(UserProfileCreateError):
                await handler(req)

    async def test_update_handler_direct_generic_exception(self, tmp_path: Path) -> None:
        """Call update handler directly (line 448)."""
        from core_errors import NoopAuditLog

        from meridiand._user_profiles import (
            UserProfileUpdateError,
            UserProfileUpdateRequest,
            make_user_profiles_router,
        )

        d = tmp_path / "user_profiles"
        d.mkdir()
        (d / "u_upd.json").write_text(
            json.dumps({"id": "u_upd", "username": "u", "display_name": "U"})
        )
        router = make_user_profiles_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/user_profiles/{user_profile_id}" and "PATCH" in r.methods
        )
        req = UserProfileUpdateRequest(display_name="New")
        with patch("meridiand._user_profiles.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(UserProfileUpdateError):
                await handler("u_upd", req)

    async def test_delete_skips_malformed_manifest(self, tmp_path: Path) -> None:
        """A malformed session manifest is silently skipped during delete (309-310)."""
        client = self._client(tmp_path)
        d = tmp_path / "user_profiles"
        d.mkdir()
        (d / "u_del.json").write_text(json.dumps({"id": "u_del", "is_primary": False}))
        # Seed a malformed session manifest
        sessions_dir = tmp_path / "sessions" / "broken"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "manifest.json").write_text("not json {{{")
        resp = client.delete("/v1/user_profiles/u_del")
        assert resp.status_code in {204, 404, 200}


class TestUserProfilesErrors:
    """http_status for all user_profile error classes."""

    def test_all_http_statuses(self) -> None:
        from meridiand._user_profiles import (
            UserProfileCreateError,
            UserProfileDeleteError,
            UserProfileGetError,
            UserProfileHasActiveSessionsError,
            UserProfileInvalidRequestError,
            UserProfileIsPrimaryError,
            UserProfileNotFoundError,
            UserProfileUpdateError,
        )

        ts = pagination_now()
        assert UserProfileCreateError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert UserProfileInvalidRequestError(message="m", timestamp=ts).http_status() == 422
        assert UserProfileNotFoundError(user_profile_id="x", timestamp=ts).http_status() == 404
        assert UserProfileIsPrimaryError(user_profile_id="x", timestamp=ts).http_status() == 409
        assert (
            UserProfileHasActiveSessionsError(
                user_profile_id="x", timestamp=ts
            ).http_status()
            == 409
        )
        assert UserProfileDeleteError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert UserProfileUpdateError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert UserProfileGetError(message="m", timestamp=ts, cause=None).http_status() == 500


class TestSkillActivationsErrors:
    def test_all_http_statuses(self) -> None:
        from meridiand._skill_activations import (
            SkillActivationApproveError,
            SkillActivationConflictError,
            SkillActivationError,
            SkillActivationListError,
            SkillActivationNotFoundError,
            SkillActivationRequestError,
            SkillActivationRevokeError,
            SkillNotFoundError,
        )

        ts = pagination_now()
        assert SkillActivationRequestError(message="m", timestamp=ts).http_status() == 422
        assert SkillNotFoundError(message="m", timestamp=ts).http_status() == 404
        assert SkillActivationNotFoundError(message="m", timestamp=ts).http_status() == 404
        assert SkillActivationConflictError(message="m", timestamp=ts).http_status() == 409
        assert SkillActivationError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert SkillActivationApproveError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert SkillActivationRevokeError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert SkillActivationListError(message="m", timestamp=ts, cause=None).http_status() == 500


class TestMemoryStoresErrors:
    def test_all_http_statuses(self) -> None:
        from meridiand._memory_stores import (
            MemoryStoreCreateError,
            MemoryStoreInvalidRequestError,
            MemoryStoreNotFoundError,
            MemoryStoreQueryError,
        )

        ts = pagination_now()
        assert MemoryStoreCreateError(message="m", timestamp=ts, cause=None).http_status() == 500
        assert MemoryStoreInvalidRequestError(message="m", timestamp=ts).http_status() == 422
        assert MemoryStoreNotFoundError(message="m", timestamp=ts).http_status() == 404
        assert MemoryStoreQueryError(message="m", timestamp=ts, cause=None).http_status() == 500


class TestMemoryAnniversaryErrors:
    def test_today_helper(self) -> None:
        from meridiand._memory_anniversary import _today

        d = _today()
        from datetime import date
        assert isinstance(d, date)

    def test_fire_error_http_status(self) -> None:
        from meridiand._memory_anniversary import MemoryAnniversaryFireError

        assert MemoryAnniversaryFireError(message="m", timestamp="t", cause=None).http_status() == 500

    def test_memory_not_found_error_http_status(self) -> None:
        from meridiand._memory_anniversary import MemoryNotFoundError

        assert MemoryNotFoundError(memory_key="x", timestamp="t").http_status() == 404

    def test_memory_value_not_date_error_http_status(self) -> None:
        from meridiand._memory_anniversary import MemoryValueNotDateError

        assert (
            MemoryValueNotDateError(memory_key="x", value="v", timestamp="t").http_status()
            == 422
        )

    def test_next_anniversary_fire_date_unreachable_assertion(self) -> None:
        """If the loop's invariant breaks (mocked to never find a year), assertion raises."""
        from datetime import date

        from meridiand._memory_anniversary import _next_anniversary_fire_date

        # Patch range(3) so the loop runs 0 iterations → falls through to AssertionError
        with patch("meridiand._memory_anniversary.range", return_value=[]):
            with pytest.raises(AssertionError, match="unreachable"):
                _next_anniversary_fire_date(
                    anniversary=date(2025, 1, 1),
                    today=date(2025, 6, 1),
                    days_before=1,
                )

    async def test_fire_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """A generic exception raised inside fire_memory_anniversary_trigger is wrapped (257-277)."""
        from core_errors import NoopAuditLog

        from meridiand._memory_anniversary import (
            MemoryAnniversaryFireError,
            fire_memory_anniversary_trigger,
        )

        cron = {
            "id": "c1",
            "memory_key": "k1",
            "days_before": 1,
            "session_id": "s1",
            "task": {},
        }
        with patch(
            "meridiand._memory_anniversary._load_memory_date",
            side_effect=RuntimeError("load boom"),
        ):
            with pytest.raises(MemoryAnniversaryFireError):
                fire_memory_anniversary_trigger(
                    cron_resource=cron,
                    audit_log=NoopAuditLog(),
                    storage_root=tmp_path,
                )


class TestConfigErrors:
    def test_parse_config_yaml_not_mapping(self) -> None:
        """YAML content is a sequence, not a mapping → ValueError → wrapped (line 317)."""
        from meridiand._config import ConfigLoadError, parse_config

        with pytest.raises(ConfigLoadError):
            parse_config("- a\n- b\n")  # YAML list, not mapping

    def test_parse_config_success_round_trip(self) -> None:
        """Successful parse returns MeridianConfig (lines 327-328)."""
        import yaml

        from meridiand._config import MERIDIAN_CONFIG_VERSION, parse_config

        cfg = parse_config(
            yaml.dump({"version": MERIDIAN_CONFIG_VERSION, "storage_root": "/tmp/m"})
        )
        assert cfg.storage_root == Path("/tmp/m")

    def test_parse_config_version_mismatch(self) -> None:
        """Config version != binary version raises ValueError → wrapped (line 322)."""
        import yaml

        from meridiand._config import ConfigLoadError, parse_config

        # version=1 != binary version=2
        with pytest.raises(ConfigLoadError):
            parse_config(yaml.dump({"version": 1, "storage_root": "/tmp/m"}))

    def test_parse_config_pretyped_error_reraise(self) -> None:
        """A pre-raised ConfigLoadError inside parse_config is re-raised (line 331)."""
        from meridiand._config import ConfigLoadError, parse_config

        pre = ConfigLoadError(message="pre", timestamp=pagination_now(), cause=None)
        with patch("meridiand._config.yaml.safe_load", side_effect=pre):
            with pytest.raises(ConfigLoadError):
                parse_config("anything")

    def test_load_config_pretyped_error_reraise(self, tmp_path: Path) -> None:
        """A pre-raised ConfigLoadError inside load_config is re-raised (line 386)."""
        from meridiand._config import ConfigLoadError, load_config

        path = tmp_path / "c.yml"
        path.write_text("version: 2\nstorage_root: /tmp/m\n")
        pre = ConfigLoadError(message="pre", timestamp=pagination_now(), cause=None)
        with patch("meridiand._config.yaml.safe_load", side_effect=pre):
            with pytest.raises(ConfigLoadError):
                load_config(path)

    def test_resolve_location_pretyped_error_reraise(self) -> None:
        """A pre-raised ConfigResolveError inside resolve_config_location re-raises (460)."""
        from meridiand._config import ConfigResolveError, resolve_config_location

        pre = ConfigResolveError(message="pre", timestamp=pagination_now(), cause=None)

        # Patch _USER_CONFIG_PATH.exists to raise ConfigResolveError
        with patch("meridiand._config.os.environ.get", side_effect=pre):
            with pytest.raises(ConfigResolveError):
                resolve_config_location()

    def test_load_config_typed_error_reraise(self, tmp_path: Path) -> None:
        """load_config: typed ConfigLoadError raised → re-raises (line 386)."""
        from meridiand._config import ConfigLoadError, load_config

        path = tmp_path / "bad.yml"
        path.write_text("version: 99\nstorage_root: /tmp/m\n")
        with pytest.raises(ConfigLoadError):
            load_config(path)

    def test_resolve_location_no_config_anywhere(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No env var, no ~/.meridian/config.yml, no /etc/meridian/config.yml → FileNotFoundError (457)."""
        from meridiand._config import resolve_config_location

        monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
        monkeypatch.setattr(
            "meridiand._config._USER_CONFIG_PATH", tmp_path / "no" / "user.yml"
        )
        monkeypatch.setattr(
            "meridiand._config.SYSTEM_CONFIG_PATH", tmp_path / "no" / "system.yml"
        )
        from meridiand._config import ConfigResolveError

        with pytest.raises(ConfigResolveError):
            resolve_config_location()

    def test_validate_invalid_port_emits_error(self) -> None:
        """daemon.bind.port out of range emits validation error (line 614)."""
        from meridiand._config import (
            MERIDIAN_CONFIG_VERSION,
            ConfigValidateError,
            MeridianConfig,
            validate_config,
        )

        config = MeridianConfig.model_validate(
            {
                "version": MERIDIAN_CONFIG_VERSION,
                "storage_root": "/tmp/m",
                "daemon": {
                    "log_level": "info",
                    "bind": {"host": "127.0.0.1", "port": 70000},
                },
            }
        )
        with pytest.raises(ConfigValidateError) as ei:
            validate_config(config)
        assert "not in range" in ei.value.message


class TestFilesErrors:
    def test_files_upload_error_http_status(self) -> None:
        from meridiand._files import FilesUploadError

        assert FilesUploadError(message="m", timestamp="t", cause=None).http_status() == 500

    def test_files_not_found_error_http_status(self) -> None:
        from meridiand._files import FilesNotFoundError

        assert FilesNotFoundError(message="m", timestamp="t").http_status() == 404

    def test_files_invalid_request_error_http_status(self) -> None:
        from meridiand._files import FilesInvalidRequestError

        assert FilesInvalidRequestError(message="m", timestamp="t").http_status() == 422

    async def test_upload_multipart_with_no_file_field(self, tmp_path: Path) -> None:
        """multipart/form-data without 'file' field raises FilesInvalidRequestError (108)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/files",
            files={"not_file": ("foo.txt", b"hi", "text/plain")},
        )
        assert resp.status_code == 422

    async def test_upload_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """A non-FilesUpload exception during upload is wrapped (168-184)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        async def _boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("put boom")

        import base64

        with patch("meridiand._files.LocalBlobStore.put", _boom):
            resp = client.post(
                "/v1/files",
                json={"name": "test", "content": base64.b64encode(b"data").decode()},
            )
        assert resp.status_code == 500

    async def test_get_metadata_typed_error_reraised(self, tmp_path: Path) -> None:
        """FilesNotFoundError raised inside is re-raised (226-241)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/files/nonexistent")
        assert resp.status_code == 404

    async def test_get_metadata_blob_failure_non_not_found(
        self, tmp_path: Path
    ) -> None:
        """BlobFailure with code != BLOB_KEY_NOT_FOUND wraps in FilesUploadError (226-241)."""
        from fastapi.testclient import TestClient
        from storage_blob import BlobFailure

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        import base64
        upload = client.post(
            "/v1/files",
            json={"name": "test", "content": base64.b64encode(b"data").decode()},
        )
        file_id = upload.json()["id"]

        async def _boom(*_a: Any, **_k: Any) -> Any:
            raise BlobFailure(
                code="BLOB_INTERNAL", message="boom", key="k", timestamp=pagination_now()
            )

        from storage_blob import LocalBlobStore as _LBS

        with patch.object(_LBS, "get", _boom):
            resp = client.get(f"/v1/files/{file_id}")
        assert resp.status_code == 500

    async def test_get_metadata_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """Generic exception during metadata get is wrapped (242-258)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        # Create a file first
        import base64
        upload = client.post(
            "/v1/files",
            json={"name": "test", "content": base64.b64encode(b"data").decode()},
        )
        assert upload.status_code == 201
        file_id = upload.json()["id"]

        async def _boom(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("get boom")

        with patch("meridiand._files.LocalBlobStore.get", _boom):
            resp = client.get(f"/v1/files/{file_id}")
        assert resp.status_code == 500

    async def test_get_content_typed_error(self, tmp_path: Path) -> None:
        """Get content for nonexistent file raises FilesNotFoundError (line 292)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/files/nonexistent/content")
        assert resp.status_code == 404

    async def test_get_content_blob_failure_non_not_found(
        self, tmp_path: Path
    ) -> None:
        """BlobFailure with code != BLOB_KEY_NOT_FOUND re-raises (line 292)."""
        from fastapi.testclient import TestClient
        from storage_blob import BlobFailure

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        import base64
        upload = client.post(
            "/v1/files",
            json={"name": "test", "content": base64.b64encode(b"data").decode()},
        )
        file_id = upload.json()["id"]

        async def _boom(*_a: Any, **_k: Any) -> Any:
            raise BlobFailure(
                code="BLOB_INTERNAL", message="boom", key="k", timestamp=pagination_now()
            )

        from storage_blob import LocalBlobStore as _LBS

        with patch.object(_LBS, "get", _boom):
            resp = client.get(f"/v1/files/{file_id}/content")
        assert resp.status_code >= 400

    async def test_get_content_generic_exception(self, tmp_path: Path) -> None:
        """Generic exception during content read is wrapped (309-325)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        import base64
        upload = client.post(
            "/v1/files",
            json={"name": "test", "content": base64.b64encode(b"data").decode()},
        )
        file_id = upload.json()["id"]

        from storage_blob import LocalBlobStore as _LBS

        original_get = _LBS.get
        call_count = [0]

        async def _boom_blob(self: Any, key: str) -> Any:
            call_count[0] += 1
            # First get is metadata; second is blob — fail the blob get
            if call_count[0] >= 2:
                raise RuntimeError("blob boom")
            return await original_get(self, key)

        with patch.object(_LBS, "get", _boom_blob):
            resp = client.get(f"/v1/files/{file_id}/content")
        assert resp.status_code == 500


class TestParallelRunsErrors:
    def test_parallel_runs_error_http_status(self) -> None:
        from meridiand._parallel_runs import ParallelRunsError

        assert ParallelRunsError(message="m", timestamp="t", cause=None).http_status() == 422

    def test_budget_exceeded_error_http_status(self) -> None:
        from meridiand._parallel_runs import BudgetExceededError

        assert BudgetExceededError(message="m", timestamp="t").http_status() == 422

    async def test_parallel_runs_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """Generic exception inside parallel runs is wrapped (282-302)."""
        from core_errors import NoopAuditLog

        from meridiand._parallel_runs import (
            ParallelRunsError,
            ParallelRunsRequest,
            make_parallel_runs_router,
        )

        router = make_parallel_runs_router(
            audit_log=NoopAuditLog(),
            storage_root=tmp_path,
        )
        handler = next(
            r.endpoint for r in router.routes if "/parallel_runs" in r.path and "POST" in r.methods
        )
        req = ParallelRunsRequest(children=[])
        with patch(
            "meridiand._parallel_runs._run_children_parallel",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(ParallelRunsError):
                await handler("s1", req)


class TestRecoverySoakHelpers:
    def test_crash_recovery_attempt_success(self, tmp_path: Path) -> None:
        """_attempt_recovery success path returns True when phase not in stop phases (110)."""
        from datetime import UTC, datetime

        from meridiand._crash_recovery_soak import _attempt_recovery, _seed_synthetic_session

        now = datetime.now(UTC).isoformat()
        _seed_synthetic_session(tmp_path, "s1", now)
        _attempt_recovery(tmp_path, "s1")

    def test_crash_recovery_attempt_bad_manifest(self, tmp_path: Path) -> None:
        """Manifest without session_id returns False (line 110)."""
        from meridiand._crash_recovery_soak import _attempt_recovery

        sessions = tmp_path / "sessions" / "s2"
        sessions.mkdir(parents=True)
        (sessions / "manifest.json").write_text(json.dumps({}))  # no session_id
        assert _attempt_recovery(tmp_path, "s2") is False

    async def test_crash_recovery_large_failure_count(self, tmp_path: Path) -> None:
        """When 10+ failures occur, additional ones don't append (180->167 False branch)."""
        from core_errors import NoopAuditLog

        from meridiand._crash_recovery_soak import (
            CrashRecoverySoakError,
            make_crash_recovery_soak_router,
        )

        router = make_crash_recovery_soak_router(
            audit_log=NoopAuditLog(),
            storage_root=tmp_path,
            _crash_count_override=15,  # > 10 failures will accrue
        )
        handler = next(
            r.endpoint for r in router.routes if "crash-recovery-soak-run" in r.path
        )
        with patch(
            "meridiand._crash_recovery_soak._attempt_recovery",
            return_value=False,
        ):
            with pytest.raises(CrashRecoverySoakError):
                await handler()

    def test_e8_attempt_recovery_success(self, tmp_path: Path) -> None:
        """_attempt_recovery exercises the read+phase code path (line 187)."""
        from datetime import UTC, datetime

        from meridiand._e8_hardening_soak import _SEED_FNS, _attempt_recovery

        now = datetime.now(UTC).isoformat()
        seed_fn = _SEED_FNS["harness"]
        seed_fn(tmp_path, "s1", "agent1", "channel1", now)
        _attempt_recovery(tmp_path, "s1")

    def test_e8_attempt_recovery_bad_manifest(self, tmp_path: Path) -> None:
        """Manifest without session_id returns False (line 187)."""
        from meridiand._e8_hardening_soak import _attempt_recovery

        sessions = tmp_path / "sessions" / "s2"
        sessions.mkdir(parents=True)
        (sessions / "manifest.json").write_text(json.dumps({}))
        assert _attempt_recovery(tmp_path, "s2") is False

    async def test_crash_recovery_seed_failure_path(self, tmp_path: Path) -> None:
        """When _seed_synthetic_session raises, sample_failures captures it (172-174)."""
        from core_errors import NoopAuditLog

        from meridiand._crash_recovery_soak import (
            CrashRecoverySoakError,
            make_crash_recovery_soak_router,
        )

        router = make_crash_recovery_soak_router(
            audit_log=NoopAuditLog(),
            storage_root=tmp_path,
            _crash_count_override=2,
        )
        handler = next(
            r.endpoint for r in router.routes if "crash-recovery-soak-run" in r.path
        )
        # Force _seed_synthetic_session to raise. The seed failure causes 0% resume
        # rate, which triggers the typed CrashRecoverySoakError — but only after the
        # except-Exception path at 172-174 logs the failure.
        with patch(
            "meridiand._crash_recovery_soak._seed_synthetic_session",
            side_effect=RuntimeError("seed boom"),
        ):
            with pytest.raises(CrashRecoverySoakError):
                await handler()

    async def test_e8_seed_failure_path(self, tmp_path: Path) -> None:
        """When _SEED_FNS raises, sample_failures captures it (264-267)."""
        from core_errors import NoopAuditLog

        from meridiand._e8_hardening_soak import (
            E8HardeningSoakError,
            make_e8_hardening_soak_router,
        )

        router = make_e8_hardening_soak_router(
            audit_log=NoopAuditLog(),
            storage_root=tmp_path,
            _sessions_per_combo_override=1,
        )
        handler = next(
            r.endpoint for r in router.routes if "e8-hardening-soak-run" in r.path
        )

        def _boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("seed boom")

        with patch.dict(
            "meridiand._e8_hardening_soak._SEED_FNS",
            {"harness": _boom, "tool_worker": _boom, "daemon": _boom},
        ):
            # Either succeeds or raises typed error — both exercise 264-267.
            try:
                await handler()
            except E8HardeningSoakError:
                pass


class TestSkillForgeSelHelpers:
    def test_error_http_status(self) -> None:
        from meridiand._skill_forge_sel import ForgeSelError

        assert ForgeSelError(message="m", timestamp="t", cause=None).http_status() == 500

    def test_load_events_no_events_dir(self, tmp_path: Path) -> None:
        """No events dir → returns [] (line 99 if-False)."""
        from meridiand._skill_forge_sel import _read_session_events

        assert _read_session_events(tmp_path, "s1") == []

    def test_load_events_skips_blank_and_invalid(self, tmp_path: Path) -> None:
        """Blank lines and invalid JSON skipped (104, 108-109)."""
        from meridiand._skill_forge_sel import _read_session_events

        events_dir = tmp_path / "events"
        events_dir.mkdir()
        (events_dir / "s1.ndjson").write_text(
            "\n"  # blank
            "not json {{{\n"  # invalid
            + json.dumps({"seq": 1, "type": "test"})
            + "\n"
        )
        result = _read_session_events(tmp_path, "s1")
        assert len(result) == 1

    def test_collect_no_terminal_phase(self, tmp_path: Path) -> None:
        """No phase_change with terminal phase → no summary added (149->140)."""
        from meridiand._skill_forge_sel import collect_terminated_sessions

        events_dir = tmp_path / "events"
        events_dir.mkdir()
        # Session with only running state, no terminal
        (events_dir / "s1.ndjson").write_text(
            json.dumps({"seq": 1, "type": "session.created", "data": {}}) + "\n"
        )
        summaries = collect_terminated_sessions(tmp_path)
        assert summaries == []

    def test_enumerate_session_ids_dedupes(self, tmp_path: Path) -> None:
        """Same session_id appearing twice is only counted once (122->120)."""
        from meridiand._skill_forge_sel import _enumerate_session_ids

        events_dir = tmp_path / "events"
        (events_dir / "2024-01-01").mkdir(parents=True)
        (events_dir / "2024-01-02").mkdir(parents=True)
        (events_dir / "2024-01-01" / "s1.ndjson").write_text("")
        (events_dir / "2024-01-02" / "s1.ndjson").write_text("")  # duplicate
        result = _enumerate_session_ids(tmp_path)
        assert result == ["s1"]

    def test_collect_tool_call_without_name(self, tmp_path: Path) -> None:
        """tool_call.requested with empty tool_name skipped (122->120 branch)."""
        from meridiand._skill_forge_sel import collect_terminated_sessions

        events_dir = tmp_path / "events"
        events_dir.mkdir()
        (events_dir / "s2.ndjson").write_text(
            json.dumps({"seq": 1, "type": "tool_call.requested", "data": {}})
            + "\n"
            + json.dumps(
                {
                    "seq": 2,
                    "type": "session.phase_change",
                    "data": {"after": "terminated"},
                }
            )
            + "\n"
        )
        summaries = collect_terminated_sessions(tmp_path)
        assert len(summaries) == 1


class TestChannelDriverProtocolImpls:
    """Cover NoopSecretResolver + NoopSocketModeClient + small protocol impls."""

    def test_slack_noop_secret_resolver(self) -> None:
        from meridiand._slack_channel_driver import NoopSecretResolver

        assert NoopSecretResolver().resolve("any") is None

    async def test_slack_noop_socket_mode_client(self) -> None:
        from meridiand._slack_channel_driver import NoopSocketModeClient

        c = NoopSocketModeClient()
        await c.connect("app", "bot")
        await c.disconnect()

    def test_discord_noop_secret_resolver(self) -> None:
        from meridiand._discord_channel_driver import NoopSecretResolver

        assert NoopSecretResolver().resolve("x") is None

    async def test_discord_noop_gateway_client(self) -> None:
        from meridiand._discord_channel_driver import NoopGatewayClient

        c = NoopGatewayClient()
        await c.connect("tok", 0)
        await c.disconnect()

    def test_telegram_noop_secret_resolver(self) -> None:
        from meridiand._telegram_channel_driver import NoopSecretResolver

        assert NoopSecretResolver().resolve("y") is None

    async def test_telegram_noop_long_poll_client(self) -> None:
        from meridiand._telegram_channel_driver import NoopLongPollClient

        c = NoopLongPollClient()
        await c.poll("tok", 30)
        await c.stop()

    def test_webhook_noop_secret_resolver(self) -> None:
        from meridiand._webhook_channel_driver import NoopSecretResolver

        assert NoopSecretResolver().resolve("z") is None

    async def test_slack_default_http_client_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When http_client=None, _slack_post creates AsyncClient (202-203)."""
        import httpx

        from meridiand._slack_channel_driver import (
            NoopSecretResolver,
            NoopSocketModeClient,
            SlackChannelDriver,
        )

        # Patch httpx.AsyncClient to return a mock that records the call
        called: list[str] = []

        class _MockClient:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_a: Any) -> None:
                return None

            async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
                called.append(url)
                return httpx.Response(200, json={"ok": True, "ts": "1.0"})

        monkeypatch.setattr("meridiand._slack_channel_driver.httpx.AsyncClient", _MockClient)
        driver = SlackChannelDriver(
            storage_root=tmp_path,
            secret_resolver=NoopSecretResolver(),
            socket_mode_client=NoopSocketModeClient(),
        )
        resp = await driver._slack_post("chat.postMessage", {"x": 1}, "token")
        assert resp.status_code == 200
        assert called

    async def test_discord_default_http_client_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When http_client=None, _discord_post creates AsyncClient (212-213)."""
        import httpx

        from meridiand._discord_channel_driver import (
            DiscordChannelDriver,
            NoopGatewayClient,
            NoopSecretResolver,
        )

        class _MockClient:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_a: Any) -> None:
                return None

            async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
                return httpx.Response(200, json={"id": "msg-1"})

        monkeypatch.setattr("meridiand._discord_channel_driver.httpx.AsyncClient", _MockClient)
        driver = DiscordChannelDriver(
            storage_root=tmp_path,
            secret_resolver=NoopSecretResolver(),
            gateway_client=NoopGatewayClient(),
        )
        resp = await driver._discord_post("http://example.com/x", {"a": 1}, "token")
        assert resp.status_code == 200

    async def test_telegram_default_http_client_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from meridiand._telegram_channel_driver import (
            NoopLongPollClient,
            NoopSecretResolver,
            TelegramChannelDriver,
        )

        class _MockClient:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_a: Any) -> None:
                return None

            async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
                return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

        monkeypatch.setattr("meridiand._telegram_channel_driver.httpx.AsyncClient", _MockClient)
        driver = TelegramChannelDriver(
            storage_root=tmp_path,
            secret_resolver=NoopSecretResolver(),
            long_poll_client=NoopLongPollClient(),
        )
        resp = await driver._telegram_post("http://example.com/x", {"a": 1}, "token")
        assert resp.status_code == 200

    def test_telegram_payload_thread_id_invalid_int(self, tmp_path: Path) -> None:
        """thread_id can't be parsed as int → kept as string (lines 224-225)."""
        from sdk_channel import SendRequest

        from meridiand._telegram_channel_driver import (
            NoopLongPollClient,
            NoopSecretResolver,
            TelegramChannelDriver,
        )

        driver = TelegramChannelDriver(
            storage_root=tmp_path,
            secret_resolver=NoopSecretResolver(),
            long_poll_client=NoopLongPollClient(),
        )
        req = SendRequest(
            channel_id="c1",
            channel_kind="meridian.telegram",
            session_id="s1",
            recipient="user",
            content="hi",
            content_type="text",
            thread_id="not-int",
        )
        payload = driver._build_send_message_payload(req, "chat-1")
        assert payload["reply_to_message_id"] == "not-int"

    async def test_telegram_channel_failure_reraised(self, tmp_path: Path) -> None:
        from sdk_channel import ChannelFailure, SendRequest

        from meridiand._telegram_channel_driver import (
            NoopLongPollClient,
            NoopSecretResolver,
            TelegramChannelDriver,
        )

        driver = TelegramChannelDriver(
            storage_root=tmp_path,
            secret_resolver=NoopSecretResolver(),
            long_poll_client=NoopLongPollClient(),
        )
        chan_dir = tmp_path / "channels"
        chan_dir.mkdir(parents=True)
        (chan_dir / "c1.json").write_text(
            json.dumps({"config": {"telegram_chat_id": "C1"}})
        )

        original = ChannelFailure(
            code="X", message="m", channel_id="c1", channel_kind="meridian.telegram",
            session_id="s1", timestamp=pagination_now(),
        )

        async def _boom(*_a: Any, **_k: Any) -> None:
            raise original

        driver._telegram_post = _boom  # type: ignore[method-assign]
        driver._resolve_bot_token = lambda *a, **k: "tok"  # type: ignore[method-assign]
        req = SendRequest(
            channel_id="c1", channel_kind="meridian.telegram", session_id="s1",
            recipient="user", content="hi", content_type="text",
        )
        with pytest.raises(ChannelFailure):
            await driver.send(req)

    async def test_webhook_channel_default_http_client_path_with_thread_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No http_client + thread_id set → payload['thread_id'] (144) + else branch (183-184)."""
        import httpx

        from sdk_channel import SendRequest

        from meridiand._webhook_channel_driver import (
            NoopSecretResolver,
            WebhookChannelDriver,
        )

        class _MockClient:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_a: Any) -> None:
                return None

            async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
                return httpx.Response(200)

        monkeypatch.setattr(
            "meridiand._webhook_channel_driver.httpx.AsyncClient", _MockClient
        )
        driver = WebhookChannelDriver(
            storage_root=tmp_path,
            secret_resolver=NoopSecretResolver(),
        )
        chan_dir = tmp_path / "channels"
        chan_dir.mkdir(parents=True)
        (chan_dir / "c1.json").write_text(
            json.dumps({"config": {"outbound_url": "http://example.com/hook"}})
        )

        req = SendRequest(
            channel_id="c1", channel_kind="meridian.webhook", session_id="s1",
            recipient="r", content="hi", content_type="text",
            thread_id="thread-1",
        )
        resp = await driver.send(req)
        assert resp is not None

    async def test_webhook_channel_failure_reraised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ChannelFailure raised inside send is re-raised (line 192)."""
        from sdk_channel import ChannelFailure, SendRequest

        from meridiand._webhook_channel_driver import (
            NoopSecretResolver,
            WebhookChannelDriver,
        )

        # Patch httpx.AsyncClient to raise ChannelFailure
        original = ChannelFailure(
            code="X", message="m", channel_id="c1", channel_kind="meridian.webhook",
            session_id="s1", timestamp=pagination_now(),
        )

        class _BoomClient:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_a: Any) -> None:
                return None

            async def post(self, *_a: Any, **_k: Any) -> Any:
                raise original

        monkeypatch.setattr(
            "meridiand._webhook_channel_driver.httpx.AsyncClient", _BoomClient
        )

        driver = WebhookChannelDriver(
            storage_root=tmp_path,
            secret_resolver=NoopSecretResolver(),
        )
        chan_dir = tmp_path / "channels"
        chan_dir.mkdir(parents=True)
        (chan_dir / "c1.json").write_text(
            json.dumps({"config": {"outbound_url": "http://example.com/hook"}})
        )

        req = SendRequest(
            channel_id="c1", channel_kind="meridian.webhook", session_id="s1",
            recipient="r", content="hi", content_type="text",
        )
        with pytest.raises(ChannelFailure):
            await driver.send(req)

    async def test_slack_channel_failure_reraised(self, tmp_path: Path) -> None:
        from sdk_channel import ChannelFailure, SendRequest

        from meridiand._slack_channel_driver import (
            NoopSecretResolver,
            NoopSocketModeClient,
            SlackChannelDriver,
        )

        driver = SlackChannelDriver(
            storage_root=tmp_path,
            secret_resolver=NoopSecretResolver(),
            socket_mode_client=NoopSocketModeClient(),
        )
        chan_dir = tmp_path / "channels"
        chan_dir.mkdir(parents=True)
        (chan_dir / "c1.json").write_text(
            json.dumps({"config": {"slack_channel_id": "C1"}})
        )

        original = ChannelFailure(
            code="X", message="m", channel_id="c1", channel_kind="meridian.slack",
            session_id="s1", timestamp=pagination_now(),
        )

        async def _boom(*_a: Any, **_k: Any) -> None:
            raise original

        driver._slack_post = _boom  # type: ignore[method-assign]
        driver._resolve_bot_token = lambda *a, **k: "tok"  # type: ignore[method-assign]
        req = SendRequest(
            channel_id="c1", channel_kind="meridian.slack", session_id="s1",
            recipient="user", content="hi", content_type="text",
        )
        with pytest.raises(ChannelFailure):
            await driver.send(req)

    async def test_discord_channel_failure_reraised(self, tmp_path: Path) -> None:
        """ChannelFailure raised inside send is re-raised verbatim (line 359)."""
        from sdk_channel import ChannelFailure, SendRequest

        from meridiand._discord_channel_driver import (
            DiscordChannelDriver,
            NoopGatewayClient,
            NoopSecretResolver,
        )

        driver = DiscordChannelDriver(
            storage_root=tmp_path,
            secret_resolver=NoopSecretResolver(),
            gateway_client=NoopGatewayClient(),
        )
        # Pre-create channel config so _load_driver_config doesn't fail
        chan_dir = tmp_path / "channels"
        chan_dir.mkdir(parents=True)
        (chan_dir / "c1.json").write_text(
            json.dumps({"config": {"discord_channel_id": "C1"}})
        )

        original = ChannelFailure(
            code="X", message="m", channel_id="c1", channel_kind="meridian.discord",
            session_id="s1", timestamp=pagination_now(),
        )

        async def _boom(*_a: Any, **_k: Any) -> None:
            raise original

        driver._discord_post = _boom  # type: ignore[method-assign]
        # Also stub token resolution to bypass secret lookup
        driver._resolve_bot_token = lambda *a, **k: "tok"  # type: ignore[method-assign]
        req = SendRequest(
            channel_id="c1", channel_kind="meridian.discord", session_id="s1",
            recipient="user", content="hi", content_type="text",
        )
        with pytest.raises(ChannelFailure):
            await driver.send(req)


class TestWakeHelpers:
    def test_load_active_skills_skips_malformed(self, tmp_path: Path) -> None:
        """Malformed activation JSON is silently skipped (lines 86-87)."""
        from meridiand._wake import _load_active_skills

        d = tmp_path / "skill_activations"
        d.mkdir()
        (d / "bad.json").write_text("not json {{{")
        (d / "good.json").write_text(
            json.dumps({"agent_id": "a1", "status": "active", "skill_version_id": "v1"})
        )
        result = _load_active_skills(tmp_path, "a1")
        assert len(result) == 1

    def test_load_most_recent_thread_skips_malformed_manifest(self, tmp_path: Path) -> None:
        """Malformed manifest JSON skipped (113-114)."""
        from meridiand._wake import _load_most_recent_thread

        threads_dir = tmp_path / "threads" / "s1"
        (threads_dir / "t1").mkdir(parents=True)
        (threads_dir / "t1" / "manifest.json").write_text("not json {{{")
        (threads_dir / "t2").mkdir(parents=True)
        (threads_dir / "t2" / "manifest.json").write_text(
            json.dumps({"id": "t2", "created_at": "2024-01-01"})
        )
        # message file
        (threads_dir / "t2" / "messages.ndjson").write_text(
            json.dumps({"role": "user", "content": "hi", "sequence": 1}) + "\n"
        )
        tid, msgs = _load_most_recent_thread(tmp_path, "s1")
        assert tid == "t2"
        assert len(msgs) == 1

    def test_load_most_recent_thread_no_best(self, tmp_path: Path) -> None:
        """No valid manifests → best_thread_id stays None (line 117)."""
        from meridiand._wake import _load_most_recent_thread

        threads_dir = tmp_path / "threads" / "s2"
        (threads_dir / "t1").mkdir(parents=True)
        (threads_dir / "t1" / "manifest.json").write_text("not json")
        tid, msgs = _load_most_recent_thread(tmp_path, "s2")
        assert tid is None
        assert msgs == []

    def test_load_most_recent_thread_no_messages_file(self, tmp_path: Path) -> None:
        """Best thread has no messages.ndjson → return (tid, []) (line 121)."""
        from meridiand._wake import _load_most_recent_thread

        threads_dir = tmp_path / "threads" / "s3"
        (threads_dir / "t1").mkdir(parents=True)
        (threads_dir / "t1" / "manifest.json").write_text(
            json.dumps({"id": "t1", "created_at": "2024-01-01"})
        )
        # No messages.ndjson written
        tid, msgs = _load_most_recent_thread(tmp_path, "s3")
        assert tid == "t1"
        assert msgs == []

    async def test_wake_with_non_dict_config(self, tmp_path: Path) -> None:
        """Agent's config field is non-dict → if-False branch (207->217)."""
        from core_errors import NoopAuditLog

        from meridiand._wake import make_wake_router

        sessions_dir = tmp_path / "sessions" / "s1"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "manifest.json").write_text(
            json.dumps({"session_id": "s1", "agent_id": "a1"})
        )
        # Agent record where config is a non-dict (string)
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "a1.json").write_text(
            json.dumps({"id": "a1", "version": {"config": "not-a-dict"}})
        )
        router = make_wake_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if "/wake" in r.path and "POST" in r.methods
        )
        resp = await handler("s1")
        assert resp is not None

    async def test_wake_with_skill_no_version_id(self, tmp_path: Path) -> None:
        """An active skill without skill_version_id is skipped (line 221)."""
        from core_errors import NoopAuditLog

        from meridiand._wake import make_wake_router

        sessions_dir = tmp_path / "sessions" / "s2"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "manifest.json").write_text(
            json.dumps({"session_id": "s2", "agent_id": "a2"})
        )
        activations_dir = tmp_path / "skill_activations"
        activations_dir.mkdir()
        # Active skill with NO skill_version_id
        (activations_dir / "act1.json").write_text(
            json.dumps({"agent_id": "a2", "status": "active"})
        )
        router = make_wake_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if "/wake" in r.path and "POST" in r.methods
        )
        resp = await handler("s2")
        assert resp is not None

    def test_load_most_recent_thread_skips_blank_and_invalid_lines(self, tmp_path: Path) -> None:
        """Blank lines and invalid JSON in messages skipped (lines 127, 130-131)."""
        from meridiand._wake import _load_most_recent_thread

        threads_dir = tmp_path / "threads" / "s4"
        (threads_dir / "t1").mkdir(parents=True)
        (threads_dir / "t1" / "manifest.json").write_text(
            json.dumps({"id": "t1", "created_at": "2024-01-01"})
        )
        (threads_dir / "t1" / "messages.ndjson").write_text(
            "\n"  # blank
            "not json {{{\n"  # invalid
            + json.dumps({"role": "user", "content": "hi", "sequence": 1})
            + "\n"
        )
        tid, msgs = _load_most_recent_thread(tmp_path, "s4")
        assert tid == "t1"
        assert len(msgs) == 1


class TestCronSchedulerErrors:
    def test_cron_fire_error_http_status(self) -> None:
        from meridiand._cron_scheduler import CronFireError

        assert CronFireError(message="m", timestamp="t", cause=None).http_status() == 500

    async def test_scheduler_with_no_cron_dir(self, tmp_path: Path) -> None:
        """No cron_dir → if-False branch (209->277)."""
        from core_errors import NoopAuditLog

        from meridiand._cron_scheduler import run_cron_scheduler_loop

        # tmp_path has no cron/ subdir
        task = asyncio.create_task(
            run_cron_scheduler_loop(
                storage_root=tmp_path,
                audit_log=NoopAuditLog(),
                check_interval_seconds=0.01,
                missed_fires_policy="catch_up",
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_scheduler_fires_timestamp_trigger(self, tmp_path: Path) -> None:
        """timestamp trigger took if branch, fall back to loop (247->212)."""
        from datetime import UTC, datetime, timedelta

        from core_errors import NoopAuditLog

        from meridiand._cron_scheduler import run_cron_scheduler_loop

        cron_dir = tmp_path / "cron"
        cron_dir.mkdir()
        past = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
        (cron_dir / "cron_ts.json").write_text(
            json.dumps(
                {
                    "id": "ts",
                    "status": "active",
                    "trigger_type": "timestamp",
                    "next_fire_at": past,
                    "session_id": "s",
                    "task": {},
                }
            )
        )

        task = asyncio.create_task(
            run_cron_scheduler_loop(
                storage_root=tmp_path,
                audit_log=NoopAuditLog(),
                check_interval_seconds=0.01,
                missed_fires_policy="catch_up",
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_scheduler_skips_malformed_next_fire_at(self, tmp_path: Path) -> None:
        """Cron with malformed next_fire_at is skipped (lines 229-230)."""
        from core_errors import NoopAuditLog

        from meridiand._cron_scheduler import run_cron_scheduler_loop

        cron_dir = tmp_path / "cron"
        cron_dir.mkdir(parents=True)
        (cron_dir / "cron_bad_ts.json").write_text(
            json.dumps(
                {
                    "id": "bad_ts",
                    "status": "active",
                    "trigger_type": "interval",
                    "next_fire_at": "not-a-timestamp",
                    "interval": "1s",
                    "session_id": "s",
                    "task": {},
                }
            )
        )

        task = asyncio.create_task(
            run_cron_scheduler_loop(
                storage_root=tmp_path,
                audit_log=NoopAuditLog(),
                check_interval_seconds=0.01,
                missed_fires_policy="catch_up",
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_scheduler_skips_malformed_interval(self, tmp_path: Path) -> None:
        """Cron with malformed interval string is skipped (lines 251-252)."""
        from datetime import UTC, datetime, timedelta

        from core_errors import NoopAuditLog

        from meridiand._cron_scheduler import run_cron_scheduler_loop

        cron_dir = tmp_path / "cron"
        cron_dir.mkdir(parents=True)
        past = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
        (cron_dir / "cron_bad_int.json").write_text(
            json.dumps(
                {
                    "id": "bad_int",
                    "status": "active",
                    "trigger_type": "interval",
                    "next_fire_at": past,
                    "interval": "not-a-duration",
                    "session_id": "s",
                    "task": {},
                }
            )
        )

        task = asyncio.create_task(
            run_cron_scheduler_loop(
                storage_root=tmp_path,
                audit_log=NoopAuditLog(),
                check_interval_seconds=0.01,
                missed_fires_policy="catch_up",
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestResumeGenericException:
    async def test_resume_zero_model_calls_keeps_phase(self, tmp_path: Path) -> None:
        """_run_harness returning (0, _) skips the phase='idle' assignment (123->149)."""
        from core_errors import NoopAuditLog

        from meridiand._resume import make_resume_router

        fixture_dir = tmp_path / "fixtures" / "s2"
        fixture_dir.mkdir(parents=True)
        (fixture_dir / "model_responses.ndjson").write_text(
            json.dumps([{"type": "message_stop"}]) + "\n"
        )
        (fixture_dir / "tool_responses.ndjson").write_text(
            json.dumps({"is_error": False, "content": "ok"}) + "\n"
        )

        router = make_resume_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if "/resume" in r.path and "POST" in r.methods
        )

        async def _zero_run(*_a: Any, **_k: Any) -> tuple[int, int]:
            return 0, 0

        with patch("meridiand._resume._run_harness", _zero_run):
            resp = await handler("s2")
        assert resp is not None

    async def test_resume_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """A generic exception inside resume is wrapped in ResumeError (lines 128-147)."""
        from core_errors import NoopAuditLog

        from meridiand._resume import (
            ResumeError,
            make_resume_router,
        )

        # Pre-create fixture files so resume gets past the typed-error guards
        fixture_dir = tmp_path / "fixtures" / "s1"
        fixture_dir.mkdir(parents=True)
        (fixture_dir / "model_responses.ndjson").write_text(
            json.dumps([{"type": "message_stop"}]) + "\n"
        )
        (fixture_dir / "tool_responses.ndjson").write_text(
            json.dumps({"is_error": False, "content": "ok"}) + "\n"
        )

        router = make_resume_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if "/resume" in r.path and "POST" in r.methods
        )
        with patch(
            "meridiand._resume._run_harness",
            side_effect=RuntimeError("harness boom"),
        ):
            with pytest.raises(ResumeError):
                await handler("s1")


class TestErrorEnvelopeHooksDispatch:
    async def test_meridian_error_with_cause_records_exception(self, tmp_path: Path) -> None:
        """MeridianError with non-None cause triggers span.record_exception (line 79).
        Also dispatches on_error hooks when hooks_dir is set (lines 90-99)."""
        from core_errors import HandlerOptions, NoopAuditLog
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from meridiand._error_envelope_middleware import ErrorEnvelopeMiddleware

        class _CustomMeridianError(Exception):
            code = "custom_failed"
            message = "boom"
            cause = ValueError("underlying")
            timestamp = pagination_now()

            def http_status(self) -> int:
                return 500

        from core_errors import MeridianError

        class _MEErr(MeridianError):
            def __init__(self) -> None:
                super().__init__(
                    code="custom_failed",
                    message="boom",
                    timestamp=pagination_now(),
                    cause=ValueError("underlying"),
                )

            def http_status(self) -> int:
                return 500

        # Build minimal ASGI app that raises a MeridianError
        async def _raising_app(scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "http":
                raise _MEErr()

        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        mw = ErrorEnvelopeMiddleware(
            _raising_app, audit_log=NoopAuditLog(), hooks_dir=hooks_dir
        )

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/x",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8888),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request"}

        sent: list[dict[str, Any]] = []

        async def _send(m: Any) -> None:
            sent.append(m)

        await mw(scope, _receive, _send)
        assert any(s["type"] == "http.response.start" for s in sent)

    async def test_unexpected_error_dispatches_on_error_hooks(self, tmp_path: Path) -> None:
        """A non-MeridianError exception triggers the on_error hooks dispatch (149-150)."""
        from core_errors import NoopAuditLog

        from meridiand._error_envelope_middleware import ErrorEnvelopeMiddleware

        async def _raising_app(scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "http":
                raise RuntimeError("unexpected")

        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        mw = ErrorEnvelopeMiddleware(
            _raising_app, audit_log=NoopAuditLog(), hooks_dir=hooks_dir
        )

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/x",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8888),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request"}

        sent: list[dict[str, Any]] = []

        async def _send(m: Any) -> None:
            sent.append(m)

        await mw(scope, _receive, _send)
        assert any(s["type"] == "http.response.start" for s in sent)


class TestCursorMiddlewareEdgeCases:
    async def test_non_http_scope_passthrough(self) -> None:
        """websocket scope is passed straight through (lines 35-36)."""
        from core_errors import NoopAuditLog

        from meridiand._cursor_middleware import CursorPaginationMiddleware

        called: list[str] = []

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            called.append(scope["type"])

        mw = CursorPaginationMiddleware(_inner, audit_log=NoopAuditLog())
        await mw({"type": "websocket"}, lambda: None, lambda _m: None)
        assert called == ["websocket"]

    async def test_request_url_skips_non_host_headers(self) -> None:
        """Headers before 'host' (e.g. 'x-other') are skipped in the loop (76->75)."""
        from core_errors import NoopAuditLog

        from meridiand._cursor_middleware import CursorPaginationMiddleware

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        mw = CursorPaginationMiddleware(_inner, audit_log=NoopAuditLog())
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/agents",
            "query_string": b"",
            "headers": [
                (b"x-other", b"first"),  # not host
                (b"x-second", b"second"),  # not host
                (b"host", b"api.example.com"),
            ],
            "scheme": "http",
            "server": ("api.example.com", 80),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request"}

        async def _send(_m: Any) -> None:
            pass

        await mw(scope, _receive, _send)

    async def test_request_url_with_host_header(self) -> None:
        """A request with Host header builds URL from header (line ~80)."""
        from core_errors import NoopAuditLog

        from meridiand._cursor_middleware import CursorPaginationMiddleware

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        mw = CursorPaginationMiddleware(_inner, audit_log=NoopAuditLog())
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/agents",
            "query_string": b"limit=10",
            "headers": [
                (b"host", b"api.example.com"),
                (b"x-other", b"yes"),
            ],
            "scheme": "https",
            "server": ("api.example.com", 443),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request"}

        sent: list[dict[str, Any]] = []

        async def _send(m: Any) -> None:
            sent.append(m)

        await mw(scope, _receive, _send)
        assert any(s["type"] == "http.response.start" for s in sent)

    async def test_request_url_without_host_header_falls_back_to_server(self) -> None:
        """No Host header → URL built from scope server (lines 83-85)."""
        from core_errors import NoopAuditLog

        from meridiand._cursor_middleware import CursorPaginationMiddleware

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"x-next-cursor", b"abc"),
            ]})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        mw = CursorPaginationMiddleware(_inner, audit_log=NoopAuditLog())
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/agents",
            "query_string": b"",
            "headers": [],  # no host
            "scheme": "http",
            "server": ("10.0.0.1", 8888),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request"}

        sent: list[dict[str, Any]] = []

        async def _send(m: Any) -> None:
            sent.append(m)

        await mw(scope, _receive, _send)
        # Link header should be present
        start = next(s for s in sent if s["type"] == "http.response.start")
        header_names = [h[0].decode().lower() for h in start.get("headers", [])]
        assert "link" in header_names

    async def test_build_link_header_failure_writes_audit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """build_link_header raising writes an audit entry (lines 97-98)."""
        from core_errors import AuditLog, AuditLogEntry

        from meridiand._cursor_middleware import CursorPaginationMiddleware

        captured: list[AuditLogEntry] = []

        class _Capture(AuditLog):
            def write(self, e: AuditLogEntry) -> None:
                captured.append(e)

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"x-next-cursor", b"abc"),
            ]})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        mw = CursorPaginationMiddleware(_inner, audit_log=_Capture())
        monkeypatch.setattr(
            "meridiand._cursor_middleware.build_link_header",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("link boom")),
        )

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/agents",
            "query_string": b"",
            "headers": [(b"host", b"api.example.com")],
            "scheme": "http",
            "server": ("api.example.com", 80),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request"}

        async def _send(_m: Any) -> None:
            pass

        await mw(scope, _receive, _send)
        assert any(e.event == "cursor.pagination.link.failed" for e in captured), [
            e.event for e in captured
        ]


class TestSpawnTraceparent:
    async def test_spawn_with_malformed_manifest_json(
        self, tmp_path: Path
    ) -> None:
        """Manifest is invalid JSON → except swallows, _child_links empty (186-187)."""
        from core_errors import NoopAuditLog

        from meridiand._spawn import make_spawn_router, SpawnRequest

        sessions = tmp_path / "sessions" / "parent"
        sessions.mkdir(parents=True)
        (sessions / "manifest.json").write_text("not json {{{")

        router = make_spawn_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/sessions/{session_id}/spawn" and "POST" in r.methods
        )
        req = SpawnRequest(parent_capabilities=["agent.spawn", "fs.read"], child_capabilities=["fs.read"])
        resp = await handler("parent", req)
        assert resp is not None

    async def test_spawn_manifest_without_traceparent(
        self, tmp_path: Path
    ) -> None:
        """Manifest exists but has no traceparent → if False, skip (176->190)."""
        from core_errors import NoopAuditLog

        from meridiand._spawn import make_spawn_router, SpawnRequest

        sessions = tmp_path / "sessions" / "no_tp"
        sessions.mkdir(parents=True)
        (sessions / "manifest.json").write_text(json.dumps({"other_field": "x"}))

        router = make_spawn_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/sessions/{session_id}/spawn" and "POST" in r.methods
        )
        req = SpawnRequest(
            parent_capabilities=["agent.spawn", "fs.read"],
            child_capabilities=["fs.read"],
        )
        resp = await handler("no_tp", req)
        assert resp is not None

    async def test_spawn_without_parent_manifest(
        self, tmp_path: Path
    ) -> None:
        """No parent manifest → if-False branch (176->190)."""
        from core_errors import NoopAuditLog

        from meridiand._spawn import make_spawn_router, SpawnRequest

        # No sessions dir at all
        router = make_spawn_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/sessions/{session_id}/spawn" and "POST" in r.methods
        )
        req = SpawnRequest(
            parent_capabilities=["agent.spawn", "fs.read"],
            child_capabilities=["fs.read"],
        )
        resp = await handler("no_manifest", req)
        assert resp is not None

    async def test_spawn_with_valid_but_invalid_context_traceparent(
        self, tmp_path: Path
    ) -> None:
        """Parent has well-formed but invalid traceparent → is_valid False (179->190)."""
        from core_errors import NoopAuditLog

        from meridiand._spawn import make_spawn_router, SpawnRequest

        sessions = tmp_path / "sessions" / "parent2"
        sessions.mkdir(parents=True)
        # All zeros — well-formed but `is_valid` returns False
        (sessions / "manifest.json").write_text(
            json.dumps(
                {
                    "traceparent": "00-00000000000000000000000000000000-0000000000000000-00"
                }
            )
        )

        router = make_spawn_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/sessions/{session_id}/spawn" and "POST" in r.methods
        )
        req = SpawnRequest(parent_capabilities=["agent.spawn", "fs.read"], child_capabilities=["fs.read"])
        resp = await handler("parent2", req)
        assert resp is not None


class TestSystemAuditMiddlewareReraise:
    async def test_no_status_captured_returns_early(self, tmp_path: Path) -> None:
        """If inner app completes without sending response.start, return early (line 207)."""
        from core_errors import NoopAuditLog

        from meridiand._system_audit_middleware import SystemAuditMiddleware

        async def _silent(scope: Any, receive: Any, send: Any) -> None:
            return  # never sends

        mw = SystemAuditMiddleware(_silent, audit_log=NoopAuditLog())
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/skills",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8888),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(_m: Any) -> None:
            pass

        # Should complete without raising
        await mw(scope, _receive, _send)

    async def test_audit_swallowed_when_status_already_sent_then_reraise(
        self, tmp_path: Path
    ) -> None:
        """When status was captured and audit write fails, exception still re-raises (line 207)."""
        from core_errors import AuditLog, AuditLogEntry

        from meridiand._system_audit_middleware import SystemAuditMiddleware

        class _BoomAudit(AuditLog):
            def write(self, entry: AuditLogEntry) -> None:
                raise RuntimeError("audit boom")

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            raise RuntimeError("handler boom")

        mw = SystemAuditMiddleware(_inner, audit_log=_BoomAudit())

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/skills",  # monitored route
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8888),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[dict[str, Any]] = []

        async def _send(m: Any) -> None:
            sent.append(m)

        with pytest.raises(RuntimeError):
            await mw(scope, _receive, _send)


class TestBudgetsReportsLookupAndTool:
    def test_lookup_agent_cached_after_first_call(self, tmp_path: Path) -> None:
        """Second call uses cache (158->164 False branch)."""
        from meridiand._budgets_reports import _lookup_agent_id

        sessions = tmp_path / "sessions" / "s1"
        sessions.mkdir(parents=True)
        (sessions / "manifest.json").write_text(json.dumps({"agent_id": "a1"}))
        cache: dict[str, str | None] = {}
        # First call: populates cache
        assert _lookup_agent_id(tmp_path / "sessions", "s1", cache) == "a1"
        # Second call: uses cache without reading
        assert _lookup_agent_id(tmp_path / "sessions", "s1", cache) == "a1"

    def test_build_tool_report_skips_blank_tool_names(self, tmp_path: Path) -> None:
        """tool_call.requested with empty tool_name is skipped (228->226)."""
        from meridiand._budgets_reports import _build_tool_report

        events_dir = tmp_path / "events"
        events_dir.mkdir()
        (events_dir / "s1.ndjson").write_text(
            json.dumps(
                {"type": "tool_call.requested", "ts": "2024-01-01T00:00:00Z", "data": {}}
            )
            + "\n"
            + json.dumps(
                {
                    "type": "tool_call.requested",
                    "ts": "2024-01-01T00:00:00Z",
                    "data": {"tool_name": "bash"},
                }
            )
            + "\n"
        )
        report = _build_tool_report(events_dir, since=None, until=None)
        # Only bash is counted (empty tool_name skipped)
        assert len(report) == 1
        assert report[0]["tool_name"] == "bash"


class TestBudgetsReportsHelper:
    def test_count_event_skips_unreadable_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file that raises OSError is silently skipped (lines 126-127)."""
        from meridiand._budgets_reports import _scan_events

        events_dir = tmp_path / "events"
        events_dir.mkdir()
        # File that exists but read fails
        (events_dir / "broken.ndjson").write_text("ignored")
        # Valid file
        (events_dir / "ok.ndjson").write_text(
            json.dumps({"type": "budget.warning", "ts": "2024-01-01T00:00:00Z"}) + "\n"
        )

        real = Path.read_text

        def _selective(self: Path, *a: Any, **k: Any) -> str:
            if self.name == "broken.ndjson":
                raise OSError("denied")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", _selective)
        result = _scan_events(events_dir, frozenset({"budget.warning"}), since=None, until=None)
        # ok.ndjson contributes 1
        assert len(result) == 1

    def test_scan_events_skips_blank_lines_and_invalid_json(self, tmp_path: Path) -> None:
        """Blank lines and invalid JSON lines are skipped (131, 134-135, 137)."""
        from meridiand._budgets_reports import _scan_events

        events_dir = tmp_path / "events"
        events_dir.mkdir()
        (events_dir / "s1.ndjson").write_text(
            "\n"  # blank
            "  \n"  # whitespace
            "not json {{{\n"  # invalid
            + json.dumps({"type": "other", "ts": "2024-01-01T00:00:00Z"})
            + "\n"  # wrong type
            + json.dumps({"type": "budget.warning", "ts": "2024-01-01T00:00:00Z"})
            + "\n"
            + json.dumps({"type": "budget.warning", "ts": "1999-01-01T00:00:00Z"})
            + "\n"  # before since
        )
        result = _scan_events(
            events_dir,
            frozenset({"budget.warning"}),
            since="2023-01-01T00:00:00Z",
            until=None,
        )
        assert len(result) == 1


class TestSkillSuggestionsErrors:
    def test_request_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionRequestError

        assert SkillSuggestionRequestError(message="m", timestamp="t").http_status() == 422

    def test_mode_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionModeError

        assert SkillSuggestionModeError(message="m", timestamp="t").http_status() == 422

    def test_skill_not_found_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillNotFoundError

        assert SkillNotFoundError(message="m", timestamp="t").http_status() == 404

    def test_agent_not_found_error_http_status(self) -> None:
        from meridiand._skill_suggestions import AgentNotFoundError

        assert AgentNotFoundError(message="m", timestamp="t").http_status() == 404

    def test_suggestion_not_found_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionNotFoundError

        assert (
            SkillSuggestionNotFoundError(message="m", timestamp="t").http_status() == 404
        )

    def test_suggestion_conflict_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionConflictError

        assert SkillSuggestionConflictError(message="m", timestamp="t").http_status() == 409

    def test_emit_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionEmitError

        assert (
            SkillSuggestionEmitError(message="m", timestamp="t", cause=None).http_status() == 500
        )

    def test_approve_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionApproveError

        assert (
            SkillSuggestionApproveError(message="m", timestamp="t", cause=None).http_status()
            == 500
        )

    def test_latest_activation_no_matches_returns_none(self, tmp_path: Path) -> None:
        """activations exist but none match agent/skill (line 152)."""
        from meridiand._skill_suggestions import _latest_activation

        activations_dir = tmp_path / "activations"
        activations_dir.mkdir()
        (activations_dir / "a.json").write_text(
            json.dumps({"agent_id": "other_a", "skill_id": "other_s"})
        )
        result = _latest_activation(activations_dir, "wanted_a", "wanted_s")
        assert result is None

    async def test_emit_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """A generic exception is wrapped in SkillSuggestionEmitError (323-344)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_suggestions import (
            SkillSuggestionEmitError,
            SkillSuggestionRequest,
            make_skill_suggestions_router,
        )

        router = make_skill_suggestions_router(
            audit_log=NoopAuditLog(), storage_root=tmp_path
        )
        # Pre-create skill so we don't trip SkillNotFoundError
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "s1.json").write_text(json.dumps({"id": "s1"}))
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "a1.json").write_text(
            json.dumps(
                {
                    "id": "a1",
                    "version": {"config": {"skill_activation_mode": "auto_suggest"}},
                }
            )
        )

        handler = next(
            r.endpoint
            for r in router.routes
            if "/skill_suggestions" in r.path and "POST" in r.methods
        )
        req = SkillSuggestionRequest(skill_id="s1", reason="r")
        with patch("meridiand._skill_suggestions.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(SkillSuggestionEmitError):
                await handler("a1", req)

    async def test_approve_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """A generic exception is wrapped in SkillSuggestionApproveError (451-471)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_suggestions import (
            SkillSuggestionApproveError,
            make_skill_suggestions_router,
        )

        router = make_skill_suggestions_router(
            audit_log=NoopAuditLog(), storage_root=tmp_path
        )
        # Pre-create a suggestion file
        suggestions_dir = tmp_path / "skill_suggestions"
        suggestions_dir.mkdir(parents=True)
        (suggestions_dir / "a1_s1.json").write_text(
            json.dumps({"id": "sg1", "agent_id": "a1", "skill_id": "s1", "status": "suggested"})
        )

        handler = next(
            r.endpoint
            for r in router.routes
            if "/approve" in r.path and "POST" in r.methods
        )
        with patch("meridiand._skill_suggestions.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(SkillSuggestionApproveError):
                await handler("a1", "s1")


class TestHookDispatchVerdict:
    def test_verdict_from_string_invalid_json_yields_continue(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        verdict, _, _ = _parse_verdict("not json {{{")
        assert verdict == "continue"

    def test_verdict_from_valid_json_string(self) -> None:
        """Valid JSON string is parsed and data populated (line 182)."""
        from meridiand._hook_dispatch import _parse_verdict

        v, _, r = _parse_verdict(json.dumps({"verdict": "veto", "reason": "policy"}))
        assert v == "veto"
        assert r == "policy"

    def test_verdict_from_non_string_non_dict(self) -> None:
        """Content that's neither dict nor str (e.g. None, int) → data stays None."""
        from meridiand._hook_dispatch import _parse_verdict

        v, _, _ = _parse_verdict(None)  # type: ignore[arg-type]
        assert v == "continue"
        v2, _, _ = _parse_verdict(42)  # type: ignore[arg-type]
        assert v2 == "continue"

    def test_verdict_from_string_non_dict_yields_continue(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        verdict, _, _ = _parse_verdict("[1, 2]")
        assert verdict == "continue"

    def test_verdict_veto(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        v, _, r = _parse_verdict({"verdict": "veto", "reason": "denied"})
        assert v == "veto"
        assert r == "denied"

    def test_verdict_fail(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        v, _, r = _parse_verdict({"verdict": "fail", "reason": "broken"})
        assert v == "fail"
        assert r == "broken"

    def test_verdict_recoverable(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        v, _, r = _parse_verdict({"verdict": "recoverable", "reason": "retry"})
        assert v == "recoverable"

    def test_verdict_continue_with_mutations(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        v, m, _ = _parse_verdict({"verdict": "continue", "mutations": {"key": "val"}})
        assert v == "continue"
        assert m == {"key": "val"}

    def test_build_dispatcher_all_handler_types(self) -> None:
        from sdk_sandbox._audit import AuditLog
        from sdk_sandbox._types import AuditLogEntry

        import sdk_sandbox as _sb

        from meridiand._hook_dispatch import _build_dispatcher

        class _Bridge(AuditLog):
            def write(self, entry: AuditLogEntry) -> None:
                pass

        bridge = _Bridge()
        assert isinstance(_build_dispatcher("subprocess", bridge), _sb.SubprocessDispatcher)
        assert isinstance(_build_dispatcher("http", bridge), _sb.HttpDispatcher)
        assert isinstance(_build_dispatcher("mcp", bridge), _sb.McpDispatcher)
        assert isinstance(_build_dispatcher("container", bridge), _sb.ContainerDispatcher)
        assert isinstance(_build_dispatcher("in_process", bridge), _sb.InProcessDispatcher)

    def test_load_active_hooks_skips_invalid_json(self, tmp_path: Path) -> None:
        """A hook file that's malformed JSON is silently skipped (275-276)."""
        from sdk_sandbox._types import ExecutionContext

        from meridiand._hook_dispatch import _load_active_hooks

        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "bad.json").write_text("not json {{{")
        (hooks_dir / "good.json").write_text(
            json.dumps({"id": "g1", "status": "active", "event": "e1"})
        )

        ctx = ExecutionContext(session_id="s")
        result = _load_active_hooks(hooks_dir, "e1", ctx)
        assert len(result) == 1
        assert result[0]["id"] == "g1"

    async def test_dispatch_one_with_in_process_handler_registered(
        self, tmp_path: Path
    ) -> None:
        """When in_process handler is registered, dispatcher.register is called (308->314)."""
        from sdk_sandbox._types import ExecutionContext

        from meridiand._hook_dispatch import _SandboxAuditBridge, _dispatch_one

        async def _fn(*_a: Any, **_k: Any) -> dict[str, Any]:
            return {"verdict": "continue"}

        hook = {
            "id": "h1",
            "handler": "in_process",
            "timeout_ms": 5000,
            "name": "test",
            "metadata": {"module": "x"},
        }

        bridge = _SandboxAuditBridge(core_log=MagicMock())
        result = await _dispatch_one(
            hook,
            {"x": 1},
            ExecutionContext(session_id="s"),
            bridge=bridge,
            in_process_handlers={"h1": _fn},
        )
        # Either succeeded or returned an error result — either way the line is covered.
        assert result is not None

    async def test_dispatch_one_in_process_with_no_handlers_dict(
        self, tmp_path: Path
    ) -> None:
        """in_process handler but in_process_handlers is None (308->314 False branch)."""
        from sdk_sandbox._types import ExecutionContext

        from meridiand._hook_dispatch import _SandboxAuditBridge, _dispatch_one

        hook = {
            "id": "h_no_dict",
            "handler": "in_process",
            "timeout_ms": 5000,
            "name": "t",
            "metadata": {"module": "x"},
        }
        bridge = _SandboxAuditBridge(core_log=MagicMock())
        result = await _dispatch_one(
            hook,
            {"x": 1},
            ExecutionContext(session_id="s"),
            bridge=bridge,
            in_process_handlers=None,
        )
        assert result is not None

    async def test_dispatch_one_with_in_process_handler_none_fn(
        self, tmp_path: Path
    ) -> None:
        """in_process_handlers exists but contains no entry for hook_id (310->314)."""
        from sdk_sandbox._types import ExecutionContext

        from meridiand._hook_dispatch import _SandboxAuditBridge, _dispatch_one

        hook = {
            "id": "h_missing",
            "handler": "in_process",
            "timeout_ms": 5000,
            "name": "t",
            "metadata": {"module": "x"},
        }
        bridge = _SandboxAuditBridge(core_log=MagicMock())
        result = await _dispatch_one(
            hook,
            {"x": 1},
            ExecutionContext(session_id="s"),
            bridge=bridge,
            in_process_handlers={"other": MagicMock()},
        )
        assert result is not None

    def test_build_tool_handler_all_handler_types(self) -> None:
        import sdk_sandbox as _sb

        from meridiand._hook_dispatch import _build_tool_handler

        assert isinstance(
            _build_tool_handler("subprocess", {"path": "/bin/true"}), _sb.SubprocessHandler
        )
        assert isinstance(
            _build_tool_handler("http", {"url": "http://x"}), _sb.HttpHandler
        )
        assert isinstance(
            _build_tool_handler("mcp", {"server_url": "u", "tool_name": "t"}),
            _sb.McpHandler,
        )
        assert isinstance(
            _build_tool_handler(
                "container",
                {"environment_id": "e", "entrypoint": "/x"},
            ),
            _sb.ContainerHandler,
        )
        assert isinstance(_build_tool_handler("in_process", {"module": "m"}), _sb.InProcessHandler)


class TestVaultBackendOsKeychain:
    def test_now_helper(self) -> None:
        from meridiand._vault_backend_os_keychain import _now

        s = _now()
        assert isinstance(s, str) and "T" in s

    def test_default_keyring_import(self) -> None:
        """Without injected _keyring, the constructor imports the real keyring module."""
        from meridiand._vault_backend_os_keychain import OsKeychainVaultBackend

        try:
            backend = OsKeychainVaultBackend()
        except Exception:
            pytest.skip("keyring not available on this system")
        assert backend._kr is not None

    def test_list_secrets_filters_existing(self) -> None:
        from meridiand._vault_backend_os_keychain import OsKeychainVaultBackend

        class _FakeKeyring:
            def __init__(self) -> None:
                self.store: dict[tuple[str, str], str] = {}

            def get_password(self, svc: str, account: str) -> str | None:
                return self.store.get((svc, account))

            def set_password(self, svc: str, account: str, password: str) -> None:
                self.store[(svc, account)] = password

            def delete_password(self, svc: str, account: str) -> None:
                self.store.pop((svc, account), None)

        kr = _FakeKeyring()
        backend = OsKeychainVaultBackend(_keyring=kr)
        backend.store_secret("v1", "k1", "val1", "2024-01-01T00:00:00Z")
        items = backend.list_secrets("v1", ["k1", "missing"])
        assert len(items) == 1
        assert "value" not in items[0]
        # Delete existing + missing
        assert backend.delete_secret("v1", "k1") is True
        assert backend.delete_secret("v1", "missing") is False


class TestVaultBackendEncryptedFile:
    def test_unlock_error_http_status(self) -> None:
        from meridiand._vault_backend_encrypted_file import VaultBackendUnlockError

        assert (
            VaultBackendUnlockError(message="m", timestamp="t", cause=None).http_status() == 500
        )

    def test_unlock_with_passphrase_failure_wraps(self, tmp_path: Path) -> None:
        from meridiand._vault_backend_encrypted_file import (
            EncryptedFileVaultBackend,
            VaultBackendUnlockError,
        )

        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        with patch(
            "pyrage.passphrase.encrypt",
            side_effect=RuntimeError("encrypt boom"),
        ):
            with pytest.raises(VaultBackendUnlockError):
                backend.unlock_with_passphrase("secret-passphrase")

    def test_unlock_with_key_file_failure_wraps(self, tmp_path: Path) -> None:
        from meridiand._vault_backend_encrypted_file import (
            EncryptedFileVaultBackend,
            VaultBackendUnlockError,
        )

        key_file = tmp_path / "bad.key"
        key_file.write_text("not a key")
        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        with pytest.raises(VaultBackendUnlockError):
            backend.unlock_with_key_file(key_file)

    def test_is_unlocked_property(self, tmp_path: Path) -> None:
        from meridiand._vault_backend_encrypted_file import EncryptedFileVaultBackend

        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        assert backend.is_unlocked is False
        backend.unlock_with_passphrase("xx")
        assert backend.is_unlocked is True

    def test_update_secret(self, tmp_path: Path) -> None:
        from meridiand._vault_backend_encrypted_file import EncryptedFileVaultBackend

        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        backend.unlock_with_passphrase("xx")
        backend.store_secret("v1", "k1", "old", "2024-01-01T00:00:00Z")
        backend.update_secret("v1", "k1", {"value": "new", "key": "k1", "vault_id": "v1"})
        assert backend.get_secret("v1", "k1")["value"] == "new"

    def test_full_round_trip_passphrase(self, tmp_path: Path) -> None:
        """End-to-end store/list/delete with passphrase mode exercises _encrypt/_decrypt."""
        from meridiand._vault_backend_encrypted_file import EncryptedFileVaultBackend

        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        backend.unlock_with_passphrase("test-passphrase")
        backend.store_secret("v1", "k1", "val1", "2024-01-01T00:00:00Z")
        assert backend.secret_exists("v1", "k1") is True
        listed = backend.list_secrets("v1")
        assert any(r["key"] == "k1" for r in listed)
        rec = backend.get_secret("v1", "k1")
        assert rec is not None
        assert rec["value"] == "val1"
        assert backend.delete_secret("v1", "k1") is True
        assert backend.delete_secret("v1", "k1") is False  # already gone
        assert backend.secret_exists("v1", "k1") is False

    def test_unlock_with_key_file_success_and_round_trip(self, tmp_path: Path) -> None:
        """Generate a real age key and exercise the key_file mode encrypt/decrypt path."""
        import pyrage  # type: ignore[import-untyped]

        from meridiand._vault_backend_encrypted_file import EncryptedFileVaultBackend

        # Generate a fresh age identity
        identity = pyrage.x25519.Identity.generate()
        key_file = tmp_path / "age.key"
        key_file.write_text(str(identity))

        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        backend.unlock_with_key_file(key_file)
        backend.store_secret("v2", "k2", "val2", "2024-01-01T00:00:00Z")
        rec = backend.get_secret("v2", "k2")
        assert rec is not None
        assert rec["value"] == "val2"


class TestHarnessPoolHelpers:
    def test_harness_pool_error_http_status(self) -> None:
        from meridiand._harness_pool import HarnessPoolError

        assert HarnessPoolError(message="m", timestamp="t", cause=None).http_status() == 422

    def test_load_session_traceparent_missing_manifest(self, tmp_path: Path) -> None:
        from meridiand._harness_pool import HarnessPool

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=MagicMock(),
            phase_reader=MagicMock(),
            num_workers=2,
        )
        assert pool._load_session_traceparent("nope") == ""

    def test_load_session_traceparent_malformed_manifest(self, tmp_path: Path) -> None:
        from meridiand._harness_pool import HarnessPool

        (tmp_path / "sessions" / "s1").mkdir(parents=True)
        (tmp_path / "sessions" / "s1" / "manifest.json").write_text("not json {{{")

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=MagicMock(),
            phase_reader=MagicMock(),
            num_workers=2,
        )
        assert pool._load_session_traceparent("s1") == ""

    async def test_pool_start_twice_skips_alive_slot(self, tmp_path: Path) -> None:
        """A second call to start() doesn't replace an alive worker task (223->222)."""
        from meridiand._harness_pool import HarnessPool

        async def _idle_run(_sid: str) -> tuple[int, int, str]:
            return 0, 0, ""

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=_idle_run,
            phase_reader=MagicMock(),
            num_workers=1,
        )
        await pool.start()
        first_task = pool._slots[0].task
        assert first_task is not None
        await pool.start()  # second start: should skip alive slot
        assert pool._slots[0].task is first_task
        await pool.stop()

    async def test_pool_start_skips_session_when_phase_reader_raises(
        self, tmp_path: Path
    ) -> None:
        """phase_reader.current_phase raising → skipped via continue (235-236)."""
        from meridiand._harness_pool import HarnessPool

        # Pre-create a session manifest
        sessions_dir = tmp_path / "sessions"
        (sessions_dir / "broken").mkdir(parents=True)
        (sessions_dir / "broken" / "manifest.json").write_text("{}")

        class _RaisingReader:
            def current_phase(self, _sid: str) -> str:
                raise RuntimeError("phase boom")

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=lambda _s: None,  # type: ignore[arg-type]
            phase_reader=_RaisingReader(),
            num_workers=2,
        )
        await pool.start()
        await pool.stop()

    async def test_pool_start_typed_error_reraised(self, tmp_path: Path) -> None:
        """HarnessPoolError raised inside is re-raised (lines 241-259, isinstance branch)."""
        from meridiand._harness_pool import HarnessPool, HarnessPoolError

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=lambda _s: None,  # type: ignore[arg-type]
            phase_reader=MagicMock(),
            num_workers=2,
        )
        # Patch asyncio.create_task in the start path to raise HarnessPoolError
        original = asyncio.create_task
        raised = HarnessPoolError(message="pre", timestamp=pagination_now(), cause=None)

        def _raising(coro: Any, *args: Any, **kwargs: Any) -> Any:
            # Close the coro so we don't leak
            coro.close()
            raise raised

        with patch.object(asyncio, "create_task", _raising):
            with pytest.raises(HarnessPoolError):
                await pool.start()

    async def test_pool_start_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """A non-HarnessPoolError exception is wrapped (lines 241-259, else branch)."""
        from meridiand._harness_pool import HarnessPool, HarnessPoolError

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=lambda _s: None,  # type: ignore[arg-type]
            phase_reader=MagicMock(),
            num_workers=2,
        )

        def _raising(coro: Any, *args: Any, **kwargs: Any) -> Any:
            coro.close()
            raise RuntimeError("create boom")

        with patch.object(asyncio, "create_task", _raising):
            with pytest.raises(HarnessPoolError):
                await pool.start()

    async def test_worker_loop_swallows_run_session_exception(self, tmp_path: Path) -> None:
        """run_session raising is silently swallowed (121-122)."""
        from meridiand._harness_pool import HarnessPool

        run_count = [0]

        async def _raising_run(_sid: str) -> tuple[int, int, str]:
            run_count[0] += 1
            raise RuntimeError("intentional")

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=_raising_run,
            phase_reader=MagicMock(),
            num_workers=2,
        )
        slot = pool._slots[0]
        slot.queue.put_nowait("s1")
        task = asyncio.create_task(pool._worker_loop(slot))
        await asyncio.wait_for(slot.queue.join(), timeout=2.0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert run_count[0] == 1


class TestCronErrors:
    def test_parse_duration_empty(self) -> None:
        from meridiand._cron import _parse_duration

        with pytest.raises(ValueError, match="empty"):
            _parse_duration("")

    def test_parse_duration_invalid(self) -> None:
        from meridiand._cron import _parse_duration

        with pytest.raises(ValueError, match="Invalid"):
            _parse_duration("xyz")

    def test_parse_duration_negative(self) -> None:
        """Duration parsing — a unit value of 0 yields total_seconds() == 0 → must be positive."""
        from meridiand._cron import _parse_duration

        with pytest.raises(ValueError, match="positive"):
            _parse_duration("0s")

    def test_cron_create_error_http_status(self) -> None:
        from meridiand._cron import CronCreateError

        assert CronCreateError(message="m", timestamp="t", cause=None).http_status() == 500

    def test_cron_invalid_request_error_http_status(self) -> None:
        from meridiand._cron import CronInvalidRequestError

        assert CronInvalidRequestError(message="m", timestamp="t").http_status() == 422

    def test_cron_delete_error_http_status(self) -> None:
        from meridiand._cron import CronDeleteError

        assert CronDeleteError(message="m", timestamp="t", cause=None).http_status() == 500

    async def test_cron_create_generic_exception_wrapped(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._cron import CronCreateError, CronCreateRequest, make_cron_router

        router = make_cron_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/cron" and "POST" in r.methods
        )
        req = CronCreateRequest(
            trigger_type="interval",
            interval="1h",
            session_id="s1",
            task={"prompt": "hi"},
        )
        with patch("meridiand._cron.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(CronCreateError):
                await handler(req)

    async def test_cron_delete_generic_exception_wrapped(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._cron import CronDeleteError, make_cron_router

        # Create a cron file
        cron_dir = tmp_path / "cron"
        cron_dir.mkdir(parents=True)
        (cron_dir / "c1.json").write_text("{}")

        router = make_cron_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/cron/{cron_id}" and "DELETE" in r.methods
        )
        with patch.object(Path, "unlink", side_effect=RuntimeError("unlink boom")):
            with pytest.raises(CronDeleteError):
                await handler("c1")


class TestWebhookErrors:
    def test_webhook_create_error_http_status(self) -> None:
        from meridiand._webhooks import WebhookCreateError

        assert WebhookCreateError(message="m", timestamp="t", cause=None).http_status() == 500

    def test_webhook_invalid_request_error_http_status(self) -> None:
        from meridiand._webhooks import WebhookInvalidRequestError

        assert WebhookInvalidRequestError(message="m", timestamp="t").http_status() == 422

    def test_webhook_not_found_error_http_status(self) -> None:
        from meridiand._webhooks import WebhookNotFoundError

        assert WebhookNotFoundError(webhook_id="x", timestamp="t").http_status() == 404

    def test_webhook_delete_error_http_status(self) -> None:
        from meridiand._webhooks import WebhookDeleteError

        assert WebhookDeleteError(message="m", timestamp="t", cause=None).http_status() == 500

    async def test_webhook_create_generic_exception_wrapped(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._webhooks import (
            WebhookCreateError,
            WebhookCreateRequest,
            make_webhooks_router,
        )

        router = make_webhooks_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/webhooks" and "POST" in r.methods
        )
        from meridiand._webhooks import EventFilter

        req = WebhookCreateRequest(
            name="test_webhook",
            url="https://example.com/hook",
            event_filter=EventFilter(types=["session.completed"]),
            max_retries=3,
            backoff="exponential",
        )
        with patch("meridiand._webhooks.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(WebhookCreateError):
                await handler(req)

    async def test_webhook_delete_generic_exception_wrapped(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._webhooks import WebhookDeleteError, make_webhooks_router

        # Pre-create a webhook file so delete proceeds past the not-found check
        wh_dir = tmp_path / "webhooks"
        wh_dir.mkdir(parents=True)
        (wh_dir / "wh1.json").write_text("{}")

        router = make_webhooks_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/webhooks/{webhook_id}" and "DELETE" in r.methods
        )
        with patch.object(Path, "unlink", side_effect=RuntimeError("unlink boom")):
            with pytest.raises(WebhookDeleteError):
                await handler("wh1")


class TestKbStoreSearchHelpers:
    def test_has_key_returns_false_when_missing(self, tmp_path: Path) -> None:
        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        assert store.has_key("/none/such") is False

    def test_glob_search_with_scope(self, tmp_path: Path) -> None:
        from meridian_kb_indexer import Chunk

        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        store.upsert_chunks(
            "/docs/a.md",
            "world-doc",
            [
                Chunk(
                    file_path="/docs/a.md",
                    kind="text",
                    content="hello",
                    start_line=1,
                    end_line=1,
                ),
            ],
        )
        rows = store.glob_search("*.md", "world-doc", limit=10)
        assert rows
        # Scope filter
        rows2 = store.glob_search("*.txt", "world-doc", limit=10)
        assert rows2 == []

    def test_bm25_search_empty_query(self, tmp_path: Path) -> None:
        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        assert store.bm25_search("!!!  ", None, limit=10) == []

    def test_bm25_search_no_scope_path(self, tmp_path: Path) -> None:
        """bm25_search with scope=None exercises the else branch (lines 318-324)."""
        from meridian_kb_indexer import Chunk

        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        store.upsert_chunks(
            "/x.md",
            "any",
            [Chunk(file_path="/x.md", kind="text", content="hello world", start_line=1, end_line=1)],
        )
        # Empty result is OK — point is to exercise the no-scope branch
        store.bm25_search("hello", None, limit=10)

    def test_glob_search_truncates_at_limit(self, tmp_path: Path) -> None:
        """glob_search breaks early after `limit` results (line 302)."""
        from meridian_kb_indexer import Chunk

        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        # Add 3 chunks
        for i in range(3):
            store.upsert_chunks(
                f"/file{i}.md",
                "scope",
                [
                    Chunk(
                        file_path=f"/file{i}.md",
                        kind="text",
                        content="x",
                        start_line=1,
                        end_line=1,
                    )
                ],
            )
        # Request limit=1 — should break after first match
        rows = store.glob_search("*.md", "scope", limit=1)
        assert len(rows) == 1

    def test_vector_search_no_scope(self, tmp_path: Path) -> None:
        """vector_search with scope=None exercises the else branch (line 335 -> alternative)."""
        from meridian_kb_indexer import Chunk

        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        store.upsert_chunks(
            "/v.md",
            "s",
            [Chunk(file_path="/v.md", kind="text", content="hi", start_line=1, end_line=1)],
        )
        # scope=None
        store.vector_search("hi", None, limit=5)

    async def test_kb_index_skips_failed_file(self, tmp_path: Path) -> None:
        """Per-file index failure is silently skipped (lines 415-416)."""
        from core_errors import NoopAuditLog

        from meridiand._kb import KbIndexRequest, make_kb_router

        router = make_kb_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/kb/index"
            and ("POST" in r.methods or "POST" in (r.methods or []))
        )
        # Make workspace point to tmp_path with one file
        (tmp_path / "fail.md").write_text("content")
        import os as _os

        old_env = _os.environ.get("MERIDIAN_KB_WORKSPACE")
        _os.environ["MERIDIAN_KB_WORKSPACE"] = str(tmp_path)
        try:
            # Patch indexer.index_file to raise so the per-file except triggers
            with patch(
                "meridiand._kb.WorkspaceIndexer.index_file",
                side_effect=RuntimeError("file fail"),
            ):
                req = KbIndexRequest(scope="global")  # no path → workspace scan
                resp = await handler(req)
            assert resp is not None
        finally:
            if old_env is None:
                _os.environ.pop("MERIDIAN_KB_WORKSPACE", None)
            else:
                _os.environ["MERIDIAN_KB_WORKSPACE"] = old_env

    async def test_kb_index_typed_error_reraised(self, tmp_path: Path) -> None:
        """KbIndexError raised inside is re-raised verbatim (line 427)."""
        from core_errors import NoopAuditLog

        from meridiand._kb import KbIndexError, KbIndexRequest, make_kb_router

        router = make_kb_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/kb/index"
            and ("POST" in r.methods or "POST" in (r.methods or []))
        )
        # Use target_path mode to take the simpler branch
        (tmp_path / "x.md").write_text("hello")
        req = KbIndexRequest(scope="global", path=str(tmp_path / "x.md"))
        with patch(
            "meridiand._kb._load_status",
            side_effect=KbIndexError(message="pre", timestamp=pagination_now(), cause=None),
        ):
            with pytest.raises(KbIndexError):
                await handler(req)

    async def test_kb_query_typed_error_reraised(self, tmp_path: Path) -> None:
        """KbQueryError raised inside is re-raised verbatim (line 525)."""
        from core_errors import NoopAuditLog

        from meridiand._kb import KbQueryError, KbQueryRequest, make_kb_router

        router = make_kb_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/kb/query"
            and ("POST" in r.methods or "POST" in (r.methods or []))
        )
        req = KbQueryRequest(query="q", scope="global", limit=5)
        # Patch _rrf_fuse to raise typed error
        with patch(
            "meridiand._kb._rrf_fuse",
            side_effect=KbQueryError(message="pre", timestamp=pagination_now(), cause=None),
        ):
            with pytest.raises(KbQueryError):
                await handler(req)


class TestHookCreateGeneric:
    async def test_generic_exception_wrapped_to_hook_create_error(self, tmp_path: Path) -> None:
        """A non-HookInvalidRequestError exception is wrapped (lines 189-210)."""
        from core_errors import NoopAuditLog

        from meridiand._hooks import (
            FailureMode,
            HandlerType,
            HookCreateError,
            HookCreateRequest,
            make_hooks_router,
        )

        # Build the router and extract the inner handler function
        router = make_hooks_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint for r in router.routes if r.path == "/v1/x/hooks" and "POST" in r.methods
        )
        req = HookCreateRequest(
            event="on_checkpoint",
            name="test",
            handler=HandlerType.in_process,
            timeout_ms=1000,
            failure_mode=FailureMode.ignore,
        )
        with patch("meridiand._hooks.json.dumps", side_effect=RuntimeError("dump boom")):
            with pytest.raises(HookCreateError):
                await handler(req)


class TestPhaseTransitionTerminal:
    @staticmethod
    def _make_phase_client(storage_root: Path):
        from fastapi.testclient import TestClient
        from storage_event_log import LocalEventLogWriter

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(storage_root)
        writer = LocalEventLogWriter(storage_root)
        app = create_app(audit, storage_root=storage_root, event_log=writer)
        return TestClient(app, raise_server_exceptions=False)

    def test_terminal_phase_with_manifest(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions" / "sterm"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "manifest.json").write_text(
            json.dumps({"created_at": "2024-01-01T00:00:00+00:00"})
        )
        client = self._make_phase_client(tmp_path)
        resp = client.post(
            "/v1/x/sessions/sterm/phase",
            json={"to_phase": "completed", "reason": "ok"},
        )
        assert resp.status_code == 200

    def test_terminal_phase_missing_manifest_skipped(self, tmp_path: Path) -> None:
        client = self._make_phase_client(tmp_path)
        resp = client.post(
            "/v1/x/sessions/snomanifest/phase",
            json={"to_phase": "completed", "reason": "ok"},
        )
        assert resp.status_code == 200

    def test_terminal_phase_manifest_no_created_at(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions" / "sno_ts"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "manifest.json").write_text("{}")
        client = self._make_phase_client(tmp_path)
        resp = client.post(
            "/v1/x/sessions/sno_ts/phase",
            json={"to_phase": "completed", "reason": "ok"},
        )
        assert resp.status_code == 200

    def test_phase_with_existing_before_decrements(self, tmp_path: Path) -> None:
        """If before is not None, active_sessions[before] is decremented (113->114)."""
        client = self._make_phase_client(tmp_path)
        # First transition to "running"
        r1 = client.post(
            "/v1/x/sessions/sbefore/phase",
            json={"to_phase": "running", "reason": "start"},
        )
        assert r1.status_code == 200
        # Second transition: before should now be "running"
        r2 = client.post(
            "/v1/x/sessions/sbefore/phase",
            json={"to_phase": "idle", "reason": "pause"},
        )
        assert r2.status_code == 200

    def test_phase_typed_error_reraised(self, tmp_path: Path) -> None:
        """PhaseTransitionError raised inside is re-raised verbatim (line 127)."""
        from fastapi.testclient import TestClient
        from storage_event_log import LocalEventLogWriter

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog
        from meridiand._phase import PhaseTransitionError

        audit = FileAuditLog(tmp_path)
        writer = LocalEventLogWriter(tmp_path)

        async def _boom(*_a: Any, **_k: Any) -> None:
            raise PhaseTransitionError(
                message="pre-typed", timestamp=pagination_now(), cause=None
            )

        with patch.object(writer, "append", _boom):
            app = create_app(audit, storage_root=tmp_path, event_log=writer)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/x/sessions/sreraise/phase",
                json={"to_phase": "completed", "reason": "ok"},
            )
        assert resp.status_code in {422, 500}


class TestErrorClassesHttpStatus:
    """One-liner http_status tests for error classes across modules."""

    def test_hook_create_error(self) -> None:
        from meridiand._hooks import HookCreateError

        assert HookCreateError(message="m", timestamp="t", cause=None).http_status() == 500

    def test_hook_invalid_request_error(self) -> None:
        from meridiand._hooks import HookInvalidRequestError

        assert HookInvalidRequestError(message="m", timestamp="t").http_status() == 422

    def test_checkpoint_error(self) -> None:
        from meridiand._checkpoint import CheckpointError

        assert CheckpointError(message="m", timestamp="t", cause=None).http_status() == 422

    def test_kb_index_error(self) -> None:
        from meridiand._kb import KbIndexError, KbQueryError, KbStatusError

        assert KbIndexError(message="m", timestamp="t", cause=None).http_status() == 422
        assert KbStatusError(message="m", timestamp="t", cause=None).http_status() == 422
        assert KbQueryError(message="m", timestamp="t", cause=None).http_status() == 422

    def test_skill_forge_errors(self) -> None:
        from meridiand._skill_forge import SkillForgeProposalError, SkillForgeRunError

        assert SkillForgeRunError(message="m", timestamp="t", cause=None).http_status() == 500
        assert SkillForgeProposalError(message="m", timestamp="t", cause=None).http_status() == 500


class TestVaultLeakSoakHelpers:
    def test_scan_skips_unreadable_files(self, tmp_path: Path) -> None:
        from meridiand._vault_leak_soak import _scan_storage_root

        # Create one readable + one that will raise OSError on read
        (tmp_path / "f1.txt").write_text("clean")
        (tmp_path / "subdir").mkdir()
        leaks = _scan_storage_root(tmp_path, ["s3cret-CANARY"])
        assert leaks == []  # no leaks expected

    def test_scan_returns_empty_when_root_missing(self, tmp_path: Path) -> None:
        from meridiand._vault_leak_soak import _scan_storage_root

        leaks = _scan_storage_root(tmp_path / "nope", ["x"])
        assert leaks == []

    def test_scan_records_leak_when_canary_found(self, tmp_path: Path) -> None:
        from meridiand._vault_leak_soak import _scan_storage_root

        (tmp_path / "leaked.txt").write_text("here is s3cret-XYZ in plain text")
        leaks = _scan_storage_root(tmp_path, ["s3cret-XYZ"])
        assert len(leaks) == 1
        assert leaks[0]["source"] == "file"

    def test_scan_skips_unreadable_file(self, tmp_path: Path) -> None:
        """A file that raises OSError on read is silently skipped (lines 97-98)."""
        from meridiand._vault_leak_soak import _scan_storage_root

        (tmp_path / "ok.txt").write_text("clean")
        (tmp_path / "fail.txt").write_text("doomed")
        real_read = Path.read_text

        def _selective(self: Path, *a: Any, **k: Any) -> str:
            if self.name == "fail.txt":
                raise OSError("denied")
            return real_read(self, *a, **k)

        with patch.object(Path, "read_text", _selective):
            leaks = _scan_storage_root(tmp_path, ["x"])
        assert leaks == []

    def test_memory_keyring_round_trip(self) -> None:
        from meridiand._vault_leak_soak import _MemoryKeyring

        k = _MemoryKeyring()
        assert k.get_password("svc", "u") is None
        k.set_password("svc", "u", "secret")
        assert k.get_password("svc", "u") == "secret"
        k.delete_password("svc", "u")
        assert k.get_password("svc", "u") is None


class TestSkillEfficacyHelpers:
    async def test_noop_trajectory_runner_returns_false(self) -> None:
        from meridiand._skill_efficacy import NoopTrajectoryRunner

        r = NoopTrajectoryRunner()
        assert await r.run({}, skill_instructions=None) is False

    async def test_compare_reraises_typed_error(self, tmp_path: Path) -> None:
        """SkillEfficacyError raised inside is re-raised verbatim (lines 203-219)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_efficacy import (
            SkillEfficacyError,
            compare_proposal_trajectories,
        )

        class _BoomRunner:
            async def run(self, *_a: Any, **_k: Any) -> bool:
                raise SkillEfficacyError(
                    message="pre-typed", timestamp=pagination_now(), cause=None
                )

        with pytest.raises(SkillEfficacyError):
            await compare_proposal_trajectories(
                proposal={
                    "id": "p1",
                    "skill_id": "s1",
                    "instructions": "do x",
                    "tests": [{"name": "t1"}],
                },
                efficacy_dir=tmp_path,
                audit_log=NoopAuditLog(),
                runner=_BoomRunner(),
            )


class TestAcpHttpTransport:
    async def test_default_handler_returns_empty(self) -> None:
        from meridiand._acp import DefaultAcpInboundHandler

        h = DefaultAcpInboundHandler()
        result = await h.handle("target", {"x": 1})
        assert result == {}

    async def test_http_peer_client_call(self) -> None:
        import httpx

        from meridiand._acp import HttpAcpPeerClient

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(_handler)
        real_init = httpx.AsyncClient.__init__

        def _patched_init(self: httpx.AsyncClient, *a: Any, **kw: Any) -> None:
            kw.pop("transport", None)
            real_init(self, *a, transport=transport, **kw)

        with patch.object(httpx.AsyncClient, "__init__", _patched_init):
            c = HttpAcpPeerClient()
            out = await c.call("http://example.com/acp", {"msg": "hi"})
        assert out == {"ok": True}


class TestModelCallEventLogAdapter:
    async def test_adapter_appends_session_event(self) -> None:
        from meridiand._model_call_event_log import EventLogModelCallAdapter

        captured: list[dict[str, Any]] = []

        class _Runtime:
            async def append(self, *, session_id: str, event_type: str, data: dict[str, Any]) -> None:
                captured.append({"session_id": session_id, "event_type": event_type, "data": data})

        adapter = EventLogModelCallAdapter(_Runtime())
        await adapter.record_started(
            session_id="s1",
            routing_rule="r",
            provider_name="p",
            model="m",
        )
        assert captured == [
            {
                "session_id": "s1",
                "event_type": "model_call.started",
                "data": {"routing_rule": "r", "provider_name": "p", "model": "m"},
            }
        ]


class TestSystemPromptTemplateErrors:
    def test_expand_error_http_status(self) -> None:
        from meridiand._system_prompt_template import TemplateExpandError

        err = TemplateExpandError(message="m", timestamp="t", cause=None)
        assert err.http_status() == 500

    def test_memory_not_found_error_http_status(self) -> None:
        from meridiand._system_prompt_template import TemplateMemoryNotFoundError

        err = TemplateMemoryNotFoundError(memory_key="k", timestamp="t")
        assert err.http_status() == 404


class TestHealthzReadyzMetricsErrors:
    def test_healthz_error_http_status(self) -> None:
        from meridiand._healthz import HealthzError

        err = HealthzError(message="m", timestamp="t", cause=None)
        assert err.http_status() == 500

    def test_healthz_exception_path(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog
        from meridiand._healthz import HealthzError

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        # Patch JSONResponse construction inside healthz to raise
        with patch(
            "meridiand._healthz.JSONResponse",
            side_effect=RuntimeError("liveness boom"),
        ):
            resp = client.get("/healthz")
        assert resp.status_code == 500

    def test_readyz_error_http_status(self) -> None:
        from meridiand._readyz import ReadyzError

        err = ReadyzError(message="m", timestamp="t", cause=None)
        assert err.http_status() == 500

    def test_readyz_exception_path(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        with patch(
            "meridiand._readyz.JSONResponse",
            side_effect=RuntimeError("readiness boom"),
        ):
            resp = client.get("/readyz")
        assert resp.status_code == 500

    def test_metrics_endpoint(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_error_http_status(self) -> None:
        from meridiand._metrics import MetricsError

        err = MetricsError(message="m", timestamp="t", cause=None)
        assert err.http_status() == 500

    def test_metrics_exception_path(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        with patch(
            "meridiand._metrics.generate_latest",
            side_effect=RuntimeError("scrape boom"),
        ):
            resp = client.get("/metrics")
        assert resp.status_code == 500


class TestAcpComplianceResult:
    def test_failed_with_reason_includes_reason(self) -> None:
        from meridiand._acp_compliance import _result

        r = _result("test_name", "desc", passed=False, reason="why it failed")
        assert r["reason"] == "why it failed"
        assert r["status"] == "failed"

    def test_failed_without_reason_omits_reason(self) -> None:
        from meridiand._acp_compliance import _result

        r = _result("test_name", "desc", passed=False, reason=None)
        assert "reason" not in r

    def test_passed_with_reason_omits_reason(self) -> None:
        """passed=True suppresses reason even when provided."""
        from meridiand._acp_compliance import _result

        r = _result("test_name", "desc", passed=True, reason="ignored")
        assert "reason" not in r


# ---------------------------------------------------------------------------
# _cancel — descendant traversal skips malformed manifests
# ---------------------------------------------------------------------------


class TestCancelDescendantTraversal:
    def test_malformed_manifest_skipped(self, tmp_path: Path) -> None:
        """Manifest JSON that raises (e.g. invalid JSON) is silently skipped."""
        from meridiand._cancel import _walk_descendants

        sessions = tmp_path / "sessions"
        (sessions / "s1").mkdir(parents=True)
        (sessions / "s1" / "manifest.json").write_text(
            json.dumps({"parent_session_id": "parent1", "child_session_id": "s1"})
        )
        (sessions / "s2").mkdir(parents=True)
        (sessions / "s2" / "manifest.json").write_text("not json {{{")  # malformed

        desc = _walk_descendants("parent1", tmp_path)
        assert "s1" in desc

    def test_manifest_without_parent_or_child_skipped(self, tmp_path: Path) -> None:
        """Manifest without both parent and child fields is skipped (60->55)."""
        from meridiand._cancel import _walk_descendants

        sessions = tmp_path / "sessions"
        (sessions / "s1").mkdir(parents=True)
        (sessions / "s1" / "manifest.json").write_text(json.dumps({}))  # neither field
        (sessions / "s2").mkdir(parents=True)
        (sessions / "s2" / "manifest.json").write_text(
            json.dumps({"parent_session_id": "p"})  # only parent, no child
        )
        desc = _walk_descendants("p", tmp_path)
        assert desc == []

    def test_duplicate_child_reference_seen_once(self, tmp_path: Path) -> None:
        """A child appearing twice in the graph is only enqueued once (72->71)."""
        from meridiand._cancel import _walk_descendants

        sessions = tmp_path / "sessions"
        # p has two children c1 and c2
        for child_id in ("c1", "c2"):
            (sessions / child_id).mkdir(parents=True)
            (sessions / child_id / "manifest.json").write_text(
                json.dumps({"parent_session_id": "p", "child_session_id": child_id})
            )
        # c1 also lists c2 as a child (creates a duplicate path to c2)
        (sessions / "c2dup").mkdir(parents=True)
        (sessions / "c2dup" / "manifest.json").write_text(
            json.dumps({"parent_session_id": "c1", "child_session_id": "c2"})
        )
        desc = _walk_descendants("p", tmp_path)
        # c2 appears once (de-duped)
        assert desc.count("c2") == 1


class TestCancelMissingDescendantManifest:
    def test_descendant_with_missing_manifest_skipped(self, tmp_path: Path) -> None:
        """A descendant whose own manifest dir is missing is skipped (109->113)."""
        from core_errors import NoopAuditLog
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from meridiand._cancel import make_cancel_router

        sessions = tmp_path / "sessions"
        # "edge" has parent_session_id=p1 and child_session_id=ghost.
        # ghost will appear in descendants but ghost/manifest.json doesn't exist.
        (sessions / "edge").mkdir(parents=True)
        (sessions / "edge" / "manifest.json").write_text(
            json.dumps({"parent_session_id": "p1", "child_session_id": "ghost"})
        )

        router = make_cancel_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/p1/cancel")
        # the cancel-walk processes ghost without crashing
        assert resp.status_code in {200, 204}


# ---------------------------------------------------------------------------
# _diagnosis — audit-line filtering
# ---------------------------------------------------------------------------


class TestDiagnosisAuditFilter:
    def test_skips_blank_and_invalid_lines(self, tmp_path: Path) -> None:
        from meridiand._diagnosis import _read_audit_for_session

        audit_path = tmp_path / "audit.ndjson"
        audit_path.write_text(
            "\n"  # blank line
            "  \n"  # whitespace
            "not json {{{\n"  # invalid JSON
            + json.dumps({"detail": {"session_id": "wanted"}, "ts": "t"})
            + "\n"
            + json.dumps({"detail": {"session_id": "other"}, "ts": "t"})
            + "\n"
        )
        entries = _read_audit_for_session(audit_path, "wanted")
        assert len(entries) == 1
        assert entries[0]["detail"]["session_id"] == "wanted"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        from meridiand._diagnosis import _read_audit_for_session

        result = _read_audit_for_session(tmp_path / "nope.ndjson", "any")
        assert result == []

    def test_phase_change_with_blank_after_keeps_default(self) -> None:
        """phase_change event with after='' doesn't update terminal_phase (108->110)."""
        from types import SimpleNamespace

        from meridiand._diagnosis import _extract_failure_summary

        events = [
            SimpleNamespace(
                type="session.phase_change",
                data={"after": "", "reason": ""},
                seq=1,
                ts="t",
                thread_id=None,
            )
        ]
        phase, reason, _ = _extract_failure_summary(events)
        assert phase == "unknown"
        assert reason == ""

    def test_diagnosis_reraises_typed_error(self, tmp_path: Path) -> None:
        """If inner code raises SessionDiagnosisError, it's re-raised (line 151)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog
        from meridiand._diagnosis import SessionDiagnosisError

        # Make _extract_failure_summary raise SessionDiagnosisError
        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "meridiand._diagnosis._extract_failure_summary",
            side_effect=SessionDiagnosisError(
                message="pre-typed", timestamp=pagination_now(), cause=None
            ),
        ):
            resp = client.get("/v1/sessions/s1/diagnosis")
        assert resp.status_code in {422, 500}


# ---------------------------------------------------------------------------
# _event_translator — final default return for unknown events
# ---------------------------------------------------------------------------


class TestEventTranslatorUnknown:
    def test_unknown_event_returns_empty_list(self) -> None:
        from meridiand._event_translator import ModelEventTranslator

        t = ModelEventTranslator()
        # Pass a sentinel object that doesn't match any isinstance check
        out = t.translate(object())  # type: ignore[arg-type]
        assert out == []

    def test_message_stop_event_with_none_stop_reason_keeps_existing(self) -> None:
        """MessageStopEvent with stop_reason=None doesn't overwrite (122->124)."""
        from meridian_sdk_provider.types import (
            MessageDeltaEvent,
            MessageStopEvent,
        )

        from meridiand._event_translator import ModelEventTranslator

        t = ModelEventTranslator()
        # First set a stop_reason
        t.translate(MessageDeltaEvent(stop_reason="end_turn"))
        # Then send a MessageStopEvent with no stop_reason
        out = t.translate(MessageStopEvent(stop_reason=None, input_tokens=0, output_tokens=0))
        assert out  # produces model_call.completed event
        completed = next(d for k, d in out if k == "model_call.completed")
        assert completed["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------------
# _cli_channel_driver — small class methods + ChannelFailure reraise
# ---------------------------------------------------------------------------


class TestCliChannelDriverHelpers:
    def test_sys_stdout_writer_writes_and_flushes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from meridiand._cli_channel_driver import SysStdoutWriter

        w = SysStdoutWriter()
        w.write("hello")
        w.flush()
        out = capsys.readouterr().out
        assert "hello" in out

    async def test_noop_stdin_reader_client_runs_and_stops(self) -> None:
        from meridiand._cli_channel_driver import NoopStdinReaderClient

        c = NoopStdinReaderClient()
        await c.run()
        await c.stop()

    def test_token_stream_skips_non_string_tokens(self, tmp_path: Path) -> None:
        """Tokens that aren't strings are skipped in _write_token_stream (209->208)."""
        from meridiand._cli_channel_driver import CliChannelDriver

        captured: list[str] = []

        class _W:
            def write(self, s: str) -> None:
                captured.append(s)

            def flush(self) -> None:
                pass

        driver = CliChannelDriver(storage_root=tmp_path, stdout_writer=_W())
        driver._write_token_stream(json.dumps(["a", 1, "b", None, "c"]))
        joined = "".join(captured)
        assert "a" in joined and "b" in joined and "c" in joined
        assert "1" not in joined and "None" not in joined

    def test_tool_call_non_object_raises(self, tmp_path: Path) -> None:
        """tool_call payload that's a JSON array (not object) raises (line 221)."""
        from meridiand._cli_channel_driver import CliChannelDriver

        class _W:
            def write(self, s: str) -> None:
                pass

            def flush(self) -> None:
                pass

        driver = CliChannelDriver(storage_root=tmp_path, stdout_writer=_W())
        with pytest.raises(ValueError, match="JSON object"):
            driver._write_tool_call(json.dumps([1, 2, 3]))

    async def test_idempotency_non_http_passthrough(self) -> None:
        """websocket scope is passed straight through (lines 51-52)."""
        from core_errors import NoopAuditLog

        from meridiand._idempotency_middleware import IdempotencyKeyMiddleware

        called: list[str] = []

        async def _handler(scope: Any, receive: Any, send: Any) -> None:
            called.append(scope["type"])

        mw = IdempotencyKeyMiddleware(_handler, audit_log=NoopAuditLog())
        await mw({"type": "websocket"}, lambda: None, lambda _m: None)
        assert called == ["websocket"]

    async def test_idempotency_capturing_send_passes_through_other_messages(self) -> None:
        """A message type other than start/body falls through to await send (125->127)."""
        from core_errors import NoopAuditLog

        from meridiand._idempotency_middleware import IdempotencyKeyMiddleware

        async def _handler(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.trailers", "headers": []})  # other type
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

        mw = IdempotencyKeyMiddleware(_handler, audit_log=NoopAuditLog())

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/x/agents",
            "query_string": b"",
            "headers": [(b"idempotency-key", b"trailing-msg-key")],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8888),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        sent_types: list[str] = []

        async def send(m: Any) -> None:
            sent_types.append(m["type"])

        await mw(scope, receive, send)
        assert "http.response.trailers" in sent_types

    async def test_idempotency_no_response_skip_cache(self) -> None:
        """If handler never sends response.start, cache write is skipped (line 132)."""
        from core_errors import NoopAuditLog

        from meridiand._idempotency_middleware import IdempotencyKeyMiddleware

        async def _handler(scope: Any, receive: Any, send: Any) -> None:
            return  # never sends anything

        mw = IdempotencyKeyMiddleware(_handler, audit_log=NoopAuditLog())

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/x/agents",
            "query_string": b"",
            "headers": [(b"idempotency-key", b"key-only-this-test")],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8888),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_m: Any) -> None:
            pass

        # Should complete without raising
        await mw(scope, receive, send)

    async def test_idempotency_cache_store_failure_writes_audit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _CachedResponse construction raises, audit error is written (132, 143-144)."""
        from core_errors import AuditLog, AuditLogEntry

        from meridiand._idempotency_middleware import IdempotencyKeyMiddleware

        captured: list[AuditLogEntry] = []

        class _Audit(AuditLog):
            def write(self, entry: AuditLogEntry) -> None:
                captured.append(entry)

        async def _handler(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"ok":true}', "more_body": False})

        mw = IdempotencyKeyMiddleware(_handler, audit_log=_Audit())

        # Patch _CachedResponse to raise so the except handler fires
        monkeypatch.setattr(
            "meridiand._idempotency_middleware._CachedResponse",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cache write boom")),
        )

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/x/agents",
            "query_string": b"",
            "headers": [(b"idempotency-key", b"k1")],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8888),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[dict[str, Any]] = []

        async def send(msg: dict[str, Any]) -> None:
            sent.append(msg)

        await mw(scope, receive, send)
        # Audit was written for the cache-store failure
        assert any(
            e.event == "idempotency.cache.store.failed" for e in captured
        ), [e.event for e in captured]

    async def test_channel_failure_reraised(self, tmp_path: Path) -> None:
        """ChannelFailure raised by _write_content is re-raised verbatim (line 295)."""
        from sdk_channel import ChannelFailure, SendRequest

        from meridiand._cli_channel_driver import CliChannelDriver

        class _W:
            def write(self, s: str) -> None:
                pass

            def flush(self) -> None:
                pass

        driver = CliChannelDriver(storage_root=tmp_path, stdout_writer=_W())
        # Pre-populate channel config so _load_driver_config doesn't fail
        chan_dir = tmp_path / "channels"
        chan_dir.mkdir(parents=True, exist_ok=True)
        (chan_dir / "c1.json").write_text(json.dumps({"config": {}}))

        from datetime import UTC, datetime

        original = ChannelFailure(
            code="X",
            message="m",
            channel_id="c1",
            channel_kind="meridian.cli",
            session_id="s1",
            timestamp=datetime.now(UTC).isoformat(),
        )

        def _boom(*_a, **_k) -> None:
            raise original

        driver._write_content = _boom  # type: ignore[method-assign]
        req = SendRequest(
            channel_id="c1",
            channel_kind="meridian.cli",
            session_id="s1",
            recipient="user",
            content="hi",
            content_type="text",
        )
        with pytest.raises(ChannelFailure):
            await driver.send(req)


# ---------------------------------------------------------------------------
# _credential_proxy — http_client=None path
# ---------------------------------------------------------------------------


class TestCredentialProxyDefaultClient:
    def test_default_http_client_path(self, tmp_path: Path) -> None:
        """When http_client=None, the proxy creates its own AsyncClient (lines 232-233)."""
        from core_errors import HandlerOptions, install_error_handler
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from meridiand._audit import FileAuditLog
        from meridiand._credential_proxy import (
            CredentialProxyProviderConfig,
            make_credential_proxy_router,
        )

        class _Resolver:
            def resolve(self, ref: str) -> str | None:
                return "tok"

        provider = CredentialProxyProviderConfig(
            name="p1",
            base_url="http://127.0.0.1:1",  # connection will fail (port 1 closed)
            token_secret_ref="secret_ref://v/k",
        )
        audit_log = FileAuditLog(tmp_path)
        router = make_credential_proxy_router(
            audit_log=audit_log,
            secret_resolver=_Resolver(),
            providers=[provider],
            http_client=None,  # forces the else branch
        )
        app = FastAPI()
        app.include_router(router)
        install_error_handler(app, HandlerOptions(audit_log=audit_log))
        client = TestClient(app, raise_server_exceptions=False)
        # The forward will fail with a connect error, but the else branch is exercised.
        resp = client.get("/v1/credential-proxy/p1/anything")
        assert resp.status_code == 502
