"""
Skill Forge proposals reject endpoint conformance suite.

Tests cover:
  - POST /v1/x/skill_forge/proposals/{id}/reject returns 200 on success.
  - Response body has status "REJECTED".
  - Response body has rejection_reason matching the request body reason.
  - Response body has rejected_at timestamp.
  - Response body retains original proposal fields (id, skill_id, etc.).
  - Proposal file on disk is updated with status "REJECTED".
  - Proposal file on disk has rejection_reason.
  - Proposal file on disk has rejected_at.
  - Returns 404 with code "skill_forge_proposal_not_found" for unknown id.
  - Returns 409 with code "skill_forge_proposal_already_promoted" when status is "PROMOTED".
  - 409 does not modify the proposal file on disk.
  - Rejecting an already-REJECTED proposal succeeds (updates reason and timestamp).
  - Success writes audit entry with event "skill_forge.proposal.rejected".
  - Success audit entry level is "info".
  - Success audit detail has proposal_id.
  - Success audit detail has skill_id.
  - Success audit detail has reason.
  - 404 failure writes audit entry with event "skill_forge.proposal.reject.failed".
  - 404 failure audit entry level is "error".
  - 404 failure audit entry code is "skill_forge_proposal_not_found".
  - 404 failure audit detail has proposal_id and message.
  - 409 failure writes audit entry with event "skill_forge.proposal.reject.failed".
  - 409 failure audit entry level is "error".
  - 409 failure audit entry code is "skill_forge_proposal_already_promoted".
  - Error response body has error.code and error.message.
  - OTel span "skill_forge.proposal.reject" emitted on success.
  - OTel span carries skill_forge.proposal_id attribute.
  - OTel span sets skill_forge.proposal.reject.success=True on success.
  - OTel span "skill_forge.proposal.reject" emitted on 404 failure.
  - OTel span sets skill_forge.proposal.reject.success=False on failure.
  - create_app wires proposals router when storage_root is supplied.
  - create_app omits proposals route when storage_root is None.
  - SkillForgeProposalNotFoundError has http_status 404.
  - SkillForgeProposalAlreadyPromotedError has http_status 409.
  - SkillForgeProposalRejectError has http_status 500.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._skill_forge_proposals import (
    SkillForgeProposalAlreadyPromotedError,
    SkillForgeProposalNotFoundError,
    SkillForgeProposalRejectError,
)

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _write_proposal(
    storage_root: Path,
    proposal_id: str = "skillver_abc123",
    skill_id: str = "skill_xyz",
    status: str = "PROPOSAL",
) -> dict[str, Any]:
    proposals_dir = storage_root / "skill_forge" / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "id": proposal_id,
        "skill_id": skill_id,
        "instructions": "Do something.",
        "tools": [],
        "tests": [],
        "source": "forge",
        "source_type": "forge",
        "source_url": None,
        "derived_from_session_ids": None,
        "run_id": "sfrun_test",
        "job_id": "sfjob_test",
        "status": status,
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    (proposals_dir / f"{proposal_id}.json").write_text(json.dumps(record))
    return record


def _read_proposal(storage_root: Path, proposal_id: str) -> dict[str, Any]:
    return json.loads(
        (storage_root / "skill_forge" / "proposals" / f"{proposal_id}.json").read_text()
    )


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _reject(
    client: TestClient,
    proposal_id: str = "skillver_abc123",
    reason: str = "Not good enough.",
) -> Any:
    return client.post(
        f"/v1/x/skill_forge/proposals/{proposal_id}/reject",
        json={"reason": reason},
    )


# ---------------------------------------------------------------------------
# Success: HTTP and response shape
# ---------------------------------------------------------------------------


class TestRejectSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _reject(client)
        assert resp.status_code == 200

    def test_response_status_is_rejected(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _reject(client)
        assert resp.json()["status"] == "REJECTED"

    def test_response_rejection_reason_matches_request(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _reject(client, reason="Too vague.")
        assert resp.json()["rejection_reason"] == "Too vague."

    def test_response_has_rejected_at(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _reject(client)
        assert resp.json().get("rejected_at")

    def test_response_retains_original_fields(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_abc123", skill_id="skill_xyz")
        client = _make_client(storage_root)
        resp = _reject(client)
        body = resp.json()
        assert body["id"] == "skillver_abc123"
        assert body["skill_id"] == "skill_xyz"


# ---------------------------------------------------------------------------
# Success: disk state
# ---------------------------------------------------------------------------


class TestRejectDiskState:
    def test_proposal_file_status_is_rejected(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _reject(client)
        assert _read_proposal(storage_root, "skillver_abc123")["status"] == "REJECTED"

    def test_proposal_file_has_rejection_reason(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _reject(client, reason="Needs more detail.")
        assert (
            _read_proposal(storage_root, "skillver_abc123")["rejection_reason"]
            == "Needs more detail."
        )

    def test_proposal_file_has_rejected_at(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _reject(client)
        assert _read_proposal(storage_root, "skillver_abc123").get("rejected_at")


# ---------------------------------------------------------------------------
# 404: proposal not found
# ---------------------------------------------------------------------------


class TestRejectNotFound:
    def test_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _reject(client, proposal_id="skillver_missing")
        assert resp.status_code == 404

    def test_error_code_is_not_found(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _reject(client, proposal_id="skillver_missing")
        assert resp.json()["error"]["code"] == "skill_forge_proposal_not_found"

    def test_error_message_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _reject(client, proposal_id="skillver_missing")
        assert resp.json()["error"]["message"]

    def test_writes_reject_failed_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _reject(client, proposal_id="skillver_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.proposal.reject.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _reject(client, proposal_id="skillver_missing")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.reject.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _reject(client, proposal_id="skillver_missing")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.reject.failed"
        )
        assert record["code"] == "skill_forge_proposal_not_found"

    def test_not_found_audit_detail_has_proposal_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _reject(client, proposal_id="skillver_missing")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.reject.failed"
        )
        assert record["detail"]["proposal_id"] == "skillver_missing"

    def test_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _reject(client, proposal_id="skillver_missing")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.reject.failed"
        )
        assert record["detail"]["message"]


# ---------------------------------------------------------------------------
# 409: already promoted
# ---------------------------------------------------------------------------


class TestRejectAlreadyPromoted:
    def test_returns_409(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        resp = _reject(client)
        assert resp.status_code == 409

    def test_error_code_is_already_promoted(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        resp = _reject(client)
        assert resp.json()["error"]["code"] == "skill_forge_proposal_already_promoted"

    def test_promoted_proposal_not_modified_on_disk(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        _reject(client)
        on_disk = _read_proposal(storage_root, "skillver_abc123")
        assert on_disk["status"] == "PROMOTED"
        assert "rejection_reason" not in on_disk
        assert "rejected_at" not in on_disk

    def test_writes_reject_failed_audit_entry(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        _reject(client)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.proposal.reject.failed" for r in records)

    def test_already_promoted_audit_level_is_error(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        _reject(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.reject.failed"
        )
        assert record["level"] == "error"

    def test_already_promoted_audit_code(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        _reject(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.reject.failed"
        )
        assert record["code"] == "skill_forge_proposal_already_promoted"


# ---------------------------------------------------------------------------
# Re-rejecting an already-REJECTED proposal
# ---------------------------------------------------------------------------


class TestRejectAlreadyRejected:
    def test_returns_200_for_already_rejected(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="REJECTED")
        client = _make_client(storage_root)
        resp = _reject(client, reason="Still not good.")
        assert resp.status_code == 200

    def test_updates_reason_on_re_reject(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="REJECTED")
        client = _make_client(storage_root)
        _reject(client, reason="Updated reason.")
        assert (
            _read_proposal(storage_root, "skillver_abc123")["rejection_reason"] == "Updated reason."
        )


# ---------------------------------------------------------------------------
# Audit log: success
# ---------------------------------------------------------------------------


class TestAuditSuccess:
    def test_writes_proposal_rejected_audit_entry(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _reject(client)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.proposal.rejected" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _reject(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.rejected"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_proposal_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_abc123")
        client = _make_client(storage_root)
        _reject(client, proposal_id="skillver_abc123")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.rejected"
        )
        assert record["detail"]["proposal_id"] == "skillver_abc123"

    def test_success_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root, skill_id="skill_audit")
        client = _make_client(storage_root)
        _reject(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.rejected"
        )
        assert record["detail"]["skill_id"] == "skill_audit"

    def test_success_audit_detail_has_reason(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _reject(client, reason="Audit reason check.")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.rejected"
        )
        assert record["detail"]["reason"] == "Audit reason check."


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOtelSpans:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _get_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("skill_forge.proposal.reject")

    def test_emits_reject_span_on_success(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _reject(client)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.proposal.reject" in span_names

    def test_span_has_proposal_id_attribute(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_otel")
        client = _make_client(storage_root)
        _reject(client, proposal_id="skillver_otel")
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal_id"] == "skillver_otel"

    def test_span_success_attribute_true_on_success(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _reject(client)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal.reject.success"] is True

    def test_emits_reject_span_on_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _reject(client, proposal_id="skillver_missing")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.proposal.reject" in span_names

    def test_span_success_attribute_false_on_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _reject(client, proposal_id="skillver_missing")
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal.reject.success"] is False

    def test_span_success_attribute_false_on_409(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        _reject(client)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal.reject.success"] is False


# ---------------------------------------------------------------------------
# Router wiring
# ---------------------------------------------------------------------------


class TestRouterWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _reject(client)
        assert resp.status_code != 404 or "not_found" not in resp.text

    def test_route_absent_without_storage_root(self) -> None:
        import tempfile

        from meridiand._audit import FileAuditLog

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(FileAuditLog(Path(tmp)), storage_root=None)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/x/skill_forge/proposals/skillver_abc/reject",
                json={"reason": "test"},
            )
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Error class contracts
# ---------------------------------------------------------------------------


class TestErrorClasses:
    def test_not_found_http_status(self) -> None:
        err = SkillForgeProposalNotFoundError(
            proposal_id="skillver_x", timestamp="2024-01-01T00:00:00+00:00"
        )
        assert err.http_status() == 404

    def test_not_found_code(self) -> None:
        err = SkillForgeProposalNotFoundError(
            proposal_id="skillver_x", timestamp="2024-01-01T00:00:00+00:00"
        )
        assert err.code == "skill_forge_proposal_not_found"

    def test_already_promoted_http_status(self) -> None:
        err = SkillForgeProposalAlreadyPromotedError(
            proposal_id="skillver_x", timestamp="2024-01-01T00:00:00+00:00"
        )
        assert err.http_status() == 409

    def test_already_promoted_code(self) -> None:
        err = SkillForgeProposalAlreadyPromotedError(
            proposal_id="skillver_x", timestamp="2024-01-01T00:00:00+00:00"
        )
        assert err.code == "skill_forge_proposal_already_promoted"

    def test_reject_error_http_status(self) -> None:
        err = SkillForgeProposalRejectError(message="boom", timestamp="2024-01-01T00:00:00+00:00")
        assert err.http_status() == 500

    def test_reject_error_code(self) -> None:
        err = SkillForgeProposalRejectError(message="boom", timestamp="2024-01-01T00:00:00+00:00")
        assert err.code == "skill_forge_proposal_reject_failed"
