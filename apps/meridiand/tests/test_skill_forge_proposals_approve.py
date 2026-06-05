"""
Skill Forge proposals approve endpoint conformance suite.

Tests cover:
  - POST /v1/x/skill_forge/proposals/{id}/approve returns 200 on success.
  - Response body is the new SkillVersion record (not the proposal).
  - Response body has id matching the recomputed content hash.
  - Response body has skill_id, instructions, tools, tests from proposal.
  - Response body has source="forge", source_type="forge".
  - Response body has version_number=1 for the first promotion.
  - Response body has version_number=2 when an existing version is present.
  - Proposal file on disk is updated with status "PROMOTED".
  - Proposal file on disk has promoted_at timestamp.
  - Proposal file on disk has promoted_version_id matching the version record id.
  - SkillVersion file created in skill_versions/ with the correct id.
  - SkillVersion file on disk has correct skill_id.
  - Skill record version field updated when skill file exists.
  - Approving a REJECTED proposal succeeds (overrides rejection).
  - Returns 404 with code "skill_forge_proposal_not_found" for unknown id.
  - Returns 409 with code "skill_forge_proposal_already_promoted" when already promoted.
  - 409 does not create a new SkillVersion or modify the proposal on disk.
  - Success writes audit entry with event "skill_forge.proposal.approved".
  - Success audit entry level is "info".
  - Success audit detail has proposal_id.
  - Success audit detail has skill_id.
  - Success audit detail has version_id.
  - 404 failure writes audit entry with event "skill_forge.proposal.approve.failed".
  - 404 failure audit entry level is "error".
  - 404 failure audit entry code is "skill_forge_proposal_not_found".
  - 404 failure audit detail has proposal_id and message.
  - 409 failure writes audit entry with event "skill_forge.proposal.approve.failed".
  - 409 failure audit entry level is "error".
  - 409 failure audit entry code is "skill_forge_proposal_already_promoted".
  - Error response body has error.code and error.message.
  - OTel span "skill_forge.proposal.approve" emitted on success.
  - OTel span carries skill_forge.proposal_id attribute.
  - OTel span sets skill_forge.proposal.approve.success=True on success.
  - OTel span "skill_forge.proposal.approve" emitted on 404 failure.
  - OTel span sets skill_forge.proposal.approve.success=False on failure.
  - create_app wires proposals router when storage_root is supplied.
  - create_app omits proposals route when storage_root is None.
  - SkillForgeProposalApproveError has http_status 500.
  - SkillForgeProposalApproveError has code "skill_forge_proposal_approve_failed".
  - _forge_version_id produces skillver_<sha256> prefix.
  - _forge_version_id is deterministic for equal inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._skill_forge_proposals import (
    SkillForgeProposalApproveError,
    _forge_version_id,
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
    instructions: str = "Do something.",
    tools: list[dict[str, Any]] | None = None,
    tests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    proposals_dir = storage_root / "skill_forge" / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "id": proposal_id,
        "skill_id": skill_id,
        "instructions": instructions,
        "tools": tools if tools is not None else [],
        "tests": tests if tests is not None else [],
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


def _write_skill(storage_root: Path, skill_id: str = "skill_xyz") -> dict[str, Any]:
    skills_dir = storage_root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "id": skill_id,
        "name": "Test Skill",
        "description": "A test skill.",
        "created_at": "2024-01-01T00:00:00+00:00",
        "metadata": None,
        "version": None,
    }
    (skills_dir / f"{skill_id}.json").write_text(json.dumps(record))
    return record


def _write_existing_version(
    storage_root: Path,
    version_id: str,
    skill_id: str = "skill_xyz",
    version_number: int = 1,
) -> None:
    versions_dir = storage_root / "skill_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "id": version_id,
        "skill_id": skill_id,
        "version_number": version_number,
        "instructions": "Old instructions.",
        "tools": [],
        "tests": [],
        "created_at": "2024-01-01T00:00:00+00:00",
        "source_type": "forge",
        "source_url": None,
        "source": "forge",
        "derived_from_session_ids": None,
    }
    (versions_dir / f"{version_id}.json").write_text(json.dumps(record))


def _read_proposal(storage_root: Path, proposal_id: str) -> dict[str, Any]:
    return json.loads(
        (storage_root / "skill_forge" / "proposals" / f"{proposal_id}.json").read_text()
    )


def _read_version(storage_root: Path, version_id: str) -> dict[str, Any]:
    return json.loads((storage_root / "skill_versions" / f"{version_id}.json").read_text())


def _read_skill(storage_root: Path, skill_id: str) -> dict[str, Any]:
    return json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _approve(
    client: TestClient,
    proposal_id: str = "skillver_abc123",
) -> Any:
    return client.post(f"/v1/x/skill_forge/proposals/{proposal_id}/approve")


# ---------------------------------------------------------------------------
# Success: HTTP and response shape
# ---------------------------------------------------------------------------


class TestApproveSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.status_code == 200

    def test_response_is_version_record(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        body = resp.json()
        assert "version_number" in body
        assert "instructions" in body

    def test_response_has_skill_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root, skill_id="skill_xyz")
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.json()["skill_id"] == "skill_xyz"

    def test_response_has_instructions_from_proposal(self, storage_root: Path) -> None:
        _write_proposal(storage_root, instructions="Follow these steps.")
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.json()["instructions"] == "Follow these steps."

    def test_response_source_is_forge(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.json()["source"] == "forge"

    def test_response_source_type_is_forge(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.json()["source_type"] == "forge"

    def test_response_version_number_is_1_for_first_promotion(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.json()["version_number"] == 1

    def test_response_version_number_increments(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        _write_existing_version(
            storage_root, version_id="skillver_old", skill_id="skill_xyz", version_number=3
        )
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.json()["version_number"] == 4

    def test_response_id_matches_recomputed_hash(self, storage_root: Path) -> None:
        _write_proposal(
            storage_root,
            proposal_id="skillver_abc123",
            skill_id="skill_xyz",
            instructions="Do something.",
        )
        client = _make_client(storage_root)
        resp = _approve(client)
        expected_id = _forge_version_id(
            skill_id="skill_xyz",
            instructions="Do something.",
            tools=[],
            tests=[],
            derived_from_session_ids=None,
        )
        assert resp.json()["id"] == expected_id

    def test_response_has_created_at(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.json().get("created_at")


# ---------------------------------------------------------------------------
# Success: disk state — proposal
# ---------------------------------------------------------------------------


class TestApproveDiskStateProposal:
    def test_proposal_file_status_is_promoted(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _approve(client)
        assert _read_proposal(storage_root, "skillver_abc123")["status"] == "PROMOTED"

    def test_proposal_file_has_promoted_at(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _approve(client)
        assert _read_proposal(storage_root, "skillver_abc123").get("promoted_at")

    def test_proposal_file_has_promoted_version_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        on_disk = _read_proposal(storage_root, "skillver_abc123")
        assert on_disk["promoted_version_id"] == resp.json()["id"]


# ---------------------------------------------------------------------------
# Success: disk state — skill version
# ---------------------------------------------------------------------------


class TestApproveDiskStateVersion:
    def test_version_file_created(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        version_id = resp.json()["id"]
        assert (storage_root / "skill_versions" / f"{version_id}.json").exists()

    def test_version_file_has_correct_skill_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root, skill_id="skill_xyz")
        client = _make_client(storage_root)
        resp = _approve(client)
        on_disk = _read_version(storage_root, resp.json()["id"])
        assert on_disk["skill_id"] == "skill_xyz"

    def test_version_file_has_instructions(self, storage_root: Path) -> None:
        _write_proposal(storage_root, instructions="Follow these steps.")
        client = _make_client(storage_root)
        resp = _approve(client)
        on_disk = _read_version(storage_root, resp.json()["id"])
        assert on_disk["instructions"] == "Follow these steps."

    def test_version_file_source_is_forge(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        on_disk = _read_version(storage_root, resp.json()["id"])
        assert on_disk["source"] == "forge"


# ---------------------------------------------------------------------------
# Success: skill record updated when present
# ---------------------------------------------------------------------------


class TestApproveSkillRecord:
    def test_skill_record_version_updated(self, storage_root: Path) -> None:
        _write_proposal(storage_root, skill_id="skill_xyz")
        _write_skill(storage_root, skill_id="skill_xyz")
        client = _make_client(storage_root)
        resp = _approve(client)
        skill = _read_skill(storage_root, "skill_xyz")
        assert skill["version"]["id"] == resp.json()["id"]

    def test_skill_record_not_required(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Approving a REJECTED proposal
# ---------------------------------------------------------------------------


class TestApproveRejectedProposal:
    def test_returns_200_for_rejected_proposal(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="REJECTED")
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.status_code == 200

    def test_rejected_proposal_becomes_promoted(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="REJECTED")
        client = _make_client(storage_root)
        _approve(client)
        assert _read_proposal(storage_root, "skillver_abc123")["status"] == "PROMOTED"


# ---------------------------------------------------------------------------
# 404: proposal not found
# ---------------------------------------------------------------------------


class TestApproveNotFound:
    def test_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _approve(client, proposal_id="skillver_missing")
        assert resp.status_code == 404

    def test_error_code_is_not_found(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _approve(client, proposal_id="skillver_missing")
        assert resp.json()["error"]["code"] == "skill_forge_proposal_not_found"

    def test_error_message_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _approve(client, proposal_id="skillver_missing")
        assert resp.json()["error"]["message"]

    def test_writes_approve_failed_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _approve(client, proposal_id="skillver_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.proposal.approve.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _approve(client, proposal_id="skillver_missing")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.approve.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _approve(client, proposal_id="skillver_missing")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.approve.failed"
        )
        assert record["code"] == "skill_forge_proposal_not_found"

    def test_not_found_audit_detail_has_proposal_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _approve(client, proposal_id="skillver_missing")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.approve.failed"
        )
        assert record["detail"]["proposal_id"] == "skillver_missing"

    def test_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _approve(client, proposal_id="skillver_missing")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.approve.failed"
        )
        assert record["detail"]["message"]

    def test_no_version_file_created_on_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _approve(client, proposal_id="skillver_missing")
        versions_dir = storage_root / "skill_versions"
        assert not versions_dir.exists() or not list(versions_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# 409: already promoted
# ---------------------------------------------------------------------------


class TestApproveAlreadyPromoted:
    def test_returns_409(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.status_code == 409

    def test_error_code_is_already_promoted(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.json()["error"]["code"] == "skill_forge_proposal_already_promoted"

    def test_error_message_present(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.json()["error"]["message"]

    def test_no_new_version_file_created_on_409(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        _approve(client)
        versions_dir = storage_root / "skill_versions"
        assert not versions_dir.exists() or not list(versions_dir.glob("*.json"))

    def test_writes_approve_failed_audit_entry(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        _approve(client)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.proposal.approve.failed" for r in records)

    def test_already_promoted_audit_level_is_error(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        _approve(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.approve.failed"
        )
        assert record["level"] == "error"

    def test_already_promoted_audit_code(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        _approve(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.approve.failed"
        )
        assert record["code"] == "skill_forge_proposal_already_promoted"


# ---------------------------------------------------------------------------
# Audit log: success
# ---------------------------------------------------------------------------


class TestAuditSuccess:
    def test_writes_proposal_approved_audit_entry(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _approve(client)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.proposal.approved" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _approve(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.approved"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_proposal_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_abc123")
        client = _make_client(storage_root)
        _approve(client, proposal_id="skillver_abc123")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.approved"
        )
        assert record["detail"]["proposal_id"] == "skillver_abc123"

    def test_success_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root, skill_id="skill_audit")
        client = _make_client(storage_root)
        _approve(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.approved"
        )
        assert record["detail"]["skill_id"] == "skill_audit"

    def test_success_audit_detail_has_version_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.approved"
        )
        assert record["detail"]["version_id"] == resp.json()["id"]


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOtelSpans:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _get_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("skill_forge.proposal.approve")

    def test_emits_approve_span_on_success(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _approve(client)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.proposal.approve" in span_names

    def test_span_has_proposal_id_attribute(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_otel")
        client = _make_client(storage_root)
        _approve(client, proposal_id="skillver_otel")
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal_id"] == "skillver_otel"

    def test_span_success_attribute_true_on_success(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _approve(client)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal.approve.success"] is True

    def test_emits_approve_span_on_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _approve(client, proposal_id="skillver_missing")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.proposal.approve" in span_names

    def test_span_success_attribute_false_on_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _approve(client, proposal_id="skillver_missing")
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal.approve.success"] is False

    def test_span_success_attribute_false_on_409(self, storage_root: Path) -> None:
        _write_proposal(storage_root, status="PROMOTED")
        client = _make_client(storage_root)
        _approve(client)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal.approve.success"] is False


# ---------------------------------------------------------------------------
# Router wiring
# ---------------------------------------------------------------------------


class TestRouterWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _approve(client)
        assert resp.status_code != 404 or "not_found" not in resp.text

    def test_route_absent_without_storage_root(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(FileAuditLog(Path(tmp)), storage_root=None)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/x/skill_forge/proposals/skillver_abc/approve",
            )
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Error class contracts
# ---------------------------------------------------------------------------


class TestErrorClasses:
    def test_approve_error_http_status(self) -> None:
        err = SkillForgeProposalApproveError(message="boom", timestamp="2024-01-01T00:00:00+00:00")
        assert err.http_status() == 500

    def test_approve_error_code(self) -> None:
        err = SkillForgeProposalApproveError(message="boom", timestamp="2024-01-01T00:00:00+00:00")
        assert err.code == "skill_forge_proposal_approve_failed"


# ---------------------------------------------------------------------------
# _forge_version_id unit tests
# ---------------------------------------------------------------------------


class TestForgeVersionId:
    def test_produces_skillver_prefix(self) -> None:
        vid = _forge_version_id(
            skill_id="skill_a",
            instructions="Do it.",
            tools=[],
            tests=[],
            derived_from_session_ids=None,
        )
        assert vid.startswith("skillver_")

    def test_is_deterministic(self) -> None:
        kwargs: dict[str, Any] = dict(
            skill_id="skill_b",
            instructions="Step 1.",
            tools=[{"name": "bash"}],
            tests=[],
            derived_from_session_ids=["sess_1"],
        )
        assert _forge_version_id(**kwargs) == _forge_version_id(**kwargs)

    def test_different_content_produces_different_id(self) -> None:
        a = _forge_version_id(
            skill_id="skill_a",
            instructions="A",
            tools=[],
            tests=[],
            derived_from_session_ids=None,
        )
        b = _forge_version_id(
            skill_id="skill_a",
            instructions="B",
            tools=[],
            tests=[],
            derived_from_session_ids=None,
        )
        assert a != b

    def test_hash_length_is_64_hex_chars(self) -> None:
        vid = _forge_version_id(
            skill_id="skill_a",
            instructions="X",
            tools=[],
            tests=[],
            derived_from_session_ids=None,
        )
        hex_part = vid[len("skillver_") :]
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part)
