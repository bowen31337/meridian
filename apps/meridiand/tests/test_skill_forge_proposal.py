"""
Skill Forge proposal build conformance suite.

Tests cover:
  - build_skill_version_proposal stores proposal record in proposals_dir with status "PROPOSAL".
  - Proposal record has source="forge".
  - Proposal record has source_type="forge".
  - Proposal record has source_url=None.
  - Proposal record has derived_from_session_ids from the job.
  - Proposal record has derived_from_session_ids=None when job has no session ids.
  - Proposal record has correct skill_id.
  - Proposal record has instructions, tools, tests extracted from result_text JSON.
  - Proposal record has content-addressed ID with "skillver_" prefix.
  - Proposal record has job_id and run_id fields.
  - Proposal record has created_at timestamp.
  - build_skill_version_proposal returns the proposal_id.
  - Identical result_text and job produce the same proposal_id (content-addressed).
  - Different result_text produces different proposal_id.
  - Notifies primary user by writing a notification record to notifications_dir.
  - Notification record has user_id matching the primary user.
  - Notification record has proposal_id.
  - Notification record has type="skill_forge.proposal".
  - Notification record has skill_id, job_id, run_id.
  - Notification record has "notif_" id prefix.
  - When no primary user exists, no notification is written but operation succeeds.
  - Writes audit entry "skill_forge.proposal.created" on success.
  - Audit entry level is "info" on success.
  - Audit detail contains job_id, run_id, skill_id, proposal_id.
  - Audit detail contains notified_user_id on success when primary user exists.
  - Audit detail notified_user_id is None when no primary user.
  - On invalid JSON result_text, raises SkillForgeProposalError.
  - SkillForgeProposalError has http_status 500.
  - SkillForgeProposalError code is "skill_forge_proposal_failed".
  - On failure, writes audit entry "skill_forge.proposal.failed".
  - Failed audit entry level is "error".
  - Failed audit detail contains job_id, run_id, skill_id, message.
  - OTel span "skill_forge.build_proposal" emitted on success.
  - OTel span carries skill_forge.job_id, run_id, skill_id attributes.
  - OTel span sets skill_forge.proposal_id attribute on success.
  - OTel span sets skill_forge.build_proposal.success=True on success.
  - OTel span "skill_forge.build_proposal" emitted on failure.
  - OTel span sets skill_forge.build_proposal.success=False on failure.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from meridiand._audit import FileAuditLog
from meridiand._skill_forge import (
    SkillForgeProposalError,
    build_skill_version_proposal,
)
import pytest

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    job_id: str = "sfjob_test",
    skill_id: str = "skill_abc",
    derived_from_session_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "skill_id": skill_id,
        "job_type": "build_proposal",
        "derived_from_session_ids": derived_from_session_ids,
    }


def _make_result(
    instructions: str = "Do something useful.",
    tools: list[dict[str, Any]] | None = None,
    tests: list[dict[str, Any]] | None = None,
) -> str:
    return json.dumps(
        {
            "instructions": instructions,
            "tools": tools if tools is not None else [{"name": "tool1", "description": "A tool"}],
            "tests": tests if tests is not None else [],
        }
    )


def _write_primary_user(
    user_profiles_dir: Path,
    user_id: str = "user_primary",
) -> dict[str, Any]:
    user_profiles_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "id": user_id,
        "username": "primary",
        "is_primary": True,
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    (user_profiles_dir / f"{user_id}.json").write_text(json.dumps(record))
    return record


def _proposals(storage_root: Path) -> list[dict[str, Any]]:
    d = storage_root / "skill_forge" / "proposals"
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in d.glob("*.json")]


def _notifications(storage_root: Path) -> list[dict[str, Any]]:
    d = storage_root / "notifications"
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in d.glob("*.json")]


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _call(
    storage_root: Path,
    *,
    result_text: str | None = None,
    job: dict[str, Any] | None = None,
    run_id: str = "sfrun_test",
) -> str:
    audit_log = FileAuditLog(storage_root)
    return asyncio.run(
        build_skill_version_proposal(
            result_text=result_text if result_text is not None else _make_result(),
            job=job if job is not None else _make_job(),
            run_id=run_id,
            proposals_dir=storage_root / "skill_forge" / "proposals",
            user_profiles_dir=storage_root / "user_profiles",
            notifications_dir=storage_root / "notifications",
            audit_log=audit_log,
        )
    )


# ---------------------------------------------------------------------------
# Proposal record: storage and shape
# ---------------------------------------------------------------------------


class TestProposalRecordStorage:
    def test_stores_proposal_with_proposal_status(self, storage_root: Path) -> None:
        _call(storage_root)
        records = _proposals(storage_root)
        assert len(records) == 1
        assert records[0]["status"] == "PROPOSAL"

    def test_proposal_source_is_forge(self, storage_root: Path) -> None:
        _call(storage_root)
        assert _proposals(storage_root)[0]["source"] == "forge"

    def test_proposal_source_type_is_forge(self, storage_root: Path) -> None:
        _call(storage_root)
        assert _proposals(storage_root)[0]["source_type"] == "forge"

    def test_proposal_source_url_is_none(self, storage_root: Path) -> None:
        _call(storage_root)
        assert _proposals(storage_root)[0]["source_url"] is None

    def test_proposal_has_skill_id(self, storage_root: Path) -> None:
        _call(storage_root, job=_make_job(skill_id="skill_xyz"))
        assert _proposals(storage_root)[0]["skill_id"] == "skill_xyz"

    def test_proposal_has_instructions_from_result(self, storage_root: Path) -> None:
        result = _make_result(instructions="My instructions here.")
        _call(storage_root, result_text=result)
        assert _proposals(storage_root)[0]["instructions"] == "My instructions here."

    def test_proposal_has_tools_from_result(self, storage_root: Path) -> None:
        tools = [{"name": "my_tool", "description": "Does stuff"}]
        result = _make_result(tools=tools)
        _call(storage_root, result_text=result)
        assert _proposals(storage_root)[0]["tools"] == tools

    def test_proposal_has_tests_from_result(self, storage_root: Path) -> None:
        tests = [{"name": "test1", "input": {"x": 1}}]
        result = _make_result(tests=tests)
        _call(storage_root, result_text=result)
        assert _proposals(storage_root)[0]["tests"] == tests

    def test_proposal_has_derived_from_session_ids(self, storage_root: Path) -> None:
        job = _make_job(derived_from_session_ids=["sess_a", "sess_b"])
        _call(storage_root, job=job)
        assert _proposals(storage_root)[0]["derived_from_session_ids"] == ["sess_a", "sess_b"]

    def test_proposal_derived_from_session_ids_none_when_absent(self, storage_root: Path) -> None:
        _call(storage_root, job=_make_job())
        assert _proposals(storage_root)[0]["derived_from_session_ids"] is None

    def test_proposal_id_has_skillver_prefix(self, storage_root: Path) -> None:
        proposal_id = _call(storage_root)
        assert proposal_id.startswith("skillver_")

    def test_proposal_record_id_matches_returned_id(self, storage_root: Path) -> None:
        proposal_id = _call(storage_root)
        assert _proposals(storage_root)[0]["id"] == proposal_id

    def test_proposal_has_job_id(self, storage_root: Path) -> None:
        _call(storage_root, job=_make_job(job_id="sfjob_abc"))
        assert _proposals(storage_root)[0]["job_id"] == "sfjob_abc"

    def test_proposal_has_run_id(self, storage_root: Path) -> None:
        _call(storage_root, run_id="sfrun_xyz")
        assert _proposals(storage_root)[0]["run_id"] == "sfrun_xyz"

    def test_proposal_has_created_at(self, storage_root: Path) -> None:
        _call(storage_root)
        assert _proposals(storage_root)[0].get("created_at")


# ---------------------------------------------------------------------------
# Content-addressed ID determinism
# ---------------------------------------------------------------------------


class TestProposalVersionId:
    def test_same_inputs_produce_same_id(self, storage_root: Path, tmp_path: Path) -> None:
        root2 = tmp_path / "storage2"
        root2.mkdir()
        job = _make_job(skill_id="skill_same")
        result = _make_result(instructions="same")
        id1 = _call(storage_root, result_text=result, job=job)
        id2 = _call(root2, result_text=result, job=job)
        assert id1 == id2

    def test_different_instructions_produce_different_id(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        root2 = tmp_path / "storage2"
        root2.mkdir()
        job = _make_job(skill_id="skill_diff")
        id1 = _call(storage_root, result_text=_make_result(instructions="a"), job=job)
        id2 = _call(root2, result_text=_make_result(instructions="b"), job=job)
        assert id1 != id2

    def test_different_session_ids_produce_different_id(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        root2 = tmp_path / "storage2"
        root2.mkdir()
        result = _make_result()
        id1 = _call(
            storage_root,
            result_text=result,
            job=_make_job(derived_from_session_ids=["sess_1"]),
        )
        id2 = _call(
            root2,
            result_text=result,
            job=_make_job(derived_from_session_ids=["sess_2"]),
        )
        assert id1 != id2


# ---------------------------------------------------------------------------
# Primary user notification
# ---------------------------------------------------------------------------


class TestPrimaryUserNotification:
    def test_writes_notification_when_primary_user_exists(self, storage_root: Path) -> None:
        _write_primary_user(storage_root / "user_profiles")
        _call(storage_root)
        assert len(_notifications(storage_root)) == 1

    def test_notification_user_id_matches_primary_user(self, storage_root: Path) -> None:
        _write_primary_user(storage_root / "user_profiles", user_id="user_prim_test")
        _call(storage_root)
        notif = _notifications(storage_root)[0]
        assert notif["user_id"] == "user_prim_test"

    def test_notification_has_proposal_id(self, storage_root: Path) -> None:
        _write_primary_user(storage_root / "user_profiles")
        proposal_id = _call(storage_root)
        notif = _notifications(storage_root)[0]
        assert notif["proposal_id"] == proposal_id

    def test_notification_type_is_skill_forge_proposal(self, storage_root: Path) -> None:
        _write_primary_user(storage_root / "user_profiles")
        _call(storage_root)
        assert _notifications(storage_root)[0]["type"] == "skill_forge.proposal"

    def test_notification_has_skill_id(self, storage_root: Path) -> None:
        _write_primary_user(storage_root / "user_profiles")
        _call(storage_root, job=_make_job(skill_id="skill_notify"))
        assert _notifications(storage_root)[0]["skill_id"] == "skill_notify"

    def test_notification_has_job_id(self, storage_root: Path) -> None:
        _write_primary_user(storage_root / "user_profiles")
        _call(storage_root, job=_make_job(job_id="sfjob_notif"))
        assert _notifications(storage_root)[0]["job_id"] == "sfjob_notif"

    def test_notification_has_run_id(self, storage_root: Path) -> None:
        _write_primary_user(storage_root / "user_profiles")
        _call(storage_root, run_id="sfrun_notif")
        assert _notifications(storage_root)[0]["run_id"] == "sfrun_notif"

    def test_notification_id_has_notif_prefix(self, storage_root: Path) -> None:
        _write_primary_user(storage_root / "user_profiles")
        _call(storage_root)
        assert _notifications(storage_root)[0]["id"].startswith("notif_")

    def test_no_notification_when_no_primary_user(self, storage_root: Path) -> None:
        _call(storage_root)
        assert _notifications(storage_root) == []

    def test_no_primary_user_does_not_fail(self, storage_root: Path) -> None:
        proposal_id = _call(storage_root)
        assert proposal_id.startswith("skillver_")


# ---------------------------------------------------------------------------
# Audit log: success
# ---------------------------------------------------------------------------


class TestAuditLogSuccess:
    def test_writes_proposal_created_audit_entry(self, storage_root: Path) -> None:
        _call(storage_root)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.proposal.created" for r in records)

    def test_audit_level_is_info(self, storage_root: Path) -> None:
        _call(storage_root)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.created"
        )
        assert record["level"] == "info"

    def test_audit_detail_has_job_id(self, storage_root: Path) -> None:
        _call(storage_root, job=_make_job(job_id="sfjob_audit"))
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.created"
        )
        assert record["detail"]["job_id"] == "sfjob_audit"

    def test_audit_detail_has_run_id(self, storage_root: Path) -> None:
        _call(storage_root, run_id="sfrun_audit")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.created"
        )
        assert record["detail"]["run_id"] == "sfrun_audit"

    def test_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        _call(storage_root, job=_make_job(skill_id="skill_audit"))
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.created"
        )
        assert record["detail"]["skill_id"] == "skill_audit"

    def test_audit_detail_has_proposal_id(self, storage_root: Path) -> None:
        proposal_id = _call(storage_root)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.created"
        )
        assert record["detail"]["proposal_id"] == proposal_id

    def test_audit_detail_has_notified_user_id_when_primary_user_exists(
        self, storage_root: Path
    ) -> None:
        _write_primary_user(storage_root / "user_profiles", user_id="user_notif_audit")
        _call(storage_root)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.created"
        )
        assert record["detail"]["notified_user_id"] == "user_notif_audit"

    def test_audit_detail_notified_user_id_none_when_no_primary_user(
        self, storage_root: Path
    ) -> None:
        _call(storage_root)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.created"
        )
        assert record["detail"]["notified_user_id"] is None


# ---------------------------------------------------------------------------
# Failure: invalid JSON
# ---------------------------------------------------------------------------


class TestFailureInvalidJson:
    def test_raises_skill_forge_proposal_error(self, storage_root: Path) -> None:
        with pytest.raises(SkillForgeProposalError):
            _call(storage_root, result_text="not valid json{{{{")

    def test_error_http_status_is_500(self) -> None:
        err = SkillForgeProposalError(message="boom", timestamp="2024-01-01T00:00:00+00:00")
        assert err.http_status() == 500

    def test_error_code_is_skill_forge_proposal_failed(self) -> None:
        err = SkillForgeProposalError(message="boom", timestamp="2024-01-01T00:00:00+00:00")
        assert err.code == "skill_forge_proposal_failed"

    def test_failure_writes_proposal_failed_audit_entry(self, storage_root: Path) -> None:
        with pytest.raises(SkillForgeProposalError):
            _call(storage_root, result_text="not valid json")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.proposal.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        with pytest.raises(SkillForgeProposalError):
            _call(storage_root, result_text="bad")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_job_id(self, storage_root: Path) -> None:
        with pytest.raises(SkillForgeProposalError):
            _call(storage_root, result_text="bad", job=_make_job(job_id="sfjob_fail"))
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.failed"
        )
        assert record["detail"]["job_id"] == "sfjob_fail"

    def test_failure_audit_detail_has_run_id(self, storage_root: Path) -> None:
        with pytest.raises(SkillForgeProposalError):
            _call(storage_root, result_text="bad", run_id="sfrun_fail")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.failed"
        )
        assert record["detail"]["run_id"] == "sfrun_fail"

    def test_failure_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        with pytest.raises(SkillForgeProposalError):
            _call(
                storage_root,
                result_text="bad",
                job=_make_job(skill_id="skill_fail"),
            )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.failed"
        )
        assert record["detail"]["skill_id"] == "skill_fail"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        with pytest.raises(SkillForgeProposalError):
            _call(storage_root, result_text="bad")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.failed"
        )
        assert "message" in record["detail"] and record["detail"]["message"]

    def test_no_proposal_stored_on_failure(self, storage_root: Path) -> None:
        with pytest.raises(SkillForgeProposalError):
            _call(storage_root, result_text="not json")
        assert _proposals(storage_root) == []


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOtelSpans:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _get_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("skill_forge.build_proposal")

    def test_emits_build_proposal_span_on_success(self, storage_root: Path) -> None:
        _call(storage_root)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.build_proposal" in span_names

    def test_span_has_job_id_attribute(self, storage_root: Path) -> None:
        _call(storage_root, job=_make_job(job_id="sfjob_otel"))
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.job_id"] == "sfjob_otel"

    def test_span_has_run_id_attribute(self, storage_root: Path) -> None:
        _call(storage_root, run_id="sfrun_otel")
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.run_id"] == "sfrun_otel"

    def test_span_has_skill_id_attribute(self, storage_root: Path) -> None:
        _call(storage_root, job=_make_job(skill_id="skill_otel"))
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.skill_id"] == "skill_otel"

    def test_span_has_proposal_id_attribute_on_success(self, storage_root: Path) -> None:
        proposal_id = _call(storage_root)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal_id"] == proposal_id

    def test_span_success_attribute_true_on_success(self, storage_root: Path) -> None:
        _call(storage_root)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.build_proposal.success"] is True

    def test_emits_build_proposal_span_on_failure(self, storage_root: Path) -> None:
        with pytest.raises(SkillForgeProposalError):
            _call(storage_root, result_text="bad json")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.build_proposal" in span_names

    def test_span_success_attribute_false_on_failure(self, storage_root: Path) -> None:
        with pytest.raises(SkillForgeProposalError):
            _call(storage_root, result_text="bad json")
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.build_proposal.success"] is False
