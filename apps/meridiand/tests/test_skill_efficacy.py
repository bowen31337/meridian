"""
Skill efficacy A/B trajectory comparison conformance suite.

Tests cover:
  - compare_proposal_trajectories stores efficacy record in efficacy_dir.
  - Efficacy record has id with "skefficacy_" prefix.
  - Efficacy record has proposal_id matching the proposal.
  - Efficacy record has skill_id matching the proposal.
  - Efficacy record has test_case_count equal to len(proposal tests).
  - Efficacy record has pass_rate_without_skill computed from runner results.
  - Efficacy record has pass_rate_with_skill computed from runner results.
  - Efficacy record has lift = pass_rate_with_skill - pass_rate_without_skill.
  - Efficacy record has case_results list with one entry per test case.
  - Each case_result has test_name from the test case.
  - Each case_result has passed_without_skill from the without-arm runner call.
  - Each case_result has passed_with_skill from the with-arm runner call.
  - Efficacy record has created_at timestamp.
  - compare_proposal_trajectories returns the metric record dict.
  - Runner called with skill_instructions=None for without-skill arm.
  - Runner called with skill_instructions=proposal["instructions"] for with-skill arm.
  - When proposal has no tests, lift is 0.0 and test_case_count is 0.
  - When all tests pass with skill and none pass without, lift is 1.0.
  - When all tests pass without skill and none pass with, lift is -1.0.
  - Writes audit entry "skill_efficacy.compared" on success.
  - Audit entry level is "info" on success.
  - Audit detail contains metric_id, proposal_id, skill_id, test_case_count.
  - Audit detail contains pass_rate_without_skill, pass_rate_with_skill, lift.
  - On runner exception, raises SkillEfficacyError.
  - SkillEfficacyError has http_status 500.
  - SkillEfficacyError code is "skill_efficacy_failed".
  - On failure, writes audit entry "skill_efficacy.compare.failed".
  - Failed audit entry level is "error".
  - Failed audit detail contains metric_id, proposal_id, skill_id, message.
  - No efficacy record written on failure.
  - OTel span "skill_efficacy.compare_trajectories" emitted on success.
  - OTel span carries proposal_id, skill_id, metric_id, test_case_count attributes.
  - OTel span sets pass_rate_without_skill, pass_rate_with_skill, lift on success.
  - OTel span sets skill_efficacy.compare_trajectories.success=True on success.
  - OTel span "skill_efficacy.compare_trajectories" emitted on failure.
  - OTel span sets skill_efficacy.compare_trajectories.success=False on failure.
  - Proposal integration: build_skill_version_proposal stores efficacy record for proposal.
  - Proposal integration: efficacy record has proposal_id matching the created proposal.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from meridiand._audit import FileAuditLog
from meridiand._skill_efficacy import (
    SkillEfficacyError,
    compare_proposal_trajectories,
)
from meridiand._skill_forge import build_skill_version_proposal

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FixedRunner:
    """Returns configurable pass/fail for each arm; records calls."""

    def __init__(
        self,
        *,
        with_skill: bool = True,
        without_skill: bool = False,
    ) -> None:
        self._with = with_skill
        self._without = without_skill
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        test_case: dict[str, Any],
        *,
        skill_instructions: str | None,
    ) -> bool:
        self.calls.append(
            {"test_case": test_case, "skill_instructions": skill_instructions}
        )
        return self._with if skill_instructions is not None else self._without


class _ErrorRunner:
    async def run(
        self,
        test_case: dict[str, Any],
        *,
        skill_instructions: str | None,
    ) -> bool:
        raise RuntimeError("runner blew up")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proposal(
    proposal_id: str = "skillver_abc123",
    skill_id: str = "skill_xyz",
    instructions: str = "Do something useful.",
    tests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": proposal_id,
        "skill_id": skill_id,
        "instructions": instructions,
        "tools": [],
        "tests": tests if tests is not None else [{"name": "test_one", "input": {}}],
        "source": "forge",
        "source_type": "forge",
        "source_url": None,
        "derived_from_session_ids": None,
        "run_id": "sfrun_test",
        "job_id": "sfjob_test",
        "status": "PROPOSAL",
        "created_at": "2024-01-01T00:00:00+00:00",
    }


def _make_job(
    job_id: str = "sfjob_test",
    skill_id: str = "skill_xyz",
    derived_from_session_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "skill_id": skill_id,
        "job_type": "build_proposal",
        "derived_from_session_ids": derived_from_session_ids,
    }


def _make_forge_result(
    instructions: str = "Do something useful.",
    tests: list[dict[str, Any]] | None = None,
) -> str:
    return json.dumps({
        "instructions": instructions,
        "tools": [],
        "tests": tests if tests is not None else [{"name": "test_one", "input": {}}],
    })


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _efficacy_records(efficacy_dir: Path) -> list[dict[str, Any]]:
    if not efficacy_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in efficacy_dir.glob("*_efficacy.json")]


def _call(
    storage_root: Path,
    *,
    proposal: dict[str, Any] | None = None,
    runner: Any = None,
    efficacy_dir: Path | None = None,
) -> dict[str, Any]:
    audit_log = FileAuditLog(storage_root)
    _eff_dir = efficacy_dir if efficacy_dir is not None else storage_root / "efficacy"
    _proposal = proposal if proposal is not None else _make_proposal()
    return asyncio.run(
        compare_proposal_trajectories(
            proposal=_proposal,
            efficacy_dir=_eff_dir,
            audit_log=audit_log,
            runner=runner if runner is not None else _FixedRunner(),
        )
    )


# ---------------------------------------------------------------------------
# Efficacy record: storage and shape
# ---------------------------------------------------------------------------


class TestEfficacyRecordStorage:
    def test_stores_efficacy_record_in_efficacy_dir(self, storage_root: Path) -> None:
        eff_dir = storage_root / "efficacy"
        _call(storage_root, efficacy_dir=eff_dir)
        assert len(_efficacy_records(eff_dir)) == 1

    def test_efficacy_record_id_has_skefficacy_prefix(
        self, storage_root: Path
    ) -> None:
        result = _call(storage_root)
        assert result["id"].startswith("skefficacy_")

    def test_efficacy_record_has_proposal_id(self, storage_root: Path) -> None:
        proposal = _make_proposal(proposal_id="skillver_pid")
        result = _call(storage_root, proposal=proposal)
        assert result["proposal_id"] == "skillver_pid"

    def test_efficacy_record_has_skill_id(self, storage_root: Path) -> None:
        proposal = _make_proposal(skill_id="skill_abc")
        result = _call(storage_root, proposal=proposal)
        assert result["skill_id"] == "skill_abc"

    def test_efficacy_record_has_test_case_count(self, storage_root: Path) -> None:
        proposal = _make_proposal(
            tests=[{"name": "t1"}, {"name": "t2"}, {"name": "t3"}]
        )
        result = _call(storage_root, proposal=proposal)
        assert result["test_case_count"] == 3

    def test_efficacy_record_has_pass_rate_without_skill(
        self, storage_root: Path
    ) -> None:
        result = _call(storage_root, runner=_FixedRunner(with_skill=True, without_skill=False))
        assert "pass_rate_without_skill" in result

    def test_efficacy_record_has_pass_rate_with_skill(
        self, storage_root: Path
    ) -> None:
        result = _call(storage_root, runner=_FixedRunner(with_skill=True, without_skill=False))
        assert "pass_rate_with_skill" in result

    def test_efficacy_record_has_lift(self, storage_root: Path) -> None:
        result = _call(storage_root)
        assert "lift" in result

    def test_efficacy_record_has_case_results_list(self, storage_root: Path) -> None:
        result = _call(storage_root)
        assert isinstance(result["case_results"], list)

    def test_case_results_count_matches_test_cases(self, storage_root: Path) -> None:
        proposal = _make_proposal(tests=[{"name": "a"}, {"name": "b"}])
        result = _call(storage_root, proposal=proposal)
        assert len(result["case_results"]) == 2

    def test_case_result_has_test_name(self, storage_root: Path) -> None:
        proposal = _make_proposal(tests=[{"name": "my_test"}])
        result = _call(storage_root, proposal=proposal)
        assert result["case_results"][0]["test_name"] == "my_test"

    def test_case_result_has_passed_without_skill(self, storage_root: Path) -> None:
        result = _call(storage_root, runner=_FixedRunner(without_skill=False))
        assert result["case_results"][0]["passed_without_skill"] is False

    def test_case_result_has_passed_with_skill(self, storage_root: Path) -> None:
        result = _call(storage_root, runner=_FixedRunner(with_skill=True))
        assert result["case_results"][0]["passed_with_skill"] is True

    def test_efficacy_record_has_created_at(self, storage_root: Path) -> None:
        result = _call(storage_root)
        assert result.get("created_at")

    def test_compare_returns_metric_record(self, storage_root: Path) -> None:
        result = _call(storage_root)
        assert isinstance(result, dict)
        assert "id" in result and "lift" in result


# ---------------------------------------------------------------------------
# Metric computation correctness
# ---------------------------------------------------------------------------


class TestMetricComputation:
    def test_runner_called_with_none_instructions_for_without_arm(
        self, storage_root: Path
    ) -> None:
        runner = _FixedRunner()
        _call(storage_root, runner=runner)
        without_calls = [c for c in runner.calls if c["skill_instructions"] is None]
        assert len(without_calls) == 1

    def test_runner_called_with_instructions_for_with_arm(
        self, storage_root: Path
    ) -> None:
        runner = _FixedRunner()
        proposal = _make_proposal(instructions="My instructions.")
        _call(storage_root, proposal=proposal, runner=runner)
        with_calls = [c for c in runner.calls if c["skill_instructions"] == "My instructions."]
        assert len(with_calls) == 1

    def test_no_tests_gives_zero_lift(self, storage_root: Path) -> None:
        proposal = _make_proposal(tests=[])
        result = _call(storage_root, proposal=proposal)
        assert result["lift"] == 0.0
        assert result["test_case_count"] == 0

    def test_all_pass_with_skill_none_without_lift_is_one(
        self, storage_root: Path
    ) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}, {"name": "t2"}])
        runner = _FixedRunner(with_skill=True, without_skill=False)
        result = _call(storage_root, proposal=proposal, runner=runner)
        assert result["lift"] == pytest.approx(1.0)

    def test_all_pass_without_skill_none_with_lift_is_negative_one(
        self, storage_root: Path
    ) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}, {"name": "t2"}])
        runner = _FixedRunner(with_skill=False, without_skill=True)
        result = _call(storage_root, proposal=proposal, runner=runner)
        assert result["lift"] == pytest.approx(-1.0)

    def test_lift_equals_pass_rate_with_minus_pass_rate_without(
        self, storage_root: Path
    ) -> None:
        result = _call(storage_root, runner=_FixedRunner(with_skill=True, without_skill=False))
        expected = result["pass_rate_with_skill"] - result["pass_rate_without_skill"]
        assert result["lift"] == pytest.approx(expected)

    def test_pass_rate_with_skill_computed_correctly(
        self, storage_root: Path
    ) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}, {"name": "t2"}])
        runner = _FixedRunner(with_skill=True, without_skill=False)
        result = _call(storage_root, proposal=proposal, runner=runner)
        assert result["pass_rate_with_skill"] == pytest.approx(1.0)

    def test_pass_rate_without_skill_computed_correctly(
        self, storage_root: Path
    ) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}, {"name": "t2"}])
        runner = _FixedRunner(with_skill=True, without_skill=False)
        result = _call(storage_root, proposal=proposal, runner=runner)
        assert result["pass_rate_without_skill"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Audit log: success
# ---------------------------------------------------------------------------


class TestAuditLogSuccess:
    def test_writes_compared_audit_entry(self, storage_root: Path) -> None:
        _call(storage_root)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_efficacy.compared" for r in records)

    def test_audit_level_is_info(self, storage_root: Path) -> None:
        _call(storage_root)
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compared"
        )
        assert record["level"] == "info"

    def test_audit_detail_has_metric_id(self, storage_root: Path) -> None:
        result = _call(storage_root)
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compared"
        )
        assert record["detail"]["metric_id"] == result["id"]

    def test_audit_detail_has_proposal_id(self, storage_root: Path) -> None:
        proposal = _make_proposal(proposal_id="skillver_audit")
        _call(storage_root, proposal=proposal)
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compared"
        )
        assert record["detail"]["proposal_id"] == "skillver_audit"

    def test_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        proposal = _make_proposal(skill_id="skill_audit")
        _call(storage_root, proposal=proposal)
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compared"
        )
        assert record["detail"]["skill_id"] == "skill_audit"

    def test_audit_detail_has_test_case_count(self, storage_root: Path) -> None:
        proposal = _make_proposal(tests=[{"name": "a"}, {"name": "b"}])
        _call(storage_root, proposal=proposal)
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compared"
        )
        assert record["detail"]["test_case_count"] == 2

    def test_audit_detail_has_pass_rate_without_skill(
        self, storage_root: Path
    ) -> None:
        _call(storage_root)
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compared"
        )
        assert "pass_rate_without_skill" in record["detail"]

    def test_audit_detail_has_pass_rate_with_skill(self, storage_root: Path) -> None:
        _call(storage_root)
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compared"
        )
        assert "pass_rate_with_skill" in record["detail"]

    def test_audit_detail_has_lift(self, storage_root: Path) -> None:
        _call(storage_root)
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compared"
        )
        assert "lift" in record["detail"]


# ---------------------------------------------------------------------------
# Failure: runner raises
# ---------------------------------------------------------------------------


class TestFailureRunnerRaises:
    def test_raises_skill_efficacy_error(self, storage_root: Path) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}])
        with pytest.raises(SkillEfficacyError):
            _call(storage_root, proposal=proposal, runner=_ErrorRunner())

    def test_error_http_status_is_500(self) -> None:
        err = SkillEfficacyError(message="boom", timestamp="2024-01-01T00:00:00+00:00")
        assert err.http_status() == 500

    def test_error_code_is_skill_efficacy_failed(self) -> None:
        err = SkillEfficacyError(message="boom", timestamp="2024-01-01T00:00:00+00:00")
        assert err.code == "skill_efficacy_failed"

    def test_failure_writes_compare_failed_audit_entry(
        self, storage_root: Path
    ) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}])
        with pytest.raises(SkillEfficacyError):
            _call(storage_root, proposal=proposal, runner=_ErrorRunner())
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_efficacy.compare.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}])
        with pytest.raises(SkillEfficacyError):
            _call(storage_root, proposal=proposal, runner=_ErrorRunner())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compare.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_metric_id(self, storage_root: Path) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}])
        with pytest.raises(SkillEfficacyError):
            _call(storage_root, proposal=proposal, runner=_ErrorRunner())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compare.failed"
        )
        assert record["detail"]["metric_id"].startswith("skefficacy_")

    def test_failure_audit_detail_has_proposal_id(self, storage_root: Path) -> None:
        proposal = _make_proposal(proposal_id="skillver_fail", tests=[{"name": "t1"}])
        with pytest.raises(SkillEfficacyError):
            _call(storage_root, proposal=proposal, runner=_ErrorRunner())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compare.failed"
        )
        assert record["detail"]["proposal_id"] == "skillver_fail"

    def test_failure_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        proposal = _make_proposal(skill_id="skill_fail", tests=[{"name": "t1"}])
        with pytest.raises(SkillEfficacyError):
            _call(storage_root, proposal=proposal, runner=_ErrorRunner())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compare.failed"
        )
        assert record["detail"]["skill_id"] == "skill_fail"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}])
        with pytest.raises(SkillEfficacyError):
            _call(storage_root, proposal=proposal, runner=_ErrorRunner())
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_efficacy.compare.failed"
        )
        assert "message" in record["detail"] and record["detail"]["message"]

    def test_no_efficacy_record_written_on_failure(self, storage_root: Path) -> None:
        eff_dir = storage_root / "efficacy"
        proposal = _make_proposal(tests=[{"name": "t1"}])
        with pytest.raises(SkillEfficacyError):
            _call(storage_root, proposal=proposal, runner=_ErrorRunner(), efficacy_dir=eff_dir)
        assert _efficacy_records(eff_dir) == []


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOtelSpans:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _get_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("skill_efficacy.compare_trajectories")

    def test_emits_compare_trajectories_span_on_success(
        self, storage_root: Path
    ) -> None:
        _call(storage_root)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_efficacy.compare_trajectories" in span_names

    def test_span_has_proposal_id_attribute(self, storage_root: Path) -> None:
        proposal = _make_proposal(proposal_id="skillver_otel")
        _call(storage_root, proposal=proposal)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_efficacy.proposal_id"] == "skillver_otel"

    def test_span_has_skill_id_attribute(self, storage_root: Path) -> None:
        proposal = _make_proposal(skill_id="skill_otel")
        _call(storage_root, proposal=proposal)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_efficacy.skill_id"] == "skill_otel"

    def test_span_has_metric_id_attribute(self, storage_root: Path) -> None:
        result = _call(storage_root)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_efficacy.metric_id"] == result["id"]

    def test_span_has_test_case_count_attribute(self, storage_root: Path) -> None:
        proposal = _make_proposal(tests=[{"name": "a"}, {"name": "b"}])
        _call(storage_root, proposal=proposal)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_efficacy.test_case_count"] == 2

    def test_span_has_pass_rate_without_skill_attribute_on_success(
        self, storage_root: Path
    ) -> None:
        _call(storage_root, runner=_FixedRunner(with_skill=True, without_skill=False))
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_efficacy.pass_rate_without_skill"] == pytest.approx(0.0)

    def test_span_has_pass_rate_with_skill_attribute_on_success(
        self, storage_root: Path
    ) -> None:
        _call(storage_root, runner=_FixedRunner(with_skill=True, without_skill=False))
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_efficacy.pass_rate_with_skill"] == pytest.approx(1.0)

    def test_span_has_lift_attribute_on_success(self, storage_root: Path) -> None:
        _call(storage_root, runner=_FixedRunner(with_skill=True, without_skill=False))
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_efficacy.lift"] == pytest.approx(1.0)

    def test_span_success_attribute_true_on_success(
        self, storage_root: Path
    ) -> None:
        _call(storage_root)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_efficacy.compare_trajectories.success"] is True

    def test_emits_compare_trajectories_span_on_failure(
        self, storage_root: Path
    ) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}])
        with pytest.raises(SkillEfficacyError):
            _call(storage_root, proposal=proposal, runner=_ErrorRunner())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_efficacy.compare_trajectories" in span_names

    def test_span_success_attribute_false_on_failure(
        self, storage_root: Path
    ) -> None:
        proposal = _make_proposal(tests=[{"name": "t1"}])
        with pytest.raises(SkillEfficacyError):
            _call(storage_root, proposal=proposal, runner=_ErrorRunner())
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_efficacy.compare_trajectories.success"] is False


# ---------------------------------------------------------------------------
# Proposal integration: efficacy recorded on build_skill_version_proposal
# ---------------------------------------------------------------------------


class TestProposalIntegration:
    def _call_build(
        self,
        storage_root: Path,
        *,
        runner: Any = None,
        tests: list[dict[str, Any]] | None = None,
    ) -> tuple[str, Path]:
        audit_log = FileAuditLog(storage_root)
        eff_dir = storage_root / "skill_forge" / "efficacy"
        result_text = _make_forge_result(
            tests=tests if tests is not None else [{"name": "t1"}]
        )
        proposal_id = asyncio.run(
            build_skill_version_proposal(
                result_text=result_text,
                job=_make_job(),
                run_id="sfrun_integ",
                proposals_dir=storage_root / "skill_forge" / "proposals",
                user_profiles_dir=storage_root / "user_profiles",
                notifications_dir=storage_root / "notifications",
                audit_log=audit_log,
                efficacy_dir=eff_dir,
                trajectory_runner=runner if runner is not None else _FixedRunner(),
            )
        )
        return proposal_id, eff_dir

    def test_build_proposal_stores_efficacy_record(
        self, storage_root: Path
    ) -> None:
        _, eff_dir = self._call_build(storage_root)
        assert len(_efficacy_records(eff_dir)) == 1

    def test_efficacy_record_proposal_id_matches_created_proposal(
        self, storage_root: Path
    ) -> None:
        proposal_id, eff_dir = self._call_build(storage_root)
        records = _efficacy_records(eff_dir)
        assert records[0]["proposal_id"] == proposal_id

    def test_efficacy_stored_even_when_no_tests(
        self, storage_root: Path
    ) -> None:
        _, eff_dir = self._call_build(storage_root, tests=[])
        records = _efficacy_records(eff_dir)
        assert len(records) == 1
        assert records[0]["test_case_count"] == 0
        assert records[0]["lift"] == 0.0
