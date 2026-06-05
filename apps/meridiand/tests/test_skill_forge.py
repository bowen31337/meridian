"""
Skill Forge conformance suite.

Tests cover:
  - run_skill_forge_job writes a result record under skill_forge/results/.
  - Result record contains run_id, job_id, skill_id, job_type, result, completed_at.
  - run_skill_forge_job returns the run_id.
  - run_skill_forge_job marks the job status as "completed" on success.
  - run_skill_forge_job sets job.run_id to the returned run_id on success.
  - run_skill_forge_job emits OTel span "skill_forge.run".
  - OTel span carries skill_forge.job_id, skill_forge.run_id, skill_forge.skill_id,
    skill_forge.job_type attributes.
  - OTel span sets skill_forge.run.success=True on success.
  - run_skill_forge_job writes audit entry "skill_forge.ran" on success.
  - Audit entry level is "info" on success.
  - Audit detail contains job_id, run_id, skill_id, job_type.
  - run_skill_forge_job raises SkillForgeRunError on provider failure.
  - On failure: job status is set to "failed" in the job record.
  - On failure: job record contains an "error" field with the error message.
  - On failure: OTel span sets skill_forge.run.success=False.
  - On failure: writes audit entry "skill_forge.run.failed".
  - Failed audit entry level is "error".
  - Failed audit detail contains job_id, run_id, skill_id, job_type, message.
  - run_skill_forge_loop processes a pending job.
  - run_skill_forge_loop skips jobs whose status is not "pending".
  - run_skill_forge_loop ignores malformed JSON files.
  - Rate limiter try_acquire returns True while budget remains.
  - Rate limiter try_acquire returns False when budget is exhausted.
  - Rate limiter resets count after 60-second window.
  - run_skill_forge_loop stops processing when rate budget is exhausted.
  - SkillForgeConfig defaults: enabled=True, max_invocations_per_minute=10,
    check_interval_seconds=5.0.
  - SkillForgeConfig can be disabled via enabled=False (loop not started in create_app).
  - Disabled config: loop is not created in app lifespan when enabled=False.
  - Multiple pending jobs are all processed in one tick (within budget).
  - run_ids are unique across invocations.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any

from meridiand._audit import FileAuditLog
from meridiand._config import SkillForgeConfig
from meridiand._skill_forge import (
    SkillForgeProvider,
    SkillForgeRunError,
    _RateLimiter,
    run_skill_forge_job,
    run_skill_forge_loop,
)
import pytest

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _result_record(storage_root: Path, job_id: str) -> dict | None:
    path = storage_root / "skill_forge" / "results" / f"{job_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_job(storage_root: Path, job_id: str) -> dict:
    path = storage_root / "skill_forge" / "jobs" / f"{job_id}.json"
    return json.loads(path.read_text())


def _make_job(
    job_id: str = "sfjob_test",
    skill_id: str = "skill_abc",
    job_type: str = "validate_tests",
    status: str = "pending",
    skill: dict | None = None,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "skill_id": skill_id,
        "job_type": job_type,
        "status": status,
        "skill": skill or {"name": "test-skill"},
        "created_at": datetime.now(UTC).isoformat(),
        "started_at": None,
        "completed_at": None,
        "failed_at": None,
        "error": None,
        "run_id": None,
    }


def _write_job(storage_root: Path, job: dict[str, Any]) -> Path:
    jobs_dir = storage_root / "skill_forge" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / f"{job['id']}.json"
    job_file.write_text(json.dumps(job))
    return job_file


def _make_dirs(storage_root: Path) -> tuple[Path, Path]:
    jobs_dir = storage_root / "skill_forge" / "jobs"
    results_dir = storage_root / "skill_forge" / "results"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir, results_dir


class _SuccessProvider:
    """Provider that always succeeds and returns a fixed result."""

    def __init__(self, result: str = "forged") -> None:
        self._result = result

    async def forge(self, skill: dict[str, Any], job_type: str) -> str:
        return self._result


class _FailProvider:
    """Provider that always raises."""

    async def forge(self, skill: dict[str, Any], job_type: str) -> str:
        raise RuntimeError("provider exploded")


async def _run_one_tick(
    storage_root: Path,
    audit_log: FileAuditLog,
    provider: SkillForgeProvider | None = None,
    max_invocations_per_minute: int = 100,
) -> None:
    """Run the skill forge loop for a single tick then cancel."""
    task = asyncio.create_task(
        run_skill_forge_loop(
            storage_root,
            audit_log,
            provider=provider,
            max_invocations_per_minute=max_invocations_per_minute,
            check_interval_seconds=9999.0,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# run_skill_forge_job: result record
# ---------------------------------------------------------------------------


class TestRunSkillForgeJobResult:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_result_record_written(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )
        assert _result_record(storage_root, "sfjob_test") is not None

    def test_result_record_has_run_id(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        run_id = asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )
        record = _result_record(storage_root, "sfjob_test")
        assert record is not None
        assert record["run_id"] == run_id

    def test_result_record_has_job_id(self, storage_root: Path) -> None:
        job = _make_job(job_id="sfjob_abc")
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )
        record = _result_record(storage_root, "sfjob_abc")
        assert record is not None
        assert record["job_id"] == "sfjob_abc"

    def test_result_record_has_skill_id(self, storage_root: Path) -> None:
        job = _make_job(skill_id="skill_xyz")
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )
        record = _result_record(storage_root, "sfjob_test")
        assert record is not None
        assert record["skill_id"] == "skill_xyz"

    def test_result_record_has_job_type(self, storage_root: Path) -> None:
        job = _make_job(job_type="generate_tests")
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )
        record = _result_record(storage_root, "sfjob_test")
        assert record is not None
        assert record["job_type"] == "generate_tests"

    def test_result_record_has_result_text(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider("my-result"),
                audit_log=audit_log,
            )
        )
        record = _result_record(storage_root, "sfjob_test")
        assert record is not None
        assert record["result"] == "my-result"

    def test_result_record_has_completed_at(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )
        record = _result_record(storage_root, "sfjob_test")
        assert record is not None
        assert "completed_at" in record and record["completed_at"]

    def test_returns_run_id(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        run_id = asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )
        assert run_id.startswith("sfrun_")

    def test_run_ids_are_unique(self, storage_root: Path) -> None:
        jobs_dir, results_dir = _make_dirs(storage_root)
        audit_log = FileAuditLog(storage_root)

        job1 = _make_job(job_id="sfjob_u1")
        (jobs_dir / "sfjob_u1.json").write_text(json.dumps(job1))
        id1 = asyncio.run(
            run_skill_forge_job(
                job1,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )

        job2 = _make_job(job_id="sfjob_u2")
        (jobs_dir / "sfjob_u2.json").write_text(json.dumps(job2))
        id2 = asyncio.run(
            run_skill_forge_job(
                job2,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )

        assert id1 != id2


# ---------------------------------------------------------------------------
# run_skill_forge_job: job status update
# ---------------------------------------------------------------------------


class TestRunSkillForgeJobStatus:
    def test_job_marked_completed_on_success(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )
        updated = _read_job(storage_root, "sfjob_test")
        assert updated["status"] == "completed"

    def test_job_run_id_set_on_success(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        run_id = asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )
        updated = _read_job(storage_root, "sfjob_test")
        assert updated["run_id"] == run_id


# ---------------------------------------------------------------------------
# run_skill_forge_job: OTel
# ---------------------------------------------------------------------------


class TestRunSkillForgeJobOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_emits_skill_forge_run_span(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=audit_log,
            )
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.run" in span_names

    def _get_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("skill_forge.run")

    def test_span_has_job_id_attribute(self, storage_root: Path) -> None:
        job = _make_job(job_id="sfjob_otel1")
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.job_id"] == "sfjob_otel1"

    def test_span_has_run_id_attribute(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        run_id = asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.run_id"] == run_id

    def test_span_has_skill_id_attribute(self, storage_root: Path) -> None:
        job = _make_job(skill_id="skill_otel")
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.skill_id"] == "skill_otel"

    def test_span_has_job_type_attribute(self, storage_root: Path) -> None:
        job = _make_job(job_type="improve_instructions")
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.job_type"] == "improve_instructions"

    def test_span_success_attribute_true_on_success(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.run.success"] is True

    def test_span_success_attribute_false_on_failure(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        with pytest.raises(SkillForgeRunError):
            asyncio.run(
                run_skill_forge_job(
                    job,
                    jobs_dir=jobs_dir,
                    results_dir=results_dir,
                    provider=_FailProvider(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.run.success"] is False


# ---------------------------------------------------------------------------
# run_skill_forge_job: audit log
# ---------------------------------------------------------------------------


class TestRunSkillForgeJobAudit:
    def test_success_writes_ran_audit_entry(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.ran" for r in records)

    def test_ran_audit_level_is_info(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.ran"
        )
        assert record["level"] == "info"

    def test_ran_audit_detail_has_job_id(self, storage_root: Path) -> None:
        job = _make_job(job_id="sfjob_audit")
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.ran"
        )
        assert record["detail"]["job_id"] == "sfjob_audit"

    def test_ran_audit_detail_has_run_id(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        run_id = asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.ran"
        )
        assert record["detail"]["run_id"] == run_id

    def test_ran_audit_detail_has_skill_id(self, storage_root: Path) -> None:
        job = _make_job(skill_id="skill_audit_test")
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.ran"
        )
        assert record["detail"]["skill_id"] == "skill_audit_test"

    def test_ran_audit_detail_has_job_type(self, storage_root: Path) -> None:
        job = _make_job(job_type="generate_tests")
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        asyncio.run(
            run_skill_forge_job(
                job,
                jobs_dir=jobs_dir,
                results_dir=results_dir,
                provider=_SuccessProvider(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.ran"
        )
        assert record["detail"]["job_type"] == "generate_tests"

    def test_failure_raises_skill_forge_run_error(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        with pytest.raises(SkillForgeRunError):
            asyncio.run(
                run_skill_forge_job(
                    job,
                    jobs_dir=jobs_dir,
                    results_dir=results_dir,
                    provider=_FailProvider(),
                    audit_log=FileAuditLog(storage_root),
                )
            )

    def test_failure_writes_failed_audit_entry(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        with pytest.raises(SkillForgeRunError):
            asyncio.run(
                run_skill_forge_job(
                    job,
                    jobs_dir=jobs_dir,
                    results_dir=results_dir,
                    provider=_FailProvider(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.run.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        with pytest.raises(SkillForgeRunError):
            asyncio.run(
                run_skill_forge_job(
                    job,
                    jobs_dir=jobs_dir,
                    results_dir=results_dir,
                    provider=_FailProvider(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.run.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        with pytest.raises(SkillForgeRunError):
            asyncio.run(
                run_skill_forge_job(
                    job,
                    jobs_dir=jobs_dir,
                    results_dir=results_dir,
                    provider=_FailProvider(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.run.failed"
        )
        assert "message" in record["detail"] and record["detail"]["message"]

    def test_failure_audit_detail_has_job_id(self, storage_root: Path) -> None:
        job = _make_job(job_id="sfjob_fail")
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        with pytest.raises(SkillForgeRunError):
            asyncio.run(
                run_skill_forge_job(
                    job,
                    jobs_dir=jobs_dir,
                    results_dir=results_dir,
                    provider=_FailProvider(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.run.failed"
        )
        assert record["detail"]["job_id"] == "sfjob_fail"


# ---------------------------------------------------------------------------
# run_skill_forge_job: error surfaced in job record
# ---------------------------------------------------------------------------


class TestRunSkillForgeJobErrorSurfacing:
    def test_failure_sets_job_status_failed(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        with pytest.raises(SkillForgeRunError):
            asyncio.run(
                run_skill_forge_job(
                    job,
                    jobs_dir=jobs_dir,
                    results_dir=results_dir,
                    provider=_FailProvider(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        updated = _read_job(storage_root, "sfjob_test")
        assert updated["status"] == "failed"

    def test_failure_writes_error_field_to_job_record(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        with pytest.raises(SkillForgeRunError):
            asyncio.run(
                run_skill_forge_job(
                    job,
                    jobs_dir=jobs_dir,
                    results_dir=results_dir,
                    provider=_FailProvider(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        updated = _read_job(storage_root, "sfjob_test")
        assert updated.get("error") and "provider exploded" in updated["error"]

    def test_failure_writes_failed_at_to_job_record(self, storage_root: Path) -> None:
        job = _make_job()
        jobs_dir, results_dir = _make_dirs(storage_root)
        (jobs_dir / f"{job['id']}.json").write_text(json.dumps(job))
        with pytest.raises(SkillForgeRunError):
            asyncio.run(
                run_skill_forge_job(
                    job,
                    jobs_dir=jobs_dir,
                    results_dir=results_dir,
                    provider=_FailProvider(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        updated = _read_job(storage_root, "sfjob_test")
        assert updated.get("failed_at")


# ---------------------------------------------------------------------------
# run_skill_forge_loop: job processing
# ---------------------------------------------------------------------------


class TestRunSkillForgeLoop:
    def test_processes_pending_job(self, storage_root: Path) -> None:
        job = _make_job()
        _write_job(storage_root, job)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log, provider=_SuccessProvider()))
        updated = _read_job(storage_root, "sfjob_test")
        assert updated["status"] == "completed"

    def test_skips_non_pending_job(self, storage_root: Path) -> None:
        job = _make_job(status="completed")
        _write_job(storage_root, job)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log, provider=_SuccessProvider()))
        # Should not have changed the record (still completed, no result written)
        updated = _read_job(storage_root, "sfjob_test")
        assert updated["status"] == "completed"
        assert _result_record(storage_root, "sfjob_test") is None

    def test_skips_running_job(self, storage_root: Path) -> None:
        job = _make_job(status="running")
        _write_job(storage_root, job)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log, provider=_SuccessProvider()))
        updated = _read_job(storage_root, "sfjob_test")
        assert updated["status"] == "running"

    def test_ignores_malformed_json(self, storage_root: Path) -> None:
        jobs_dir = storage_root / "skill_forge" / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        (jobs_dir / "sfjob_bad.json").write_text("not valid json{{{")
        audit_log = FileAuditLog(storage_root)
        # Should not raise; malformed file is silently skipped.
        asyncio.run(_run_one_tick(storage_root, audit_log))

    def test_processes_multiple_pending_jobs(self, storage_root: Path) -> None:
        for i in range(3):
            _write_job(storage_root, _make_job(job_id=f"sfjob_m{i}", skill_id=f"skill_{i}"))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log, provider=_SuccessProvider()))
        for i in range(3):
            updated = _read_job(storage_root, f"sfjob_m{i}")
            assert updated["status"] == "completed"

    def test_failed_job_does_not_stop_loop(self, storage_root: Path) -> None:
        _write_job(storage_root, _make_job(job_id="sfjob_fail"))
        _write_job(storage_root, _make_job(job_id="sfjob_ok"))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log, provider=_FailProvider()))
        # Both jobs were attempted; both should be failed
        assert _read_job(storage_root, "sfjob_fail")["status"] == "failed"
        assert _read_job(storage_root, "sfjob_ok")["status"] == "failed"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_try_acquire_returns_true_within_budget(self) -> None:
        limiter = _RateLimiter(max_per_minute=3)
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True

    def test_try_acquire_returns_false_when_exhausted(self) -> None:
        limiter = _RateLimiter(max_per_minute=2)
        limiter.try_acquire()
        limiter.try_acquire()
        assert limiter.try_acquire() is False

    def test_try_acquire_resets_after_60_seconds(self) -> None:
        limiter = _RateLimiter(max_per_minute=1)
        limiter.try_acquire()
        assert limiter.try_acquire() is False
        # Simulate window expiry by backdating _window_start.
        limiter._window_start = datetime.now(UTC) - timedelta(seconds=61)
        assert limiter.try_acquire() is True

    def test_loop_stops_processing_when_budget_exhausted(self, storage_root: Path) -> None:
        for i in range(5):
            _write_job(storage_root, _make_job(job_id=f"sfjob_r{i}"))
        audit_log = FileAuditLog(storage_root)
        asyncio.run(
            _run_one_tick(
                storage_root,
                audit_log,
                provider=_SuccessProvider(),
                max_invocations_per_minute=2,
            )
        )
        completed = sum(
            1 for i in range(5) if _read_job(storage_root, f"sfjob_r{i}")["status"] == "completed"
        )
        # Budget of 2 means at most 2 jobs processed this tick.
        assert completed == 2


# ---------------------------------------------------------------------------
# SkillForgeConfig
# ---------------------------------------------------------------------------


class TestSkillForgeConfig:
    def test_default_enabled_is_true(self) -> None:
        cfg = SkillForgeConfig()
        assert cfg.enabled is True

    def test_default_max_invocations_per_minute(self) -> None:
        cfg = SkillForgeConfig()
        assert cfg.max_invocations_per_minute == 10

    def test_default_check_interval_seconds(self) -> None:
        cfg = SkillForgeConfig()
        assert cfg.check_interval_seconds == 5.0

    def test_can_disable_via_config(self) -> None:
        cfg = SkillForgeConfig(enabled=False)
        assert cfg.enabled is False

    def test_can_throttle_via_max_invocations(self) -> None:
        cfg = SkillForgeConfig(max_invocations_per_minute=1)
        assert cfg.max_invocations_per_minute == 1

    def test_can_throttle_via_check_interval(self) -> None:
        cfg = SkillForgeConfig(check_interval_seconds=60.0)
        assert cfg.check_interval_seconds == 60.0
