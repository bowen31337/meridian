"""Skill Forge background worker.

Processes pending skill-forge jobs from storage_root/skill_forge/jobs/.
Each job consumes one token from the rate-limited model budget.  The budget
is a sliding per-minute window controlled by ``max_invocations_per_minute``.

The worker can be disabled by setting ``SkillForgeConfig.enabled = false`` in
the daemon config; throttled by lowering ``max_invocations_per_minute`` or
raising ``check_interval_seconds``.

On every invocation: emits OTel span ``"skill_forge.run"`` and logs a
structured audit event.  On failure: records the error to the span, surfaces
the error message in the job record (so callers polling the job can observe
it), and writes the failure to the audit log before re-raising.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class SkillForgeRunError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_forge_run_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Model provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SkillForgeProvider(Protocol):
    """Processes a skill-forge job; returns the forged result as a string."""

    async def forge(self, skill: dict[str, Any], job_type: str) -> str:
        ...


class NoopSkillForgeProvider:
    """No-op provider — returns an empty string; used when no provider is wired."""

    async def forge(self, skill: dict[str, Any], job_type: str) -> str:
        return ""


# ---------------------------------------------------------------------------
# Rate limiter (sliding per-minute window)
# ---------------------------------------------------------------------------


class _RateLimiter:
    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._count = 0
        self._window_start = datetime.now(UTC)

    def try_acquire(self) -> bool:
        """Consume one token if budget remains; return False when exhausted."""
        now = datetime.now(UTC)
        if (now - self._window_start).total_seconds() >= 60.0:
            self._count = 0
            self._window_start = now
        if self._count >= self._max:
            return False
        self._count += 1
        return True


# ---------------------------------------------------------------------------
# Single-job execution
# ---------------------------------------------------------------------------


async def run_skill_forge_job(
    job: dict[str, Any],
    *,
    jobs_dir: Path,
    results_dir: Path,
    provider: SkillForgeProvider,
    audit_log: AuditLog,
) -> str:
    """Forge one pending job: invoke the provider, write the result, update job status.

    Returns the ``run_id``.  Raises :class:`SkillForgeRunError` on any failure;
    the error is recorded to the span, surfaced in the job record's ``error``
    field (so callers can observe it via the job record), and written to the
    audit log before re-raising.
    """
    job_id: str = job["id"]
    skill_id: str = job["skill_id"]
    job_type: str = job["job_type"]
    run_id = f"sfrun_{uuid.uuid4().hex}"
    now = _now()
    tracer = get_tracer()

    with tracer.start_as_current_span(
        "skill_forge.run",
        attributes={
            "skill_forge.job_id": job_id,
            "skill_forge.run_id": run_id,
            "skill_forge.skill_id": skill_id,
            "skill_forge.job_type": job_type,
        },
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="skill_forge.run.invocation",
                code="skill_forge_run",
                timestamp=now,
            ),
        )

        job_file = jobs_dir / f"{job_id}.json"

        try:
            # Mark the job as running so concurrent workers skip it.
            job["status"] = "running"
            job["started_at"] = now
            job_file.write_text(json.dumps(job))

            skill: dict[str, Any] = job.get("skill") or {}
            result_text = await provider.forge(skill, job_type)

            results_dir.mkdir(parents=True, exist_ok=True)
            result_record: dict[str, Any] = {
                "run_id": run_id,
                "job_id": job_id,
                "skill_id": skill_id,
                "job_type": job_type,
                "result": result_text,
                "completed_at": _now(),
            }
            (results_dir / f"{job_id}.json").write_text(json.dumps(result_record))

            job["status"] = "completed"
            job["completed_at"] = _now()
            job["run_id"] = run_id
            job_file.write_text(json.dumps(job))

            span.set_attribute("skill_forge.run.success", True)
            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="skill_forge.ran",
                    code="skill_forge_ran",
                    timestamp=_now(),
                    detail={
                        "job_id": job_id,
                        "run_id": run_id,
                        "skill_id": skill_id,
                        "job_type": job_type,
                    },
                )
            )

        except Exception as exc:
            err = SkillForgeRunError(
                message=f"Failed to run skill-forge job {job_id!r}: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            span.set_attribute("skill_forge.run.success", False)
            record_error(span, err)

            # Surface the error in the job record so the caller can observe it.
            job["status"] = "failed"
            job["failed_at"] = err.timestamp
            job["error"] = err.message
            try:
                job_file.write_text(json.dumps(job))
            except OSError:
                pass  # best-effort; don't shadow the original error

            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="skill_forge.run.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={
                        "job_id": job_id,
                        "run_id": run_id,
                        "skill_id": skill_id,
                        "job_type": job_type,
                        "message": err.message,
                    },
                )
            )
            raise err

    return run_id


# ---------------------------------------------------------------------------
# Background worker loop
# ---------------------------------------------------------------------------


def _load_job(job_file: Path) -> dict[str, Any] | None:
    try:
        return json.loads(job_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


async def run_skill_forge_loop(
    storage_root: Path,
    audit_log: AuditLog,
    *,
    provider: SkillForgeProvider | None = None,
    max_invocations_per_minute: int = 10,
    check_interval_seconds: float = 5.0,
) -> None:
    """Background worker loop for Skill Forge job processing.

    Wakes every ``check_interval_seconds`` seconds and processes pending jobs
    from ``storage_root/skill_forge/jobs/``.  At most
    ``max_invocations_per_minute`` model invocations are issued per 60-second
    window; when the budget is exhausted, remaining pending jobs are deferred
    to the next iteration.

    On failure: the error is surfaced in the job record and written to the
    audit log; the loop continues processing remaining jobs.
    """
    _provider: SkillForgeProvider = provider if provider is not None else NoopSkillForgeProvider()
    _limiter = _RateLimiter(max_invocations_per_minute)
    jobs_dir = storage_root / "skill_forge" / "jobs"
    results_dir = storage_root / "skill_forge" / "results"

    while True:
        if jobs_dir.exists():
            for job_file in sorted(jobs_dir.glob("sfjob_*.json")):
                job = _load_job(job_file)
                if job is None:
                    continue
                if job.get("status") != "pending":
                    continue

                if not _limiter.try_acquire():
                    break  # budget exhausted; defer remaining jobs to next tick

                try:
                    await run_skill_forge_job(
                        job,
                        jobs_dir=jobs_dir,
                        results_dir=results_dir,
                        provider=_provider,
                        audit_log=audit_log,
                    )
                except SkillForgeRunError:
                    pass  # already logged inside run_skill_forge_job

        await asyncio.sleep(check_interval_seconds)
