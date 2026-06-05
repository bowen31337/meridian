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
import contextlib
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import uuid

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)

from ._metrics_registry import skill_forge_proposals_total
from ._skill_efficacy import TrajectoryRunner, compare_proposal_trajectories


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


class SkillForgeProposalError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_forge_proposal_failed",
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

    async def forge(self, skill: dict[str, Any], job_type: str) -> str: ...


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
            # best-effort; don't shadow the original error
            with contextlib.suppress(OSError):
                job_file.write_text(json.dumps(job))

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
            raise err from exc

    return run_id


# ---------------------------------------------------------------------------
# Proposal builder helpers
# ---------------------------------------------------------------------------


def _proposal_version_id(
    *,
    skill_id: str,
    instructions: str,
    tools: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    derived_from_session_ids: list[str] | None,
) -> str:
    """Return ``skillver_<sha256>`` over a canonical JSON body with source='forge'."""
    body = {
        "derived_from_session_ids": derived_from_session_ids,
        "instructions": instructions,
        "skill_id": skill_id,
        "source": "forge",
        "source_type": "forge",
        "source_url": None,
        "tests": tests,
        "tools": tools,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"skillver_{digest}"


def _find_primary_user(user_profiles_dir: Path) -> dict[str, Any] | None:
    """Return the primary user profile record, or None if none exists."""
    if not user_profiles_dir.exists():
        return None
    for path in user_profiles_dir.glob("*.json"):
        try:
            record: dict[str, Any] = json.loads(path.read_text())
            if record.get("is_primary"):
                return record
        except (json.JSONDecodeError, OSError):
            continue
    return None


# ---------------------------------------------------------------------------
# Proposal builder
# ---------------------------------------------------------------------------


async def build_skill_version_proposal(
    *,
    result_text: str,
    job: dict[str, Any],
    run_id: str,
    proposals_dir: Path,
    user_profiles_dir: Path,
    notifications_dir: Path,
    audit_log: AuditLog,
    efficacy_dir: Path | None = None,
    trajectory_runner: TrajectoryRunner | None = None,
) -> str:
    """Build an agentskills.io-shaped SkillVersionRecord from forge output.

    Parses ``result_text`` as JSON, builds a SkillVersionRecord with
    ``source='forge'`` and ``derived_from_session_ids`` taken from the job,
    stores it as a ``PROPOSAL`` in ``proposals_dir``, and notifies the primary
    user by writing a notification record to ``notifications_dir``.

    Returns the proposal ID (``skillver_<sha256>``).  On failure: surfaces the
    error message to the caller by raising :class:`SkillForgeProposalError` and
    writes the failure to the audit log before re-raising.
    """
    job_id: str = job["id"]
    skill_id: str = job["skill_id"]
    derived_from_session_ids: list[str] | None = job.get("derived_from_session_ids") or None
    now = _now()
    tracer = get_tracer()
    proposal_id = ""
    proposal_record: dict[str, Any] | None = None

    with tracer.start_as_current_span(
        "skill_forge.build_proposal",
        attributes={
            "skill_forge.job_id": job_id,
            "skill_forge.run_id": run_id,
            "skill_forge.skill_id": skill_id,
        },
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="skill_forge.build_proposal.invocation",
                code="skill_forge_build_proposal",
                timestamp=now,
            ),
        )

        try:
            try:
                forged: dict[str, Any] = json.loads(result_text)
            except json.JSONDecodeError as exc:
                raise SkillForgeProposalError(
                    message=f"Forge result is not valid JSON: {exc}",
                    timestamp=now,
                    cause=exc,
                ) from exc

            instructions: str = forged.get("instructions", "")
            tools: list[dict[str, Any]] = forged.get("tools") or []
            tests: list[dict[str, Any]] = forged.get("tests") or []

            proposal_id = _proposal_version_id(
                skill_id=skill_id,
                instructions=instructions,
                tools=tools,
                tests=tests,
                derived_from_session_ids=derived_from_session_ids,
            )

            proposal_record = {
                "id": proposal_id,
                "skill_id": skill_id,
                "instructions": instructions,
                "tools": tools,
                "tests": tests,
                "source": "forge",
                "source_type": "forge",
                "source_url": None,
                "derived_from_session_ids": derived_from_session_ids,
                "run_id": run_id,
                "job_id": job_id,
                "status": "PROPOSAL",
                "created_at": now,
            }

            proposals_dir.mkdir(parents=True, exist_ok=True)
            (proposals_dir / f"{proposal_id}.json").write_text(json.dumps(proposal_record))

            primary_user = _find_primary_user(user_profiles_dir)
            notified_user_id: str | None = None
            if primary_user is not None:
                notified_user_id = primary_user["id"]
                notification_id = f"notif_{uuid.uuid4().hex}"
                notification_record: dict[str, Any] = {
                    "id": notification_id,
                    "user_id": notified_user_id,
                    "type": "skill_forge.proposal",
                    "proposal_id": proposal_id,
                    "skill_id": skill_id,
                    "run_id": run_id,
                    "job_id": job_id,
                    "created_at": now,
                }
                notifications_dir.mkdir(parents=True, exist_ok=True)
                (notifications_dir / f"{notification_id}.json").write_text(
                    json.dumps(notification_record)
                )

            skill_forge_proposals_total.labels(outcome="proposed").inc()
            span.set_attribute("skill_forge.proposal_id", proposal_id)
            span.set_attribute("skill_forge.build_proposal.success", True)
            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="skill_forge.proposal.created",
                    code="skill_forge_proposal_created",
                    timestamp=_now(),
                    detail={
                        "job_id": job_id,
                        "run_id": run_id,
                        "skill_id": skill_id,
                        "proposal_id": proposal_id,
                        "notified_user_id": notified_user_id,
                    },
                )
            )

        except SkillForgeProposalError as err:
            span.set_attribute("skill_forge.build_proposal.success", False)
            record_error(span, err)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="skill_forge.proposal.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={
                        "job_id": job_id,
                        "run_id": run_id,
                        "skill_id": skill_id,
                        "message": err.message,
                    },
                )
            )
            raise

        except Exception as exc:
            err2 = SkillForgeProposalError(
                message=f"Failed to build skill version proposal for job {job_id!r}: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            span.set_attribute("skill_forge.build_proposal.success", False)
            record_error(span, err2)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="skill_forge.proposal.failed",
                    code=err2.code,
                    timestamp=err2.timestamp,
                    detail={
                        "job_id": job_id,
                        "run_id": run_id,
                        "skill_id": skill_id,
                        "message": err2.message,
                    },
                )
            )
            raise err2 from exc

        # Proposal stored successfully; compute A/B efficacy metric on its test
        # cases.  SkillEfficacyError propagates to the caller with its own audit
        # entry already written by compare_proposal_trajectories.
        assert proposal_record is not None  # guaranteed: we raised on any failure above
        _eff_dir = efficacy_dir if efficacy_dir is not None else proposals_dir.parent / "efficacy"
        await compare_proposal_trajectories(
            proposal=proposal_record,
            efficacy_dir=_eff_dir,
            audit_log=audit_log,
            runner=trajectory_runner,
        )

    return proposal_id


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

                # already logged inside run_skill_forge_job
                with contextlib.suppress(SkillForgeRunError):
                    await run_skill_forge_job(
                        job,
                        jobs_dir=jobs_dir,
                        results_dir=results_dir,
                        provider=_provider,
                        audit_log=audit_log,
                    )

        await asyncio.sleep(check_interval_seconds)
