"""Skill efficacy metric: A/B trajectory comparison.

Compares model trajectories with vs without the skill applied to the
proposal's test cases.  Records a pass-rate lift metric per proposal for
review.

On every invocation: emits OTel span ``"skill_efficacy.compare_trajectories"``
and logs a structured audit event.  On failure: records the error to the span,
surfaces the error message to the caller, and writes the failure to the audit
log before re-raising.
"""

from __future__ import annotations

from datetime import UTC, datetime
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


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class SkillEfficacyError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_efficacy_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Trajectory runner protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TrajectoryRunner(Protocol):
    """Runs one test case with or without the skill; returns True on success."""

    async def run(
        self,
        test_case: dict[str, Any],
        *,
        skill_instructions: str | None,
    ) -> bool: ...


class NoopTrajectoryRunner:
    """No-op runner — always returns False; used when no runner is wired."""

    async def run(
        self,
        test_case: dict[str, Any],
        *,
        skill_instructions: str | None,
    ) -> bool:
        return False


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------


async def compare_proposal_trajectories(
    *,
    proposal: dict[str, Any],
    efficacy_dir: Path,
    audit_log: AuditLog,
    runner: TrajectoryRunner | None = None,
) -> dict[str, Any]:
    """Compare model trajectories with vs without the skill on proposal test cases.

    For each test case in *proposal["tests"]*, runs two arms:
    - **without_skill**: baseline trajectory (skill_instructions=None)
    - **with_skill**: trajectory with the proposal's instructions applied

    Computes ``lift`` = pass_rate_with_skill − pass_rate_without_skill and
    stores the metric record in *efficacy_dir/<proposal_id>_efficacy.json*.
    Returns the metric record.

    On failure: surfaces the error to the caller by raising
    :class:`SkillEfficacyError` and writes the failure to the audit log.
    """
    _runner: TrajectoryRunner = runner if runner is not None else NoopTrajectoryRunner()
    proposal_id: str = proposal["id"]
    skill_id: str = proposal["skill_id"]
    instructions: str = proposal.get("instructions", "")
    tests: list[dict[str, Any]] = proposal.get("tests") or []
    now = _now()
    metric_id = f"skefficacy_{uuid.uuid4().hex}"
    tracer = get_tracer()

    with tracer.start_as_current_span(
        "skill_efficacy.compare_trajectories",
        attributes={
            "skill_efficacy.proposal_id": proposal_id,
            "skill_efficacy.skill_id": skill_id,
            "skill_efficacy.metric_id": metric_id,
            "skill_efficacy.test_case_count": len(tests),
        },
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="skill_efficacy.compare_trajectories.invocation",
                code="skill_efficacy_compare_trajectories",
                timestamp=now,
            ),
        )

        try:
            case_results: list[dict[str, Any]] = []
            for test_case in tests:
                test_name: str = test_case.get("name", "")
                passed_without = await _runner.run(test_case, skill_instructions=None)
                passed_with = await _runner.run(test_case, skill_instructions=instructions)
                case_results.append(
                    {
                        "test_name": test_name,
                        "passed_without_skill": passed_without,
                        "passed_with_skill": passed_with,
                    }
                )

            n = len(case_results)
            pass_rate_without = (
                sum(1 for c in case_results if c["passed_without_skill"]) / n if n > 0 else 0.0
            )
            pass_rate_with = (
                sum(1 for c in case_results if c["passed_with_skill"]) / n if n > 0 else 0.0
            )
            lift = pass_rate_with - pass_rate_without

            metric_record: dict[str, Any] = {
                "id": metric_id,
                "proposal_id": proposal_id,
                "skill_id": skill_id,
                "test_case_count": n,
                "pass_rate_without_skill": pass_rate_without,
                "pass_rate_with_skill": pass_rate_with,
                "lift": lift,
                "case_results": case_results,
                "created_at": now,
            }

            efficacy_dir.mkdir(parents=True, exist_ok=True)
            (efficacy_dir / f"{proposal_id}_efficacy.json").write_text(json.dumps(metric_record))

            span.set_attribute("skill_efficacy.pass_rate_without_skill", pass_rate_without)
            span.set_attribute("skill_efficacy.pass_rate_with_skill", pass_rate_with)
            span.set_attribute("skill_efficacy.lift", lift)
            span.set_attribute("skill_efficacy.compare_trajectories.success", True)

            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="skill_efficacy.compared",
                    code="skill_efficacy_compared",
                    timestamp=_now(),
                    detail={
                        "metric_id": metric_id,
                        "proposal_id": proposal_id,
                        "skill_id": skill_id,
                        "test_case_count": n,
                        "pass_rate_without_skill": pass_rate_without,
                        "pass_rate_with_skill": pass_rate_with,
                        "lift": lift,
                    },
                )
            )

        except SkillEfficacyError as err:
            span.set_attribute("skill_efficacy.compare_trajectories.success", False)
            record_error(span, err)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="skill_efficacy.compare.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={
                        "metric_id": metric_id,
                        "proposal_id": proposal_id,
                        "skill_id": skill_id,
                        "message": err.message,
                    },
                )
            )
            raise

        except Exception as exc:
            err2 = SkillEfficacyError(
                message=(f"Failed to compare trajectories for proposal {proposal_id!r}: {exc}"),
                timestamp=_now(),
                cause=exc,
            )
            span.set_attribute("skill_efficacy.compare_trajectories.success", False)
            record_error(span, err2)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="skill_efficacy.compare.failed",
                    code=err2.code,
                    timestamp=err2.timestamp,
                    detail={
                        "metric_id": metric_id,
                        "proposal_id": proposal_id,
                        "skill_id": skill_id,
                        "message": err2.message,
                    },
                )
            )
            raise err2 from exc

    return metric_record
