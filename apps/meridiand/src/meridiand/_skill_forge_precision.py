"""Skill Forge precision metric: proportion of proposals a user approves.

PRD §7.2 targets: ≥50 % at v1 release, climbing to ≥75 % within 90 days
post-v1 as the proposal corpus grows.

On every invocation: emits OTel span ``"skill_forge.precision_metric"`` and
logs a structured audit event.  On failure: records the error to the span,
surfaces the error message to the caller, and writes the failure to the audit
log before re-raising.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


class SkillForgePrecisionError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_forge_precision_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Precision metric
# ---------------------------------------------------------------------------


def compute_precision_metric(
    *,
    proposals_dir: Path,
    activations_dir: Path,
    precision_dir: Path,
    audit_log: AuditLog,
) -> dict[str, Any]:
    """Compute the Skill Forge precision metric.

    Precision = approved_count / total_proposals, where a proposal is
    counted as approved when at least one activation record references it
    via ``skill_version_id`` and has ``status == "active"``.

    Stores the metric record in
    *precision_dir/<metric_id>_precision.json* and returns it.

    On failure: raises :class:`SkillForgePrecisionError` and writes the
    failure to the audit log before re-raising.
    """
    now = _now()
    metric_id = f"skprecision_{uuid.uuid4().hex}"
    tracer = get_tracer()

    with tracer.start_as_current_span(
        "skill_forge.precision_metric",
        attributes={
            "skill_forge.precision.metric_id": metric_id,
        },
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="skill_forge.precision_metric.invocation",
                code="skill_forge_precision_metric",
                timestamp=now,
            ),
        )

        try:
            # Collect all proposal IDs from proposals_dir.
            proposal_ids: set[str] = set()
            if proposals_dir.exists():
                for path in proposals_dir.glob("*.json"):
                    try:
                        record: dict[str, Any] = json.loads(path.read_text())
                        pid = record.get("id")
                        if pid:
                            proposal_ids.add(pid)
                    except (json.JSONDecodeError, OSError):
                        continue

            total_proposals = len(proposal_ids)

            # Collect skill_version_ids from active activation records.
            approved_version_ids: set[str] = set()
            if activations_dir.exists():
                for path in activations_dir.glob("*.json"):
                    try:
                        act: dict[str, Any] = json.loads(path.read_text())
                        if act.get("status") == "active":
                            vid = act.get("skill_version_id")
                            if vid:
                                approved_version_ids.add(vid)
                    except (json.JSONDecodeError, OSError):
                        continue

            approved_count = len(proposal_ids & approved_version_ids)
            precision = approved_count / total_proposals if total_proposals > 0 else 0.0

            metric_record: dict[str, Any] = {
                "id": metric_id,
                "total_proposals": total_proposals,
                "approved_count": approved_count,
                "precision": precision,
                "computed_at": now,
            }

            precision_dir.mkdir(parents=True, exist_ok=True)
            (precision_dir / f"{metric_id}_precision.json").write_text(
                json.dumps(metric_record)
            )

            span.set_attribute(
                "skill_forge.precision.total_proposals", total_proposals
            )
            span.set_attribute(
                "skill_forge.precision.approved_count", approved_count
            )
            span.set_attribute("skill_forge.precision.precision", precision)
            span.set_attribute("skill_forge.precision_metric.success", True)

            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="skill_forge.precision.computed",
                    code="skill_forge_precision_computed",
                    timestamp=_now(),
                    detail={
                        "metric_id": metric_id,
                        "total_proposals": total_proposals,
                        "approved_count": approved_count,
                        "precision": precision,
                    },
                )
            )

        except SkillForgePrecisionError as err:
            span.set_attribute("skill_forge.precision_metric.success", False)
            record_error(span, err)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="skill_forge.precision.compute.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={
                        "metric_id": metric_id,
                        "message": err.message,
                    },
                )
            )
            raise

        except Exception as exc:
            err2 = SkillForgePrecisionError(
                message=f"Failed to compute precision metric: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            span.set_attribute("skill_forge.precision_metric.success", False)
            record_error(span, err2)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="skill_forge.precision.compute.failed",
                    code=err2.code,
                    timestamp=err2.timestamp,
                    detail={
                        "metric_id": metric_id,
                        "message": err2.message,
                    },
                )
            )
            raise err2

    return metric_record
