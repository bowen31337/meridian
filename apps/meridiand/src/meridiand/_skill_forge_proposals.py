"""Skill Forge proposals API endpoints.

POST /v1/x/skill_forge/proposals/{proposal_id}/reject — marks a proposal as
rejected with an audit-logged reason.  Returns 200 on success, 404 if the
proposal does not exist, 409 if the proposal is already promoted.

On every invocation: emits OTel span ``"skill_forge.proposal.reject"`` and
logs a structured audit event.  On failure: records the error to the span,
surfaces the error message to the caller, and writes the failure to the audit
log before re-raising.
"""

from __future__ import annotations

import json
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
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class SkillForgeProposalNotFoundError(MeridianError):
    def __init__(self, *, proposal_id: str, timestamp: str) -> None:
        super().__init__(
            code="skill_forge_proposal_not_found",
            message=f"Proposal '{proposal_id}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class SkillForgeProposalAlreadyPromotedError(MeridianError):
    def __init__(self, *, proposal_id: str, timestamp: str) -> None:
        super().__init__(
            code="skill_forge_proposal_already_promoted",
            message=f"Proposal '{proposal_id}' is already promoted",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 409


class SkillForgeProposalRejectError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_forge_proposal_reject_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class RejectProposalRequest(BaseModel):
    reason: str


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_skill_forge_proposals_router(
    *, audit_log: AuditLog, storage_root: Path
) -> APIRouter:
    router = APIRouter()
    proposals_dir = storage_root / "skill_forge" / "proposals"

    @router.post(
        "/v1/x/skill_forge/proposals/{proposal_id}/reject", status_code=200
    )
    async def reject_proposal(
        proposal_id: str, body: RejectProposalRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "skill_forge.proposal.reject",
            attributes={"skill_forge.proposal_id": proposal_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill_forge.proposal.reject.invocation",
                    code="skill_forge_proposal_reject",
                    timestamp=now,
                ),
            )

            try:
                proposal_file = proposals_dir / f"{proposal_id}.json"
                if not proposal_file.exists():
                    raise SkillForgeProposalNotFoundError(
                        proposal_id=proposal_id, timestamp=now
                    )

                proposal: dict[str, Any] = json.loads(proposal_file.read_text())

                if proposal.get("status") == "PROMOTED":
                    raise SkillForgeProposalAlreadyPromotedError(
                        proposal_id=proposal_id, timestamp=now
                    )

                proposal["status"] = "REJECTED"
                proposal["rejected_at"] = now
                proposal["rejection_reason"] = body.reason
                proposal_file.write_text(json.dumps(proposal))

                span.set_attribute("skill_forge.proposal.reject.success", True)
                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="skill_forge.proposal.rejected",
                        code="skill_forge_proposal_rejected",
                        timestamp=_now(),
                        detail={
                            "proposal_id": proposal_id,
                            "skill_id": proposal.get("skill_id"),
                            "reason": body.reason,
                        },
                    )
                )

            except (
                SkillForgeProposalNotFoundError,
                SkillForgeProposalAlreadyPromotedError,
            ) as err:
                span.set_attribute("skill_forge.proposal.reject.success", False)
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill_forge.proposal.reject.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "proposal_id": proposal_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = SkillForgeProposalRejectError(
                    message=f"Failed to reject proposal '{proposal_id}': {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                span.set_attribute("skill_forge.proposal.reject.success", False)
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill_forge.proposal.reject.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "proposal_id": proposal_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=proposal, status_code=200)

    return router
