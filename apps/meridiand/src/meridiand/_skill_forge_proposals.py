"""Skill Forge proposals API endpoints.

GET /v1/x/skill_forge/proposals — lists quarantined proposals (status=PROPOSAL)
with trajectory provenance and optional A/B efficacy comparison.  Supports
cursor-based pagination via ``cursor`` and ``limit`` query params.  Pass
``include_efficacy=true`` to attach the stored A/B metric record to each item.
Returns 200 with ``{items, next_cursor, limit}`` on success.

POST /v1/x/skill_forge/proposals/{proposal_id}/approve — promotes a quarantined
proposal to an active SkillVersion; recomputes the content hash; available to
Agents that opt in.  Returns 200 with the new SkillVersion record on success,
404 if the proposal does not exist, 409 if the proposal is already promoted.

POST /v1/x/skill_forge/proposals/{proposal_id}/reject — marks a proposal as
rejected with an audit-logged reason.  Returns 200 on success, 404 if the
proposal does not exist, 409 if the proposal is already promoted.

On every invocation: emits an OTel span and logs a structured audit event.
On failure: records the error to the span, surfaces the error message to the
caller, and writes the failure to the audit log before re-raising.
"""

from __future__ import annotations

import hashlib
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
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ._metrics_registry import skill_forge_proposals_total
from ._pagination import (
    DEFAULT_PAGE_SIZE,
    CursorDecodeError,
    apply_cursor_filter,
    decode_cursor,
    make_cursor_page,
)


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


class SkillForgeProposalListError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_forge_proposal_list_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


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


class SkillForgeProposalApproveError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_forge_proposal_approve_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Content hash (mirrors _skill_forge._proposal_version_id)
# ---------------------------------------------------------------------------


def _forge_version_id(
    *,
    skill_id: str,
    instructions: str,
    tools: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    derived_from_session_ids: list[str] | None,
) -> str:
    """Return ``skillver_<sha256>`` over the canonical forge-proposal JSON body."""
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
    efficacy_dir = storage_root / "skill_forge" / "efficacy"
    versions_dir = storage_root / "skill_versions"
    skills_dir = storage_root / "skills"

    @router.get("/v1/x/skill_forge/proposals", status_code=200)
    async def list_proposals(
        cursor: str | None = Query(default=None),
        limit: int = Query(default=DEFAULT_PAGE_SIZE),
        include_efficacy: bool = Query(default=False),
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "skill_forge.proposal.list",
            attributes={"skill_forge.include_efficacy": include_efficacy},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill_forge.proposal.list.invocation",
                    code="skill_forge_proposal_list",
                    timestamp=now,
                ),
            )

            try:
                all_proposals: list[dict[str, Any]] = []
                if proposals_dir.exists():
                    for path in proposals_dir.glob("*.json"):
                        record = json.loads(path.read_text())
                        if record.get("status") == "PROPOSAL":
                            all_proposals.append(record)

                all_proposals.sort(
                    key=lambda r: (r.get("created_at", ""), r.get("id", "")),
                    reverse=True,
                )

                if cursor is not None:
                    c_created_at, c_id = decode_cursor(cursor, timestamp=now)
                    all_proposals = apply_cursor_filter(all_proposals, c_created_at, c_id)

                page, next_cursor = make_cursor_page(all_proposals, limit)

                if include_efficacy:
                    enriched: list[dict[str, Any]] = []
                    for proposal in page:
                        proposal_id = proposal.get("id", "")
                        efficacy_file = efficacy_dir / f"{proposal_id}_efficacy.json"
                        efficacy: dict[str, Any] | None = None
                        if efficacy_file.exists():
                            efficacy = json.loads(efficacy_file.read_text())
                        enriched.append({**proposal, "efficacy": efficacy})
                    page = enriched

                span.set_attribute("skill_forge.proposal.list.count", len(page))
                span.set_attribute("skill_forge.proposal.list.success", True)

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="skill_forge.proposal.listed",
                        code="skill_forge_proposal_listed",
                        timestamp=_now(),
                        detail={
                            "count": len(page),
                            "include_efficacy": include_efficacy,
                        },
                    )
                )

            except CursorDecodeError as err:
                span.set_attribute("skill_forge.proposal.list.success", False)
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill_forge.proposal.list.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message},
                    )
                )
                raise

            except Exception as exc:
                err2 = SkillForgeProposalListError(
                    message=f"Failed to list proposals: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                span.set_attribute("skill_forge.proposal.list.success", False)
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill_forge.proposal.list.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"message": err2.message},
                    )
                )
                raise err2

        response_headers: dict[str, str] = {}
        if next_cursor is not None:
            response_headers["X-Next-Cursor"] = next_cursor

        return JSONResponse(
            content={"items": page, "next_cursor": next_cursor, "limit": limit},
            status_code=200,
            headers=response_headers,
        )

    @router.post(
        "/v1/x/skill_forge/proposals/{proposal_id}/approve", status_code=200
    )
    async def approve_proposal(proposal_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "skill_forge.proposal.approve",
            attributes={"skill_forge.proposal_id": proposal_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill_forge.proposal.approve.invocation",
                    code="skill_forge_proposal_approve",
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

                skill_id: str = proposal["skill_id"]
                instructions: str = proposal["instructions"]
                tools: list[dict[str, Any]] = proposal.get("tools", [])
                tests: list[dict[str, Any]] = proposal.get("tests", [])
                derived_from_session_ids = proposal.get("derived_from_session_ids")

                version_id = _forge_version_id(
                    skill_id=skill_id,
                    instructions=instructions,
                    tools=tools,
                    tests=tests,
                    derived_from_session_ids=derived_from_session_ids,
                )

                versions_dir.mkdir(parents=True, exist_ok=True)
                max_ver = 0
                for vpath in versions_dir.glob("*.json"):
                    try:
                        vr = json.loads(vpath.read_text())
                        if vr.get("skill_id") == skill_id:
                            max_ver = max(max_ver, vr.get("version_number", 0))
                    except Exception:
                        pass
                version_number = max_ver + 1

                version_record: dict[str, Any] = {
                    "id": version_id,
                    "skill_id": skill_id,
                    "version_number": version_number,
                    "instructions": instructions,
                    "tools": tools,
                    "tests": tests,
                    "created_at": now,
                    "source_type": "forge",
                    "source_url": None,
                    "source": "forge",
                    "derived_from_session_ids": derived_from_session_ids,
                }
                (versions_dir / f"{version_id}.json").write_text(
                    json.dumps(version_record)
                )

                skill_file = skills_dir / f"{skill_id}.json"
                if skill_file.exists():
                    skill_record = json.loads(skill_file.read_text())
                    skill_record["version"] = version_record
                    skill_file.write_text(json.dumps(skill_record))

                proposal["status"] = "PROMOTED"
                proposal["promoted_at"] = now
                proposal["promoted_version_id"] = version_id
                proposal_file.write_text(json.dumps(proposal))

                skill_forge_proposals_total.labels(outcome="approved").inc()
                span.set_attribute("skill_forge.proposal.approve.success", True)
                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="skill_forge.proposal.approved",
                        code="skill_forge_proposal_approved",
                        timestamp=_now(),
                        detail={
                            "proposal_id": proposal_id,
                            "skill_id": skill_id,
                            "version_id": version_id,
                        },
                    )
                )

            except (
                SkillForgeProposalNotFoundError,
                SkillForgeProposalAlreadyPromotedError,
            ) as err:
                span.set_attribute("skill_forge.proposal.approve.success", False)
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill_forge.proposal.approve.failed",
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
                err2 = SkillForgeProposalApproveError(
                    message=f"Failed to approve proposal '{proposal_id}': {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                span.set_attribute("skill_forge.proposal.approve.success", False)
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill_forge.proposal.approve.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "proposal_id": proposal_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=version_record, status_code=200)

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

                skill_forge_proposals_total.labels(outcome="rejected").inc()
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
