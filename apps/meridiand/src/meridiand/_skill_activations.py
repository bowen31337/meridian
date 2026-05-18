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
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class SkillActivationRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="skill_activation_invalid_request", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


class SkillNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="skill_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


class SkillActivationNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="skill_activation_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


class SkillActivationConflictError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="skill_activation_conflict", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 409


class SkillActivationError(MeridianError):
    def __init__(
        self, *, message: str, timestamp: str, cause: BaseException | None = None
    ) -> None:
        super().__init__(
            code="skill_activation_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class SkillActivationApproveError(MeridianError):
    def __init__(
        self, *, message: str, timestamp: str, cause: BaseException | None = None
    ) -> None:
        super().__init__(
            code="skill_activation_approve_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class SkillActivationRevokeError(MeridianError):
    def __init__(
        self, *, message: str, timestamp: str, cause: BaseException | None = None
    ) -> None:
        super().__init__(
            code="skill_activation_revoke_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class SkillActivationListError(MeridianError):
    def __init__(
        self, *, message: str, timestamp: str, cause: BaseException | None = None
    ) -> None:
        super().__init__(
            code="skill_activation_list_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class SkillActivationRequest(BaseModel):
    skill_id: str
    skill_version_id: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _latest_activation(
    activations_dir: Path, agent_id: str, skill_id: str
) -> dict[str, Any] | None:
    """Return the most-recently-requested activation for (agent_id, skill_id), or None."""
    if not activations_dir.exists():
        return None
    candidates: list[dict[str, Any]] = []
    for path in activations_dir.glob("*.json"):
        record: dict[str, Any] = json.loads(path.read_text())
        if record.get("agent_id") == agent_id and record.get("skill_id") == skill_id:
            candidates.append(record)
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("requested_at", ""))


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_skill_activations_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    """
    Implements explicit per-agent skill activation (PRD D2).

    Installing a skill never auto-binds it to any agent.  Each binding
    requires an explicit request followed by an audit-logged human approval.
    """
    router = APIRouter()
    skills_dir = storage_root / "skills"
    activations_dir = storage_root / "skill_activations"

    # ------------------------------------------------------------------
    # POST /v1/agents/{agent_id}/skills — request activation (→ pending)
    # ------------------------------------------------------------------

    @router.post("/v1/agents/{agent_id}/skills", status_code=201)
    async def request_skill_activation(
        agent_id: str, body: SkillActivationRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        activation_id = f"skillact_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "skill.activation.request",
            attributes={
                "agent.id": agent_id,
                "skill.id": body.skill_id,
                "skill.activation.id": activation_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.activation.request.invocation",
                    code="skill_activation_request",
                    timestamp=now,
                ),
            )

            activation_record: dict[str, Any] = {}
            try:
                if not body.skill_id.strip():
                    raise SkillActivationRequestError(
                        message="'skill_id' must not be empty",
                        timestamp=now,
                    )

                skill_path = skills_dir / f"{body.skill_id}.json"
                if not skill_path.exists():
                    raise SkillNotFoundError(
                        message=f"Skill '{body.skill_id}' not found",
                        timestamp=now,
                    )

                existing = _latest_activation(activations_dir, agent_id, body.skill_id)
                if existing is not None and existing.get("status") in ("pending", "active"):
                    raise SkillActivationConflictError(
                        message=(
                            f"Skill '{body.skill_id}' already has a pending or active"
                            f" activation for agent '{agent_id}'"
                        ),
                        timestamp=now,
                    )

                skill_record: dict[str, Any] = json.loads(skill_path.read_text())
                skill_version_id = body.skill_version_id or skill_record.get("version", {}).get(
                    "id"
                )

                activations_dir.mkdir(parents=True, exist_ok=True)
                activation_record = {
                    "id": activation_id,
                    "agent_id": agent_id,
                    "skill_id": body.skill_id,
                    "skill_version_id": skill_version_id,
                    "status": "pending",
                    "requested_at": now,
                    "approved_at": None,
                    "revoked_at": None,
                }
                (activations_dir / f"{activation_id}.json").write_text(
                    json.dumps(activation_record)
                )

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="skill.activation.requested",
                        code="skill_activation_request",
                        timestamp=now,
                        detail={
                            "activation_id": activation_id,
                            "agent_id": agent_id,
                            "skill_id": body.skill_id,
                            "skill_version_id": skill_version_id,
                        },
                    )
                )

            except (
                SkillActivationRequestError,
                SkillNotFoundError,
                SkillActivationConflictError,
            ) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.activation.request.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "activation_id": activation_id,
                            "agent_id": agent_id,
                            "skill_id": body.skill_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = SkillActivationError(
                    message=f"Failed to request skill activation: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.activation.request.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "activation_id": activation_id,
                            "agent_id": agent_id,
                            "skill_id": body.skill_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=activation_record, status_code=201)

    # ------------------------------------------------------------------
    # POST /v1/agents/{agent_id}/skills/{skill_id}/approve — human approval (→ active)
    # ------------------------------------------------------------------

    @router.post("/v1/agents/{agent_id}/skills/{skill_id}/approve", status_code=200)
    async def approve_skill_activation(agent_id: str, skill_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "skill.activation.approve",
            attributes={
                "agent.id": agent_id,
                "skill.id": skill_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.activation.approve.invocation",
                    code="skill_activation_approve",
                    timestamp=now,
                ),
            )

            activation: dict[str, Any] = {}
            try:
                found = _latest_activation(activations_dir, agent_id, skill_id)
                if found is None:
                    raise SkillActivationNotFoundError(
                        message=(
                            f"No activation found for skill '{skill_id}' on agent '{agent_id}'"
                        ),
                        timestamp=now,
                    )

                if found.get("status") != "pending":
                    raise SkillActivationConflictError(
                        message=(
                            f"Activation is not in 'pending' state"
                            f" (current: {found.get('status')})"
                        ),
                        timestamp=now,
                    )

                activation = found
                activation["status"] = "active"
                activation["approved_at"] = now
                span.set_attribute("skill.activation.id", activation["id"])

                activation_path = activations_dir / f"{activation['id']}.json"
                activation_path.write_text(json.dumps(activation))

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="skill.activation.approved",
                        code="skill_activation_approve",
                        timestamp=now,
                        detail={
                            "activation_id": activation["id"],
                            "agent_id": agent_id,
                            "skill_id": skill_id,
                            "skill_version_id": activation.get("skill_version_id"),
                        },
                    )
                )

            except (SkillActivationNotFoundError, SkillActivationConflictError) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.activation.approve.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "agent_id": agent_id,
                            "skill_id": skill_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = SkillActivationApproveError(
                    message=f"Failed to approve skill activation: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.activation.approve.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "agent_id": agent_id,
                            "skill_id": skill_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=activation, status_code=200)

    # ------------------------------------------------------------------
    # DELETE /v1/agents/{agent_id}/skills/{skill_id} — revoke activation
    # ------------------------------------------------------------------

    @router.delete("/v1/agents/{agent_id}/skills/{skill_id}", status_code=200)
    async def revoke_skill_activation(agent_id: str, skill_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "skill.activation.revoke",
            attributes={
                "agent.id": agent_id,
                "skill.id": skill_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.activation.revoke.invocation",
                    code="skill_activation_revoke",
                    timestamp=now,
                ),
            )

            activation: dict[str, Any] = {}
            try:
                found = _latest_activation(activations_dir, agent_id, skill_id)
                if found is None:
                    raise SkillActivationNotFoundError(
                        message=(
                            f"No activation found for skill '{skill_id}' on agent '{agent_id}'"
                        ),
                        timestamp=now,
                    )

                if found.get("status") not in ("pending", "active"):
                    raise SkillActivationConflictError(
                        message=(
                            f"Activation is not in a revocable state"
                            f" (current: {found.get('status')})"
                        ),
                        timestamp=now,
                    )

                activation = found
                activation["status"] = "revoked"
                activation["revoked_at"] = now
                span.set_attribute("skill.activation.id", activation["id"])

                activation_path = activations_dir / f"{activation['id']}.json"
                activation_path.write_text(json.dumps(activation))

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="skill.activation.revoked",
                        code="skill_activation_revoke",
                        timestamp=now,
                        detail={
                            "activation_id": activation["id"],
                            "agent_id": agent_id,
                            "skill_id": skill_id,
                        },
                    )
                )

            except (SkillActivationNotFoundError, SkillActivationConflictError) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.activation.revoke.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "agent_id": agent_id,
                            "skill_id": skill_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = SkillActivationRevokeError(
                    message=f"Failed to revoke skill activation: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.activation.revoke.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "agent_id": agent_id,
                            "skill_id": skill_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=activation, status_code=200)

    # ------------------------------------------------------------------
    # GET /v1/agents/{agent_id}/skills — list activations for an agent
    # ------------------------------------------------------------------

    @router.get("/v1/agents/{agent_id}/skills")
    async def list_agent_skill_activations(
        agent_id: str,
        limit: int = Query(default=20),
        offset: int = Query(default=0),
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "skill.activation.list",
            attributes={"agent.id": agent_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.activation.list.invocation",
                    code="skill_activation_list",
                    timestamp=now,
                ),
            )

            try:
                all_activations: list[dict[str, Any]] = []
                if activations_dir.exists():
                    for path in activations_dir.glob("*.json"):
                        record: dict[str, Any] = json.loads(path.read_text())
                        if record.get("agent_id") == agent_id:
                            all_activations.append(record)

                all_activations.sort(key=lambda r: r.get("requested_at", ""), reverse=True)
                total = len(all_activations)
                page = all_activations[offset : offset + limit]

            except Exception as exc:
                err = SkillActivationListError(
                    message=f"Failed to list skill activations: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.activation.list.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"agent_id": agent_id, "message": err.message},
                    )
                )
                raise err

        return JSONResponse(
            content={"items": page, "total": total, "limit": limit, "offset": offset},
            status_code=200,
        )

    return router
