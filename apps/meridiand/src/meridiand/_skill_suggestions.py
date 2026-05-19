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
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_AUTO_SUGGEST_MODE = "auto_suggest"


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class SkillSuggestionRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="skill_suggestion_invalid_request", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


class SkillSuggestionModeError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="skill_suggestion_mode_not_enabled", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


class SkillNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="skill_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


class AgentNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="agent_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


class SkillSuggestionNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="skill_suggestion_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


class SkillSuggestionConflictError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="skill_suggestion_conflict", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 409


class SkillSuggestionEmitError(MeridianError):
    def __init__(
        self, *, message: str, timestamp: str, cause: BaseException | None = None
    ) -> None:
        super().__init__(
            code="skill_suggestion_emit_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class SkillSuggestionApproveError(MeridianError):
    def __init__(
        self, *, message: str, timestamp: str, cause: BaseException | None = None
    ) -> None:
        super().__init__(
            code="skill_suggestion_approve_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class SkillSuggestionRequest(BaseModel):
    skill_id: str
    skill_version_id: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _latest_suggestion(
    suggestions_dir: Path, agent_id: str, skill_id: str
) -> dict[str, Any] | None:
    """Return the most-recently-suggested record for (agent_id, skill_id), or None."""
    if not suggestions_dir.exists():
        return None
    candidates: list[dict[str, Any]] = []
    for path in suggestions_dir.glob("*.json"):
        record: dict[str, Any] = json.loads(path.read_text())
        if record.get("agent_id") == agent_id and record.get("skill_id") == skill_id:
            candidates.append(record)
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("suggested_at", ""))


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


def make_skill_suggestions_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    """
    Implements opt-in auto-suggest activation mode (PRD D2).

    When agent.config['skill_activation_mode'] == 'auto_suggest', the harness
    may emit skill_suggestion events.  Each suggestion requires explicit human
    approval before a skill binding (activation) is created.  No skill
    auto-activates without audit-logged human approval.
    """
    router = APIRouter()
    agents_dir = storage_root / "agents"
    skills_dir = storage_root / "skills"
    suggestions_dir = storage_root / "skill_suggestions"
    activations_dir = storage_root / "skill_activations"

    # ------------------------------------------------------------------
    # POST /v1/agents/{agent_id}/skill_suggestions — emit suggestion
    # ------------------------------------------------------------------

    @router.post("/v1/agents/{agent_id}/skill_suggestions", status_code=201)
    async def emit_skill_suggestion(
        agent_id: str, body: SkillSuggestionRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        suggestion_id = f"skillsugg_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "skill.suggestion.emit",
            attributes={
                "agent.id": agent_id,
                "skill.id": body.skill_id,
                "skill.suggestion.id": suggestion_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.suggestion.emit.invocation",
                    code="skill_suggestion_emit",
                    timestamp=now,
                ),
            )

            suggestion_record: dict[str, Any] = {}
            try:
                if not body.skill_id.strip():
                    raise SkillSuggestionRequestError(
                        message="'skill_id' must not be empty",
                        timestamp=now,
                    )

                agent_path = agents_dir / f"{agent_id}.json"
                if not agent_path.exists():
                    raise AgentNotFoundError(
                        message=f"Agent '{agent_id}' not found",
                        timestamp=now,
                    )

                agent_record: dict[str, Any] = json.loads(agent_path.read_text())
                agent_config: dict[str, Any] = (
                    agent_record.get("version", {}).get("config", {})
                )
                activation_mode = agent_config.get("skill_activation_mode")
                if activation_mode != _AUTO_SUGGEST_MODE:
                    raise SkillSuggestionModeError(
                        message=(
                            f"Agent '{agent_id}' does not have"
                            f" skill_activation_mode='auto_suggest'"
                            f" (current: {activation_mode!r})"
                        ),
                        timestamp=now,
                    )

                skill_path = skills_dir / f"{body.skill_id}.json"
                if not skill_path.exists():
                    raise SkillNotFoundError(
                        message=f"Skill '{body.skill_id}' not found",
                        timestamp=now,
                    )

                existing_suggestion = _latest_suggestion(suggestions_dir, agent_id, body.skill_id)
                if existing_suggestion is not None and existing_suggestion.get("status") == "suggested":
                    raise SkillSuggestionConflictError(
                        message=(
                            f"Skill '{body.skill_id}' already has a pending suggestion"
                            f" for agent '{agent_id}'"
                        ),
                        timestamp=now,
                    )

                existing_activation = _latest_activation(activations_dir, agent_id, body.skill_id)
                if existing_activation is not None and existing_activation.get("status") in (
                    "pending",
                    "active",
                ):
                    raise SkillSuggestionConflictError(
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

                suggestions_dir.mkdir(parents=True, exist_ok=True)
                suggestion_record = {
                    "id": suggestion_id,
                    "agent_id": agent_id,
                    "skill_id": body.skill_id,
                    "skill_version_id": skill_version_id,
                    "status": "suggested",
                    "suggested_at": now,
                    "approved_at": None,
                    "dismissed_at": None,
                }
                (suggestions_dir / f"{suggestion_id}.json").write_text(
                    json.dumps(suggestion_record)
                )

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="skill.suggestion.emitted",
                        code="skill_suggestion_emit",
                        timestamp=now,
                        detail={
                            "suggestion_id": suggestion_id,
                            "agent_id": agent_id,
                            "skill_id": body.skill_id,
                            "skill_version_id": skill_version_id,
                        },
                    )
                )

            except (
                SkillSuggestionRequestError,
                SkillSuggestionModeError,
                AgentNotFoundError,
                SkillNotFoundError,
                SkillSuggestionConflictError,
            ) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.suggestion.emit.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "suggestion_id": suggestion_id,
                            "agent_id": agent_id,
                            "skill_id": body.skill_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = SkillSuggestionEmitError(
                    message=f"Failed to emit skill suggestion: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.suggestion.emit.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "suggestion_id": suggestion_id,
                            "agent_id": agent_id,
                            "skill_id": body.skill_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=suggestion_record, status_code=201)

    # ------------------------------------------------------------------
    # POST /v1/agents/{agent_id}/skill_suggestions/{skill_id}/approve
    # Human approval: creates an active skill binding.
    # ------------------------------------------------------------------

    @router.post("/v1/agents/{agent_id}/skill_suggestions/{skill_id}/approve", status_code=200)
    async def approve_skill_suggestion(agent_id: str, skill_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "skill.suggestion.approve",
            attributes={
                "agent.id": agent_id,
                "skill.id": skill_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.suggestion.approve.invocation",
                    code="skill_suggestion_approve",
                    timestamp=now,
                ),
            )

            suggestion: dict[str, Any] = {}
            try:
                found = _latest_suggestion(suggestions_dir, agent_id, skill_id)
                if found is None:
                    raise SkillSuggestionNotFoundError(
                        message=(
                            f"No suggestion found for skill '{skill_id}' on agent '{agent_id}'"
                        ),
                        timestamp=now,
                    )

                if found.get("status") != "suggested":
                    raise SkillSuggestionConflictError(
                        message=(
                            f"Suggestion is not in 'suggested' state"
                            f" (current: {found.get('status')})"
                        ),
                        timestamp=now,
                    )

                suggestion = found
                suggestion["status"] = "approved"
                suggestion["approved_at"] = now
                span.set_attribute("skill.suggestion.id", suggestion["id"])

                suggestion_path = suggestions_dir / f"{suggestion['id']}.json"
                suggestion_path.write_text(json.dumps(suggestion))

                activation_id = f"skillact_{uuid.uuid4().hex}"
                activations_dir.mkdir(parents=True, exist_ok=True)
                activation_record: dict[str, Any] = {
                    "id": activation_id,
                    "agent_id": agent_id,
                    "skill_id": skill_id,
                    "skill_version_id": suggestion.get("skill_version_id"),
                    "status": "active",
                    "requested_at": now,
                    "approved_at": now,
                    "revoked_at": None,
                }
                (activations_dir / f"{activation_id}.json").write_text(
                    json.dumps(activation_record)
                )

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="skill.suggestion.approved",
                        code="skill_suggestion_approve",
                        timestamp=now,
                        detail={
                            "suggestion_id": suggestion["id"],
                            "activation_id": activation_id,
                            "agent_id": agent_id,
                            "skill_id": skill_id,
                            "skill_version_id": suggestion.get("skill_version_id"),
                        },
                    )
                )

            except (SkillSuggestionNotFoundError, SkillSuggestionConflictError) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.suggestion.approve.failed",
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
                err2 = SkillSuggestionApproveError(
                    message=f"Failed to approve skill suggestion: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.suggestion.approve.failed",
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

        return JSONResponse(content=suggestion, status_code=200)

    return router
