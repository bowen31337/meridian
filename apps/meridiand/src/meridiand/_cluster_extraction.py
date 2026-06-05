"""Cluster extraction: for each cluster, Forge asks a model via the Model Router to extract
skill metadata.

On every invocation: emits OTel span ``"skill_forge.cluster.extract"`` and logs a structured
event.  On failure: records the error to the span, surfaces the error message to the caller,
and writes the failure to the audit log before re-raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
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
from meridian_sdk_provider import Message, ModelCallOpts, ModelRouter
from meridian_sdk_provider.types import TextDeltaEvent


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


@dataclass
class ClusterMember:
    """One session trajectory within a cluster."""

    session_id: str
    tool_calls: list[str]


@dataclass
class Cluster:
    """A group of structurally similar session trajectories."""

    id: str
    members: list[ClusterMember]


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class ToolRequirement:
    name: str
    capabilities: list[str] = field(default_factory=list)


@dataclass
class SkillTestCase:
    description: str
    steps: list[str] = field(default_factory=list)


@dataclass
class ClusterExtractionResult:
    cluster_id: str
    name: str
    description: str
    instructions: str
    tools: list[ToolRequirement]
    tests: list[SkillTestCase]


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ClusterExtractionError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="cluster_extraction_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a skill-extraction assistant for an AI agent platform.

Given a cluster of agent trajectories (tool-call sequences from similar sessions),
extract the following in a single JSON object:

{
  "name": "<short snake_case skill name>",
  "description": "<one-sentence description of what the skill does>",
  "instructions": "<concise, general instructions — no session-specific details>",
  "tools": [{"name": "<tool_name>", "capabilities": ["<cap1>", ...]}],
  "tests": [{"description": "<what the test verifies>", "steps": ["<step1>", ...]}]
}

Rules:
- instructions must be general and reusable, never referencing specific files, users, or sessions
- include at least one test case
- respond with ONLY valid JSON, no prose
"""


def _build_user_message(cluster: Cluster) -> str:
    lines = [f"Cluster ID: {cluster.id}", f"Members: {len(cluster.members)}", ""]
    for i, member in enumerate(cluster.members, 1):
        calls = ", ".join(member.tool_calls) if member.tool_calls else "(none)"
        lines.append(f"Session {i} ({member.session_id}): {calls}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(text: str, cluster_id: str) -> ClusterExtractionResult:
    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response is not valid JSON: {exc}") from exc

    tools = [
        ToolRequirement(
            name=t.get("name", ""),
            capabilities=t.get("capabilities", []),
        )
        for t in data.get("tools", [])
    ]
    tests = [
        SkillTestCase(
            description=tc.get("description", ""),
            steps=tc.get("steps", []),
        )
        for tc in data.get("tests", [])
    ]
    return ClusterExtractionResult(
        cluster_id=cluster_id,
        name=data.get("name", ""),
        description=data.get("description", ""),
        instructions=data.get("instructions", ""),
        tools=tools,
        tests=tests,
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


async def _collect_text(router: ModelRouter, opts: ModelCallOpts) -> str:
    parts: list[str] = []
    async for event in router.call(opts):
        if isinstance(event, TextDeltaEvent):
            parts.append(event.text)
    return "".join(parts)


async def extract_cluster(
    cluster: Cluster,
    *,
    router: ModelRouter,
    audit_log: AuditLog,
) -> ClusterExtractionResult:
    """Ask the model (via router) to extract skill metadata from a cluster.

    Extracts: skill name + description, distilled general instructions,
    tool list + capability requirements, and candidate replayable test cases.

    On every invocation: emits OTel span ``"skill_forge.cluster.extract"`` and
    logs a structured event.  On failure: records the error to the span, surfaces
    the error message to the caller, and writes the failure to the audit log
    before re-raising as :class:`ClusterExtractionError`.
    """
    now = _now()
    tracer = get_tracer()

    with tracer.start_as_current_span(
        "skill_forge.cluster.extract",
        attributes={
            "skill_forge.cluster.id": cluster.id,
            "skill_forge.cluster.size": len(cluster.members),
        },
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="skill_forge.cluster.extract.invocation",
                code="skill_forge_cluster_extract",
                timestamp=now,
            ),
        )

        try:
            opts = ModelCallOpts(
                model="",
                messages=[
                    Message(role="user", content=_build_user_message(cluster)),
                ],
                system=_SYSTEM_PROMPT,
                role="skill_forge_extractor",
                metadata={"skill_forge.cluster.id": cluster.id},
            )
            raw = await _collect_text(router, opts)
            result = _parse_response(raw, cluster.id)

            span.set_attribute("skill_forge.cluster.extract.success", True)
            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="skill_forge.cluster.extracted",
                    code="skill_forge_cluster_extract",
                    timestamp=_now(),
                    detail={
                        "cluster_id": cluster.id,
                        "cluster_size": len(cluster.members),
                        "skill_name": result.name,
                    },
                )
            )
            return result

        except Exception as exc:
            err = ClusterExtractionError(
                message=f"Failed to extract cluster {cluster.id!r}: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            span.set_attribute("skill_forge.cluster.extract.success", False)
            record_error(span, err)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="skill_forge.cluster.extract.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={
                        "cluster_id": cluster.id,
                        "cluster_size": len(cluster.members),
                        "message": err.message,
                    },
                )
            )
            raise err from exc
