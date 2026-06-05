"""Skill Forge soak test runner.

POST /v1/x/skill-forge/soak-run reads recorded fixture files from
storage_root/skill_forge/soak_fixtures/, runs the configured forge provider
on each, computes proposal precision (hit_count / fixture_count), and asserts
≥ 50% per PRD §7.2.

On every invocation: emits OTel span "skill_forge.soak.run" and logs a
structured audit event.  On failure: records the error to the span, surfaces
the error message in the JSON error body, and writes the failure to the audit log.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
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
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ._skill_forge import NoopSkillForgeProvider, SkillForgeProvider

PRECISION_THRESHOLD = 0.50


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class SkillForgeSoakError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_forge_soak_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _proposal_matches(result: str, expected: str) -> bool:
    return result == expected


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_skill_forge_soak_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    provider: SkillForgeProvider | None = None,
) -> APIRouter:
    _provider: SkillForgeProvider = provider if provider is not None else NoopSkillForgeProvider()
    router = APIRouter()

    @router.post("/v1/x/skill-forge/soak-run")
    async def run_soak() -> JSONResponse:
        now = _now()
        run_id = f"soak_{uuid.uuid4().hex}"
        fixtures_dir = storage_root / "skill_forge" / "soak_fixtures"
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "skill_forge.soak.run",
            attributes={
                "skill_forge.soak.run_id": run_id,
                "skill_forge.soak.fixtures_dir": str(fixtures_dir),
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill_forge.soak.run.invocation",
                    code="skill_forge_soak_run",
                    timestamp=now,
                ),
            )

            fixture_results: list[dict[str, Any]] = []
            hit_count = 0

            if fixtures_dir.exists():
                for fixture_path in sorted(fixtures_dir.glob("*.json")):
                    fixture = _load_fixture(fixture_path)
                    if fixture is None:
                        continue
                    fixture_id = fixture.get("id", fixture_path.stem)
                    skill: dict[str, Any] = fixture.get("skill") or {}
                    job_type: str = fixture.get("job_type", "")
                    expected: str = fixture.get("expected_proposal", "")

                    try:
                        result = await _provider.forge(skill, job_type)
                        hit = _proposal_matches(result, expected)
                        if hit:
                            hit_count += 1
                        fixture_results.append(
                            {
                                "fixture_id": fixture_id,
                                "status": "hit" if hit else "miss",
                                "result": result,
                                "expected_proposal": expected,
                            }
                        )
                    except Exception as exc:
                        fixture_results.append(
                            {
                                "fixture_id": fixture_id,
                                "status": "error",
                                "error": str(exc),
                            }
                        )

            fixture_count = len(fixture_results)
            precision = hit_count / fixture_count if fixture_count > 0 else 0.0

            span.set_attribute("skill_forge.soak.fixture_count", fixture_count)
            span.set_attribute("skill_forge.soak.hit_count", hit_count)
            span.set_attribute("skill_forge.soak.precision", precision)

            if fixture_count > 0 and precision < PRECISION_THRESHOLD:
                msg = (
                    f"Skill Forge soak precision {precision:.2%} is below the required "
                    f"{PRECISION_THRESHOLD:.0%} threshold (PRD §7.2); "
                    f"hit {hit_count}/{fixture_count} fixtures"
                )
                err = SkillForgeSoakError(message=msg, timestamp=_now())
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill_forge.soak.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "run_id": run_id,
                            "precision": precision,
                            "hit_count": hit_count,
                            "fixture_count": fixture_count,
                            "message": msg,
                        },
                    )
                )
                raise err

            span.set_attribute("skill_forge.soak.success", True)
            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="skill_forge.soak.ran",
                    code="skill_forge_soak_ran",
                    timestamp=_now(),
                    detail={
                        "run_id": run_id,
                        "precision": precision,
                        "hit_count": hit_count,
                        "fixture_count": fixture_count,
                    },
                )
            )

        return JSONResponse(
            content={
                "run_id": run_id,
                "status": "passed",
                "precision": precision,
                "hit_count": hit_count,
                "fixture_count": fixture_count,
                "fixtures": fixture_results,
            }
        )

    return router
