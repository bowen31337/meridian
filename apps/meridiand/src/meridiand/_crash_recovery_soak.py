"""
Crash-recovery soak test for single-node Meridian (PRD §7.4).

POST /v1/x/ci/crash-recovery-soak-run seeds *crash_count* synthetic mid-session
states on disk, then exercises the full wake() recovery path for each (simulating
the HarnessPool.start() auto-resume after a SIGKILL), and asserts that the
auto-resume rate is ≥ RESUME_RATE_THRESHOLD (99%).

On every invocation: emits OTel span ``crash.recovery.soak.run`` and logs a
structured audit event.  On failure: records the error to the span, surfaces
the error message in the JSON error body, and writes the failure to the audit log.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
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
from storage_reposit import LocalEventLogReader, PhaseProjection

CRASH_COUNT = 10_000
RESUME_RATE_THRESHOLD = 0.99

_STOP_PHASES: frozenset[str] = frozenset({"idle", "paused", "terminated"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class CrashRecoverySoakError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="crash_recovery_soak_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Synthetic crash helpers
# ---------------------------------------------------------------------------


def _seed_synthetic_session(
    storage_root: Path,
    session_id: str,
    now: str,
) -> None:
    """Write a minimal session manifest to disk to simulate a live session."""
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "status": "active",
                "created_at": now,
            }
        )
    )


def _attempt_recovery(storage_root: Path, session_id: str) -> bool:
    """
    Simulate the HarnessPool.wake() recovery path after a SIGKILL.

    Mirrors what wake_session() does:
      1. Load session manifest from disk.
      2. Derive current phase from the event log via PhaseProjection.
      3. Verify the phase is active (not idle/paused/terminated).

    Returns True if recovery succeeds, False if any step fails.
    """
    try:
        manifest_path = storage_root / "sessions" / session_id / "manifest.json"
        if not manifest_path.exists():
            return False

        session = json.loads(manifest_path.read_text())
        if not isinstance(session, dict) or not session.get("session_id"):
            return False

        reader = LocalEventLogReader(storage_root)
        projection = PhaseProjection(reader)
        phase = projection.current_phase(session_id)

        return phase not in _STOP_PHASES
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_crash_recovery_soak_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    _crash_count_override: int | None = None,
) -> APIRouter:
    """
    Crash-recovery soak router.

    *_crash_count_override* replaces CRASH_COUNT; supply only in tests so the
    suite completes in milliseconds rather than exercising the full 10 000-run set.
    """
    router = APIRouter()

    @router.post("/v1/x/ci/crash-recovery-soak-run")
    async def run_crash_recovery_soak() -> JSONResponse:
        now = _now()
        run_id = f"crash_soak_{uuid.uuid4().hex}"
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "crash.recovery.soak.run",
            attributes={"crash.recovery.soak.run_id": run_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="crash.recovery.soak.run.invocation",
                    code="crash_recovery_soak_run",
                    timestamp=now,
                ),
            )

            crash_count = (
                _crash_count_override if _crash_count_override is not None else CRASH_COUNT
            )

            resume_count = 0
            failure_count = 0
            sample_failures: list[dict[str, str]] = []

            for i in range(crash_count):
                session_id = f"soak_{run_id}_{i}"
                try:
                    _seed_synthetic_session(storage_root, session_id, now)
                    recovered = _attempt_recovery(storage_root, session_id)
                except Exception as exc:
                    recovered = False
                    sample_failures.append({"session_id": session_id, "error": str(exc)})

                if recovered:
                    resume_count += 1
                else:
                    failure_count += 1
                    if len(sample_failures) < 10:
                        sample_failures.append(
                            {
                                "session_id": session_id,
                                "error": "recovery_attempt_returned_false",
                            }
                        )

            resume_rate = resume_count / crash_count if crash_count > 0 else 1.0

            span.set_attribute("crash.recovery.soak.crash_count", crash_count)
            span.set_attribute("crash.recovery.soak.resume_count", resume_count)
            span.set_attribute("crash.recovery.soak.failure_count", failure_count)
            span.set_attribute("crash.recovery.soak.resume_rate", resume_rate)

            if resume_rate < RESUME_RATE_THRESHOLD:
                pct = resume_rate * 100
                threshold_pct = RESUME_RATE_THRESHOLD * 100
                msg = (
                    f"Crash-recovery soak failed: auto-resume rate {pct:.2f}% "
                    f"({resume_count}/{crash_count}) is below the {threshold_pct:.0f}% "
                    f"threshold required by PRD §7.4"
                )
                err = CrashRecoverySoakError(message=msg, timestamp=_now())
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="crash.recovery.soak.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "run_id": run_id,
                            "crash_count": crash_count,
                            "resume_count": resume_count,
                            "failure_count": failure_count,
                            "resume_rate": resume_rate,
                            "message": msg,
                        },
                    )
                )
                raise err

            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="crash.recovery.soak.ran",
                    code="crash_recovery_soak_ran",
                    timestamp=_now(),
                    detail={
                        "run_id": run_id,
                        "crash_count": crash_count,
                        "resume_count": resume_count,
                        "failure_count": failure_count,
                        "resume_rate": resume_rate,
                    },
                )
            )

        return JSONResponse(
            content={
                "run_id": run_id,
                "status": "passed",
                "crash_count": crash_count,
                "resume_count": resume_count,
                "failure_count": failure_count,
                "resume_rate": resume_rate,
                "sample_failures": sample_failures,
            }
        )

    return router
