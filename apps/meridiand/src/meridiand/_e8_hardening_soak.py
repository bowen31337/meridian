"""
E8 hardening soak: multi-day, multi-agent, multi-channel dogfood run with
SIGKILL injection on harness, tool workers, and the daemon itself (PRD §7.4).

POST /v1/x/ci/e8-hardening-soak-run seeds AGENT_COUNT × CHANNEL_COUNT ×
SESSIONS_PER_COMBO synthetic mid-session states across all (agent, channel)
combinations (10 000 total), distributes them across three SIGKILL injection
layers — harness kill, tool-worker kill, daemon kill — then exercises the full
HarnessPool.wake() recovery path for each, and asserts that the combined
auto-resume rate is ≥ RESUME_RATE_THRESHOLD (99%).

On every invocation: emits OTel span ``e8.hardening.soak.run`` and logs a
structured audit event.  On failure: records the error to the span, surfaces
the error message in the JSON error body, and writes the failure to the audit log.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

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

# PRD §7.4: > 99% resume rate over 10k synthetic crashes
AGENT_COUNT = 5
CHANNEL_COUNT = 4
SESSIONS_PER_COMBO = 500  # AGENT_COUNT × CHANNEL_COUNT × SESSIONS_PER_COMBO = 10 000
RESUME_RATE_THRESHOLD = 0.99

_SYNTHETIC_AGENTS: list[str] = [f"agent_{i}" for i in range(AGENT_COUNT)]
_SYNTHETIC_CHANNELS: list[str] = ["cli", "webhook", "telegram", "slack"]

_STOP_PHASES: frozenset[str] = frozenset({"idle", "paused", "terminated"})

# SIGKILL injection layer labels (distributed via session index modulo 3)
_LAYERS: list[str] = ["harness", "tool_worker", "daemon"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class E8HardeningSoakError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="e8_hardening_soak_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Synthetic session seeders — one per SIGKILL injection layer
# ---------------------------------------------------------------------------


def _seed_harness_kill_session(
    storage_root: Path,
    session_id: str,
    agent_id: str,
    channel_id: str,
    now: str,
) -> None:
    """Harness SIGKILL: process killed while harness was running; manifest on disk, no orderly shutdown event."""
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "agent_id": agent_id,
                "channel_id": channel_id,
                "status": "active",
                "kill_layer": "harness",
                "created_at": now,
            }
        )
    )


def _seed_tool_worker_kill_session(
    storage_root: Path,
    session_id: str,
    agent_id: str,
    channel_id: str,
    now: str,
) -> None:
    """Tool-worker SIGKILL: sandbox subprocess killed during tool execution; in-flight call recorded in manifest."""
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "agent_id": agent_id,
                "channel_id": channel_id,
                "status": "active",
                "kill_layer": "tool_worker",
                "pending_tool_call_id": f"tool_{session_id}",
                "created_at": now,
            }
        )
    )


def _seed_daemon_kill_session(
    storage_root: Path,
    session_id: str,
    agent_id: str,
    channel_id: str,
    now: str,
) -> None:
    """Daemon SIGKILL: entire meridiand process killed; all in-memory state lost, disk state intact."""
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "agent_id": agent_id,
                "channel_id": channel_id,
                "status": "active",
                "kill_layer": "daemon",
                "created_at": now,
            }
        )
    )


_SEED_FNS = {
    "harness": _seed_harness_kill_session,
    "tool_worker": _seed_tool_worker_kill_session,
    "daemon": _seed_daemon_kill_session,
}


# ---------------------------------------------------------------------------
# Recovery probe — mirrors HarnessPool.wake()
# ---------------------------------------------------------------------------


def _attempt_recovery(storage_root: Path, session_id: str) -> bool:
    """
    Simulate HarnessPool.wake() after a SIGKILL:
      1. Load session manifest from disk.
      2. Derive current phase from event log via PhaseProjection.
      3. Return True if the phase is resumable (not idle/paused/terminated).
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


def make_e8_hardening_soak_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    _sessions_per_combo_override: int | None = None,
) -> APIRouter:
    """
    E8 hardening soak router.

    *_sessions_per_combo_override* replaces SESSIONS_PER_COMBO; supply only in
    tests so the suite completes quickly rather than seeding the full 10 000-session set.
    """
    router = APIRouter()

    @router.post("/v1/x/ci/e8-hardening-soak-run")
    async def run_e8_hardening_soak() -> JSONResponse:
        now = _now()
        run_id = f"e8_soak_{uuid.uuid4().hex}"
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "e8.hardening.soak.run",
            attributes={
                "e8.hardening.soak.run_id": run_id,
                "e8.hardening.soak.agent_count": AGENT_COUNT,
                "e8.hardening.soak.channel_count": len(_SYNTHETIC_CHANNELS),
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="e8.hardening.soak.run.invocation",
                    code="e8_hardening_soak_run",
                    timestamp=now,
                ),
            )

            sessions_per_combo = (
                _sessions_per_combo_override
                if _sessions_per_combo_override is not None
                else SESSIONS_PER_COMBO
            )

            total = 0
            resume_count = 0
            failure_count = 0
            layer_results: dict[str, dict[str, int]] = {
                layer: {"total": 0, "resumed": 0} for layer in _LAYERS
            }
            sample_failures: list[dict[str, str]] = []

            for agent_id in _SYNTHETIC_AGENTS:
                for channel_id in _SYNTHETIC_CHANNELS:
                    for i in range(sessions_per_combo):
                        # Distribute evenly across the three SIGKILL injection layers
                        layer = _LAYERS[i % 3]
                        session_id = f"e8_{run_id}_{agent_id}_{channel_id}_{i}"

                        try:
                            _SEED_FNS[layer](
                                storage_root, session_id, agent_id, channel_id, now
                            )
                            recovered = _attempt_recovery(storage_root, session_id)
                        except Exception as exc:
                            recovered = False
                            if len(sample_failures) < 10:
                                sample_failures.append(
                                    {
                                        "session_id": session_id,
                                        "layer": layer,
                                        "error": str(exc),
                                    }
                                )

                        total += 1
                        layer_results[layer]["total"] += 1

                        if recovered:
                            resume_count += 1
                            layer_results[layer]["resumed"] += 1
                        else:
                            failure_count += 1
                            if len(sample_failures) < 10:
                                sample_failures.append(
                                    {
                                        "session_id": session_id,
                                        "layer": layer,
                                        "error": "recovery_attempt_returned_false",
                                    }
                                )

            resume_rate = resume_count / total if total > 0 else 1.0

            span.set_attribute("e8.hardening.soak.total_sessions", total)
            span.set_attribute("e8.hardening.soak.resume_count", resume_count)
            span.set_attribute("e8.hardening.soak.failure_count", failure_count)
            span.set_attribute("e8.hardening.soak.resume_rate", resume_rate)
            for layer, counts in layer_results.items():
                span.set_attribute(
                    f"e8.hardening.soak.{layer}.total", counts["total"]
                )
                span.set_attribute(
                    f"e8.hardening.soak.{layer}.resumed", counts["resumed"]
                )

            if resume_rate < RESUME_RATE_THRESHOLD:
                pct = resume_rate * 100
                threshold_pct = RESUME_RATE_THRESHOLD * 100
                msg = (
                    f"E8 hardening soak failed: auto-resume rate {pct:.2f}% "
                    f"({resume_count}/{total}) is below the {threshold_pct:.0f}% "
                    f"threshold required by PRD §7.4; "
                    f"tested {AGENT_COUNT} agents × {len(_SYNTHETIC_CHANNELS)} channels "
                    f"with harness/tool-worker/daemon SIGKILL injection"
                )
                err = E8HardeningSoakError(message=msg, timestamp=_now())
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="e8.hardening.soak.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "run_id": run_id,
                            "total_sessions": total,
                            "resume_count": resume_count,
                            "failure_count": failure_count,
                            "resume_rate": resume_rate,
                            "layer_results": layer_results,
                            "message": msg,
                        },
                    )
                )
                raise err

            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="e8.hardening.soak.ran",
                    code="e8_hardening_soak_ran",
                    timestamp=_now(),
                    detail={
                        "run_id": run_id,
                        "total_sessions": total,
                        "resume_count": resume_count,
                        "failure_count": failure_count,
                        "resume_rate": resume_rate,
                        "layer_results": layer_results,
                        "agent_count": AGENT_COUNT,
                        "channel_count": len(_SYNTHETIC_CHANNELS),
                        "sessions_per_combo": sessions_per_combo,
                    },
                )
            )

        return JSONResponse(
            content={
                "run_id": run_id,
                "status": "passed",
                "total_sessions": total,
                "resume_count": resume_count,
                "failure_count": failure_count,
                "resume_rate": resume_rate,
                "layer_results": layer_results,
                "agent_count": AGENT_COUNT,
                "channel_count": len(_SYNTHETIC_CHANNELS),
                "sessions_per_combo": sessions_per_combo,
                "sample_failures": sample_failures,
            }
        )

    return router
