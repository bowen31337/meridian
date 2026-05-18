from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import uuid
from datetime import UTC, datetime, timedelta
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
from storage_blob._local import LocalBlobStore

from ._config import CompactionConfig


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class CompactionError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="compaction_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class CompactionSessionNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="compaction_session_not_found", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 404


class RestoreError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="restore_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class RestoreSessionNotArchivedError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="restore_session_not_archived", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 404


# ---------------------------------------------------------------------------
# AutoCompactor
# ---------------------------------------------------------------------------


class AutoCompactor:
    """
    Scans for idle sessions and compacts their event logs.

    Compaction archives the full NDJSON event log (gzip) to the blob store and
    replaces the live log with a tail summary so disk usage is bounded.
    """

    def __init__(
        self,
        storage_root: Path,
        idle_days: int = 30,
        tail_events: int = 50,
    ) -> None:
        self._root = storage_root
        self._idle_days = idle_days
        self._tail_events = tail_events
        self._blob = LocalBlobStore(storage_root)

    def find_event_files(self, session_id: str) -> list[Path]:
        """Return all event log files for a session across date partitions, sorted."""
        return sorted(
            self._root.glob(f"events/*/*/*/{session_id}.ndjson"),
            key=lambda p: p.parts,
        )

    def find_idle_sessions(self) -> list[str]:
        """Return session IDs whose most recent event file is older than idle_days."""
        threshold = datetime.now(UTC) - timedelta(days=self._idle_days)
        events_dir = self._root / "events"
        if not events_dir.exists():
            return []

        latest: dict[str, float] = {}
        for f in events_dir.glob("*/*/*/*"):
            if f.suffix != ".ndjson":
                continue
            session_id = f.stem
            mtime = f.stat().st_mtime
            if session_id not in latest or mtime > latest[session_id]:
                latest[session_id] = mtime

        return [
            sid
            for sid, mtime in latest.items()
            if datetime.fromtimestamp(mtime, tz=UTC) < threshold
        ]

    async def compact_session(self, session_id: str) -> dict[str, Any]:
        """
        Compact one session's event log:
          1. Archive the full event log (gzip) to blob store.
          2. Replace the live log with a tail summary.
          3. Write a manifest to compaction/<session_id>/manifest.json.
        """
        now = _now()
        files = self.find_event_files(session_id)
        if not files:
            raise CompactionSessionNotFoundError(
                message=f"No event log found for session {session_id!r}",
                timestamp=now,
            )

        all_lines: list[str] = []
        for f in files:
            all_lines.extend(line for line in f.read_text().splitlines() if line.strip())

        original_count = len(all_lines)

        archive_ts = now.replace(":", "-").replace("+", "").replace(".", "-")
        archive_key = f"compaction/{session_id}/archive-{archive_ts}.ndjson.gz"
        compressed = gzip.compress(("\n".join(all_lines) + "\n").encode())
        await self._blob.put(archive_key, compressed)

        tail_lines = all_lines[-self._tail_events :]

        most_recent = files[-1]
        most_recent.write_text(
            "\n".join(tail_lines) + ("\n" if tail_lines else ""),
        )
        for f in files[:-1]:
            f.unlink(missing_ok=True)

        compaction_dir = self._root / "compaction" / session_id
        compaction_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "session_id": session_id,
            "compacted_at": now,
            "strategy": "tail",
            "tail_events": self._tail_events,
            "original_event_count": original_count,
            "summary_event_count": len(tail_lines),
            "archive_key": archive_key,
            "archived_file_count": len(files),
        }
        (compaction_dir / "manifest.json").write_text(json.dumps(manifest))

        return manifest

    async def restore_session(self, session_id: str) -> dict[str, Any]:
        """
        Restore a previously archived session:
          1. Read the compaction manifest to find the archive blob key.
          2. Decompress and recover all pre-compaction events.
          3. Append any events written to the live file after compaction.
          4. Write the full event log back to the live partition file.
          5. Delete the archive blob and manifest.
        """
        now = _now()
        manifest_path = self._root / "compaction" / session_id / "manifest.json"
        if not manifest_path.exists():
            raise RestoreSessionNotArchivedError(
                message=f"No archive found for session {session_id!r}",
                timestamp=now,
            )

        manifest = json.loads(manifest_path.read_text())
        archive_key = manifest["archive_key"]
        summary_event_count: int = manifest.get("summary_event_count", 0)

        compressed = await self._blob.get(archive_key)
        archive_text = gzip.decompress(compressed).decode()
        archive_lines = [l for l in archive_text.splitlines() if l.strip()]

        live_files = self.find_event_files(session_id)
        new_events: list[str] = []
        if live_files:
            live_lines: list[str] = []
            for f in live_files:
                live_lines.extend(l for l in f.read_text().splitlines() if l.strip())
            new_events = live_lines[summary_event_count:]

        restored_lines = archive_lines + new_events

        if live_files:
            target = live_files[-1]
            for f in live_files[:-1]:
                f.unlink(missing_ok=True)
        else:
            now_dt = datetime.now(UTC)
            target = (
                self._root
                / "events"
                / str(now_dt.year)
                / f"{now_dt.month:02d}"
                / f"{now_dt.day:02d}"
                / f"{session_id}.ndjson"
            )
            target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text("\n".join(restored_lines) + ("\n" if restored_lines else ""))

        await self._blob.delete(archive_key)
        manifest_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            (self._root / "compaction" / session_id).rmdir()

        return {
            "session_id": session_id,
            "restored_at": now,
            "restored_event_count": len(restored_lines),
            "archive_key": archive_key,
        }

    async def run(self) -> list[dict[str, Any]]:
        """Scan for idle sessions and compact each one. Returns list of manifests."""
        idle = self.find_idle_sessions()
        results: list[dict[str, Any]] = []
        for session_id in idle:
            manifest = await self.compact_session(session_id)
            results.append(manifest)
        return results


# ---------------------------------------------------------------------------
# Background compaction loop (started by app lifespan)
# ---------------------------------------------------------------------------


async def run_compaction_loop(
    storage_root: Path,
    policy: CompactionConfig,
    audit_log: AuditLog,
    *,
    check_interval_seconds: int = 86400,
) -> None:
    """
    Periodic background task: runs the auto-compaction policy every
    ``check_interval_seconds`` seconds (default 24 h).  Runs once
    immediately on startup, then sleeps between iterations.
    """
    compactor = AutoCompactor(
        storage_root,
        idle_days=policy.idle_days,
        tail_events=policy.tail_events,
    )
    while True:
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "compaction.auto_run",
            attributes={"compaction.idle_days": policy.idle_days},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="compaction.auto_run.invocation",
                    code="compaction_auto_run",
                    timestamp=now,
                ),
            )
            try:
                results = await compactor.run()
                span.set_attribute("compaction.session_count", len(results))
            except Exception as exc:
                err = CompactionError(
                    message=f"Auto-compaction run failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="compaction.auto_run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message},
                    )
                )

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(check_interval_seconds)


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_compaction_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    policy: CompactionConfig,
) -> APIRouter:
    router = APIRouter()
    compactor = AutoCompactor(
        storage_root,
        idle_days=policy.idle_days,
        tail_events=policy.tail_events,
    )

    @router.post("/v1/x/compaction/sessions/{session_id}", status_code=200)
    async def compact_session(session_id: str) -> JSONResponse:
        now = _now()
        run_id = f"compact_{uuid.uuid4().hex}"
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "compaction.compact_session",
            attributes={
                "compaction.session_id": session_id,
                "compaction.run_id": run_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="compaction.compact_session.invocation",
                    code="compact_session",
                    timestamp=now,
                ),
            )

            try:
                manifest = await compactor.compact_session(session_id)
            except CompactionSessionNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="compaction.compact_session.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err.message,
                        },
                    )
                )
                raise
            except Exception as exc:
                err2 = CompactionError(
                    message=f"Failed to compact session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="compaction.compact_session.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=manifest, status_code=200)

    @router.post("/v1/x/sessions/{session_id}/archive", status_code=200)
    async def archive_session(session_id: str) -> JSONResponse:
        now = _now()
        run_id = f"archive_{uuid.uuid4().hex}"
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "sessions.archive_session",
            attributes={
                "sessions.session_id": session_id,
                "sessions.run_id": run_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="sessions.archive_session.invocation",
                    code="archive_session",
                    timestamp=now,
                ),
            )

            try:
                manifest = await compactor.compact_session(session_id)
            except CompactionSessionNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="sessions.archive_session.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err.message,
                        },
                    )
                )
                raise
            except Exception as exc:
                err2 = CompactionError(
                    message=f"Failed to archive session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="sessions.archive_session.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=manifest, status_code=200)

    @router.post("/v1/x/sessions/{session_id}/restore", status_code=200)
    async def restore_session(session_id: str) -> JSONResponse:
        now = _now()
        run_id = f"restore_{uuid.uuid4().hex}"
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "sessions.restore_session",
            attributes={
                "sessions.session_id": session_id,
                "sessions.run_id": run_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="sessions.restore_session.invocation",
                    code="restore_session",
                    timestamp=now,
                ),
            )

            try:
                result = await compactor.restore_session(session_id)
            except RestoreSessionNotArchivedError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="sessions.restore_session.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err.message,
                        },
                    )
                )
                raise
            except Exception as exc:
                err2 = RestoreError(
                    message=f"Failed to restore session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="sessions.restore_session.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=result, status_code=200)

    @router.get("/v1/x/compaction/policy", status_code=200)
    async def get_policy() -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span("compaction.get_policy") as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="compaction.get_policy.invocation",
                    code="get_policy",
                    timestamp=now,
                ),
            )
            return JSONResponse(
                content={
                    "enabled": policy.enabled,
                    "idle_days": policy.idle_days,
                    "summary_strategy": policy.summary_strategy,
                    "tail_events": policy.tail_events,
                    "retention_days": policy.retention_days,
                },
                status_code=200,
            )

    @router.post("/v1/x/compaction/run", status_code=200)
    async def run_compaction() -> JSONResponse:
        now = _now()
        run_id = f"compact_{uuid.uuid4().hex}"
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "compaction.run",
            attributes={"compaction.run_id": run_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="compaction.run.invocation",
                    code="compaction_run",
                    timestamp=now,
                ),
            )

            try:
                results = await compactor.run()
                span.set_attribute("compaction.session_count", len(results))
            except Exception as exc:
                err = CompactionError(
                    message=f"Compaction run failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="compaction.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "run_id": run_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "run_id": run_id,
                "compacted_count": len(results),
                "results": results,
            },
            status_code=200,
        )

    return router
