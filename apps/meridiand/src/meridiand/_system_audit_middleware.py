"""System-level audit middleware for security-sensitive HTTP routes.

Monitors five operation categories and emits an append-only audit.ndjson entry
for every matched response:

    capability decisions  — POST /v1/skills, POST /v1/skills/install
    vault accesses        — POST /v1/vaults/{id}/secrets,
                            GET  /v1/vaults/{id}/secrets/{name}/meta
    channel pairings      — POST /v1/channels/{id}/pair
    skill promotions      — POST /v1/agents/{id}/skills/{id}/approve
    environment changes   — POST/PATCH/DELETE /v1/environments[/{id}]

On success (2xx) the entry is info-level; on failure (4xx/5xx or uncaught
exception) it is error-level.  On audit-write failure: surfaces an error to
the caller when no response has been sent yet; otherwise writes a
system_audit.write.failed entry to the audit log.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
from typing import Any

from core_errors import AuditLog, AuditLogEntry
from starlette.types import ASGIApp, Receive, Scope, Send


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class _Route:
    method: str
    pattern: re.Pattern[str]
    success_event: str
    success_code: str
    failure_event: str
    failure_code: str


_MONITORED_ROUTES: list[_Route] = [
    # Capability decisions
    _Route(
        method="POST",
        pattern=re.compile(r"^/v1/skills$"),
        success_event="capability.decision.skill.created",
        success_code="capability_skill_create",
        failure_event="capability.decision.skill.create.failed",
        failure_code="capability_skill_create_failed",
    ),
    _Route(
        method="POST",
        pattern=re.compile(r"^/v1/skills/install$"),
        success_event="capability.decision.skill.installed",
        success_code="capability_skill_install",
        failure_event="capability.decision.skill.install.failed",
        failure_code="capability_skill_install_failed",
    ),
    # Vault accesses
    _Route(
        method="POST",
        pattern=re.compile(r"^/v1/vaults/[^/]+/secrets$"),
        success_event="vault.access.secret.stored",
        success_code="vault_secret_store",
        failure_event="vault.access.secret.store.failed",
        failure_code="vault_secret_store_failed",
    ),
    _Route(
        method="GET",
        pattern=re.compile(r"^/v1/vaults/[^/]+/secrets/[^/]+/meta$"),
        success_event="vault.access.secret.meta.read",
        success_code="vault_secret_meta",
        failure_event="vault.access.secret.meta.failed",
        failure_code="vault_secret_meta_failed",
    ),
    # Channel pairings
    _Route(
        method="POST",
        pattern=re.compile(r"^/v1/channels/[^/]+/pair$"),
        success_event="channel.pairing.issued",
        success_code="channel_pair",
        failure_event="channel.pairing.failed",
        failure_code="channel_pair_failed",
    ),
    # Skill promotions
    _Route(
        method="POST",
        pattern=re.compile(r"^/v1/agents/[^/]+/skills/[^/]+/approve$"),
        success_event="skill.promotion.approved",
        success_code="skill_activation_approve",
        failure_event="skill.promotion.approve.failed",
        failure_code="skill_activation_approve_failed",
    ),
    # Environment changes
    _Route(
        method="POST",
        pattern=re.compile(r"^/v1/environments$"),
        success_event="environment.change.created",
        success_code="environment_create",
        failure_event="environment.change.create.failed",
        failure_code="environment_create_failed",
    ),
    _Route(
        method="PATCH",
        pattern=re.compile(r"^/v1/environments/[^/]+$"),
        success_event="environment.change.updated",
        success_code="environment_update",
        failure_event="environment.change.update.failed",
        failure_code="environment_update_failed",
    ),
    _Route(
        method="DELETE",
        pattern=re.compile(r"^/v1/environments/[^/]+$"),
        success_event="environment.change.deleted",
        success_code="environment_delete",
        failure_event="environment.change.delete.failed",
        failure_code="environment_delete_failed",
    ),
]


def _match_route(method: str, path: str) -> _Route | None:
    for route in _MONITORED_ROUTES:
        if method == route.method and route.pattern.match(path):
            return route
    return None


class SystemAuditMiddleware:
    """ASGI middleware that emits audit log entries for security-sensitive routes.

    Monitors:
    - Capability decisions: POST /v1/skills, POST /v1/skills/install
    - Vault accesses: POST /v1/vaults/{id}/secrets,
                      GET  /v1/vaults/{id}/secrets/{name}/meta
    - Channel pairings: POST /v1/channels/{id}/pair
    - Skill promotions: POST /v1/agents/{id}/skills/{id}/approve
    - Environment changes: POST/PATCH/DELETE /v1/environments[/{id}]

    On success (2xx): writes an info-level audit entry.
    On failure (4xx/5xx): surfaces the error response to the caller and writes
      an error-level audit entry.
    On uncaught exception: writes an error-level audit entry and re-raises so
      that ErrorEnvelopeMiddleware surfaces the error to the caller.
    On audit-write failure: if no response has been sent surfaces an error to
      the caller and writes a system_audit.write.failed entry; if the response
      was already sent writes a system_audit.write.failed entry best-effort.
    """

    def __init__(self, app: ASGIApp, *, audit_log: AuditLog) -> None:
        self._app = app
        self._audit_log = audit_log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        now = _now()

        route = _match_route(method, path)
        if route is None:
            await self._app(scope, receive, send)
            return

        status_captured: list[int] = []

        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start":
                status_captured.append(message["status"])
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        except Exception as exc:
            try:
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event=route.failure_event,
                        code=route.failure_code,
                        timestamp=_now(),
                        detail={"path": path, "method": method, "error": str(exc)},
                    )
                )
            except Exception as audit_exc:
                if not status_captured:
                    self._write_audit_failure(route.failure_event, audit_exc)
                    await self._send_error(
                        send,
                        500,
                        "system_audit_write_failed",
                        "Failed to write audit log entry",
                    )
                    return
            raise

        status = status_captured[0] if status_captured else None
        if status is None:
            return

        if status < 400:
            level: str = "info"
            event = route.success_event
            code = route.success_code
        else:
            level = "error"
            event = route.failure_event
            code = route.failure_code

        try:
            self._audit_log.write(
                AuditLogEntry(
                    level=level,
                    event=event,
                    code=code,
                    timestamp=now,
                    detail={"path": path, "method": method, "status": status},
                )
            )
        except Exception as audit_exc:
            self._write_audit_failure(event, audit_exc)

    def _write_audit_failure(self, original_event: str, exc: Exception) -> None:
        with contextlib.suppress(Exception):
            self._audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="system_audit.write.failed",
                    code="system_audit_write_failed",
                    timestamp=_now(),
                    detail={"error": str(exc), "original_event": original_event},
                )
            )

    async def _send_error(
        self,
        send: Send,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        body = json.dumps(
            {"error": {"code": code, "message": message}},
            separators=(",", ":"),
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )
