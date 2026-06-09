"""Agent responder: turns an inbound channel message into an LLM reply.

Wired as the long-poll inbound sink. For each decoded message it:
  1. POSTs to the daemon's own ``/v1/channels/{id}/inbound`` (creates the
     routing session, applies pairing/quarantine policy) via in-process ASGI;
  2. runs the configured LLM through the Model Router with an agent persona;
  3. POSTs the reply to ``/v1/channels/{id}/outbound`` so it is delivered back
     over the channel.

Best-effort: a model failure (e.g. rate limit) is audited and a fallback
message is sent so the user always gets a response. The app is bound after
construction to break the app <-> runtime <-> sink construction cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
import httpx
from meridian_sdk_provider import ModelCallOpts, ModelRouter
from starlette.types import ASGIApp

_DEFAULT_SYSTEM_PROMPT = (
    "You are Meridian, a helpful personal assistant replying over Telegram. "
    "Be concise, friendly, and direct. Plain text only — no markdown."
)
_FALLBACK_REPLY = "Sorry — I couldn't reach the model just now. Please try again."


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AgentResponder:
    """UpdateSink that replies to inbound messages with an LLM completion."""

    def __init__(
        self,
        *,
        model_router: ModelRouter,
        model: str,
        storage_root: Path | None = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 1024,
        audit_log: AuditLog | None = None,
        base_url: str = "http://meridian.internal",
    ) -> None:
        self._router = model_router
        self._model = model
        self._storage_root = storage_root
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._audit = audit_log or NoopAuditLog()
        self._base_url = base_url
        self._app: ASGIApp | None = None

    def bind(self, app: ASGIApp) -> None:
        self._app = app

    def _load_persona(self, channel_id: str) -> str:
        """Resolve the channel's agent instructions (persona), or the default."""
        if self._storage_root is None:
            return self._system_prompt
        try:
            channel = json.loads(
                (self._storage_root / "channels" / f"{channel_id}.json").read_text()
            )
            agent_id = channel.get("default_agent_id")
            if not agent_id:
                return self._system_prompt
            agent = json.loads((self._storage_root / "agents" / f"{agent_id}.json").read_text())
            instructions = (agent.get("version") or {}).get("instructions")
            return instructions or self._system_prompt
        except Exception:  # noqa: BLE001 - fall back to the default persona
            return self._system_prompt

    async def _generate_reply(self, content: str, system_prompt: str) -> str:
        opts = ModelCallOpts(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self._max_tokens,
            system=system_prompt,
            temperature=None,
            tools=[],
            metadata={},
            stream=False,
        )
        parts: list[str] = []
        async for event in self._router.call(opts):
            if getattr(event, "type", None) == "text_delta":
                parts.append(event.text)
        return "".join(parts).strip()

    async def dispatch(
        self, *, channel_id: str, sender_id: str, content: str, content_type: str
    ) -> None:
        if self._app is None:
            return
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(transport=transport, base_url=self._base_url) as client:
            session_id = ""
            try:
                inbound = await client.post(
                    f"/v1/channels/{channel_id}/inbound",
                    json={"sender_id": sender_id, "content": content, "content_type": content_type},
                )
                if inbound.is_success:
                    session_id = inbound.json().get("session_id", "")
            except Exception as exc:  # noqa: BLE001 - inbound failure shouldn't block a reply
                self._audit_failure(channel_id, "inbound", exc)

            try:
                reply = await self._generate_reply(content, self._load_persona(channel_id))
            except Exception as exc:  # noqa: BLE001 - model failure -> fallback reply
                self._audit_failure(channel_id, "model", exc)
                reply = _FALLBACK_REPLY
            if not reply:
                reply = _FALLBACK_REPLY

            try:
                await client.post(
                    f"/v1/channels/{channel_id}/outbound",
                    json={"session_id": session_id, "recipient": sender_id, "content": reply},
                )
            except Exception as exc:  # noqa: BLE001 - outbound failure is audited, not fatal
                self._audit_failure(channel_id, "outbound", exc)

    def _audit_failure(self, channel_id: str, stage: str, exc: Exception) -> None:
        detail: dict[str, Any] = {"channel_id": channel_id, "stage": stage, "message": str(exc)}
        self._audit.write(
            AuditLogEntry(
                level="error",
                event="agent.responder.failed",
                code="agent_responder_failed",
                timestamp=_now(),
                detail=detail,
            )
        )
