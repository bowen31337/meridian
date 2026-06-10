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
import hashlib
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

# Conversation memory: keep the last N turns (user+assistant) per sender so the
# agent has continuity within a chat. Trimmed to bound token usage.
_MAX_HISTORY_MESSAGES = 20


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
        memory_top_k: int = 5,
        extract_facts: bool = True,
        extract_model: str = "claude:claude-haiku-4-5",
        max_facts: int = 3,
        audit_log: AuditLog | None = None,
        base_url: str = "http://meridian.internal",
    ) -> None:
        self._router = model_router
        self._model = model
        self._storage_root = storage_root
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._memory_top_k = memory_top_k
        self._extract_enabled = extract_facts
        self._extract_model = extract_model
        self._max_facts = max_facts
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

    def _load_agent_context(self, channel_id: str) -> dict[str, Any] | None:
        """Resolve the channel's Agent tool context (id, tools, workspace) for the provider."""
        if self._storage_root is None:
            return None
        try:
            channel = json.loads(
                (self._storage_root / "channels" / f"{channel_id}.json").read_text()
            )
            agent_id = channel.get("default_agent_id")
            if not agent_id:
                return None
            agent = json.loads((self._storage_root / "agents" / f"{agent_id}.json").read_text())
            version = agent.get("version") or {}
            tools = [t.get("name") for t in version.get("tools", []) if t.get("name")]
            if not tools:
                return None
            workspace = str(self._storage_root)
            env_id = agent.get("default_environment_id") or version.get("default_environment_id")
            if env_id:
                env_file = self._storage_root / "environments" / f"{env_id}.json"
                if env_file.exists():
                    workspace = json.loads(env_file.read_text()).get("workspace_path") or workspace
            return {
                "agent_id": agent_id,
                "storage_root": str(self._storage_root),
                "tools": tools,
                "workspace": workspace,
            }
        except Exception:  # noqa: BLE001 - no tool context -> plain text reply
            return None

    def _history_path(self, channel_id: str, sender_id: str) -> Path:
        assert self._storage_root is not None
        return self._storage_root / "conversations" / channel_id / f"{sender_id}.json"

    def _load_history(self, channel_id: str, sender_id: str) -> list[dict[str, str]]:
        """Load this sender's recent conversation turns (oldest first)."""
        if self._storage_root is None:
            return []
        try:
            path = self._history_path(channel_id, sender_id)
            if path.exists():
                data = json.loads(path.read_text())
                return data if isinstance(data, list) else []
        except Exception:  # noqa: BLE001 - corrupt/missing history -> start fresh
            return []
        return []

    def _save_history(self, channel_id: str, sender_id: str, history: list[dict[str, str]]) -> None:
        if self._storage_root is None:
            return
        try:
            path = self._history_path(channel_id, sender_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(history[-_MAX_HISTORY_MESSAGES:]))
        except Exception as exc:  # noqa: BLE001 - persistence failure is non-fatal
            self._audit_failure(channel_id, "memory", exc)

    def _load_memory_store_id(self, channel_id: str) -> str | None:
        """Resolve the agent's first attached long-term MemoryStore, if any."""
        if self._storage_root is None:
            return None
        try:
            channel = json.loads(
                (self._storage_root / "channels" / f"{channel_id}.json").read_text()
            )
            agent_id = channel.get("default_agent_id")
            if not agent_id:
                return None
            agent = json.loads((self._storage_root / "agents" / f"{agent_id}.json").read_text())
            refs = (agent.get("version") or {}).get("memory_store_refs") or []
            return refs[0] if refs else None
        except Exception:  # noqa: BLE001 - no memory store -> skip long-term recall
            return None

    async def _retrieve_memories(
        self, client: httpx.AsyncClient, channel_id: str, store_id: str, query: str
    ) -> list[str]:
        """Hybrid-retrieve durable memories relevant to the current message."""
        try:
            resp = await client.post(
                f"/v1/memory_stores/{store_id}/query_runs",
                json={"query": query, "limit": self._memory_top_k},
            )
            if resp.is_success:
                return [
                    r.get("content", "") for r in resp.json().get("results", []) if r.get("content")
                ]
        except Exception as exc:  # noqa: BLE001 - retrieval failure -> reply without recall
            self._audit_failure(channel_id, "memory_retrieve", exc)
        return []

    async def _write_memory(
        self,
        client: httpx.AsyncClient,
        channel_id: str,
        store_id: str,
        content: str,
        *,
        dialectic: bool = False,
    ) -> None:
        """Persist a durable memory (content-keyed for dedup; optional dialectic reconcile)."""
        try:
            key = "tg-" + hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
            await client.post(
                f"/v1/memory_stores/{store_id}/write",
                json={"key": key, "content": content, "dialectic": dialectic},
            )
        except Exception as exc:  # noqa: BLE001 - write failure is audited, not fatal
            self._audit_failure(channel_id, "memory_write", exc)

    async def _extract_facts(self, channel_id: str, user_content: str, reply: str) -> list[str]:
        """Use a cheap model to extract durable facts about the user from the exchange."""
        system = (
            "You extract durable, long-term facts about the user worth remembering across "
            "conversations — identity, stable preferences, projects, goals, constraints. "
            f"Output ONLY a JSON array of at most {self._max_facts} short standalone fact "
            "strings about the user. If there is nothing durable, output []. No prose, no fences."
        )
        prompt = f"User message: {user_content}\nAssistant reply: {reply}\n\nReturn the JSON array."
        opts = ModelCallOpts(
            model=self._extract_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            system=system,
            temperature=None,
            tools=[],
            # Tag so a routing rule can send extraction to a cheap model.
            metadata={"memory_op": "extract"},
            stream=False,
        )
        parts: list[str] = []
        try:
            async for event in self._router.call(opts):
                if getattr(event, "type", None) == "text_delta":
                    parts.append(event.text)
        except Exception as exc:  # noqa: BLE001 - extraction failure -> remember nothing
            self._audit_failure(channel_id, "memory_extract", exc)
            return []
        return self._parse_facts("".join(parts))

    def _parse_facts(self, text: str) -> list[str]:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        try:
            data = json.loads(text[start : end + 1])  # always a list when valid (starts with [)
        except Exception:  # noqa: BLE001 - non-JSON output -> no facts
            return []
        facts = [str(f).strip() for f in data if isinstance(f, str) and str(f).strip()]
        return facts[: self._max_facts]

    async def _generate_reply(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        tool_context: dict[str, Any] | None,
    ) -> str:
        metadata: dict[str, Any] = {}
        if tool_context is not None:
            metadata["meridian_tools"] = tool_context
        opts = ModelCallOpts(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            system=system_prompt,
            temperature=None,
            tools=[],
            metadata=metadata,
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

            history = self._load_history(channel_id, sender_id)
            history.append({"role": "user", "content": content})

            # Long-term memory: retrieve durable facts relevant to this message and
            # fold them into the persona prompt.
            persona = self._load_persona(channel_id)
            store_id = self._load_memory_store_id(channel_id)
            if store_id is not None:
                memories = await self._retrieve_memories(client, channel_id, store_id, content)
                if memories:
                    persona += "\n\nThings you remember about the user (use when relevant):\n- "
                    persona += "\n- ".join(memories)

            model_ok = True
            try:
                reply = await self._generate_reply(
                    history, persona, self._load_agent_context(channel_id)
                )
            except Exception as exc:  # noqa: BLE001 - model failure -> fallback reply
                self._audit_failure(channel_id, "model", exc)
                reply = _FALLBACK_REPLY
                model_ok = False
            if not reply:
                reply = _FALLBACK_REPLY
                model_ok = False

            if store_id is not None and model_ok:
                if self._extract_enabled:
                    for fact in await self._extract_facts(channel_id, content, reply):
                        await self._write_memory(
                            client, channel_id, store_id, fact, dialectic=True
                        )
                else:
                    await self._write_memory(client, channel_id, store_id, content)

            # Persist the user turn always; the assistant turn only on a real reply,
            # so a transient failure can be retried next turn with context intact.
            if model_ok:
                history.append({"role": "assistant", "content": reply})
            self._save_history(channel_id, sender_id, history)

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
