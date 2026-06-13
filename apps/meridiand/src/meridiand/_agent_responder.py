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

import contextlib
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
import httpx
from meridian_sdk_provider import ModelCallOpts, ModelRouter
from sdk_capabilities import parse as parse_capability
from starlette.types import ASGIApp

from ._agent_tools import _root_from_param
from ._intelligent_router import classify_tier
from ._telegram_commands import help_text, start_text

_DEFAULT_SYSTEM_PROMPT = (
    "You are Meridian, a helpful personal assistant replying over Telegram. "
    "Be concise, friendly, and direct. Plain text only — no markdown."
)
_FALLBACK_REPLY = "Sorry — I couldn't reach the model just now. Please try again."

# Conversation memory: keep the last N turns (user+assistant) per sender so the
# agent has continuity within a chat. Trimmed to bound token usage.
_MAX_HISTORY_MESSAGES = 20

# Native (CLI-backed) web tools and the capability that gates them. They reach
# the network, so they are only forwarded to the provider when the agent holds a
# matching net.fetch grant.
_NATIVE_WEB_TOOLS = frozenset({"web_search", "web_fetch"})
_WEB_TOOL_CAP = "net.fetch"

# Intelligent routing: each message is sorted into a capability tier (see
# _intelligent_router.classify_tier — a deterministic 15-dimension scorer ported
# from openclaw) and the tier is attached as reply metadata so config routing
# rules size the model to the task instead of over-/under-skilling it.


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _granted_fs_dirs(caps: list[str], workspace: str) -> list[str]:
    """Absolute fs.read/fs.write roots the agent was granted, excluding *workspace*.

    These become ``--add-dir`` args so the Claude Code CLI permits its tools to
    reach folders outside the working directory without an interactive prompt.
    """
    ws = str(Path(workspace).resolve())
    dirs: list[str] = []
    for cap in caps:
        try:
            parsed = parse_capability(cap)
        except Exception:  # noqa: BLE001 - skip unparseable grants
            continue
        if parsed.namespace == "fs" and parsed.name in ("read", "write") and parsed.param:
            root = _root_from_param(parsed.param)
            if root and str(Path(root).resolve()) != ws and root not in dirs:
                dirs.append(root)
    return dirs


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
        intelligent_routing: bool = False,
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
        self._intelligent_routing = intelligent_routing
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
            # Capability gate (§6): the network-backed web tools are only forwarded
            # when the agent actually holds the net.fetch grant.
            caps = version.get("capabilities") or []
            if not any(str(c).startswith(_WEB_TOOL_CAP) for c in caps):
                tools = [t for t in tools if t not in _NATIVE_WEB_TOOLS]
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
                # Granted fs roots beyond the workspace, so the provider can pass
                # them to the CLI as --add-dir (else access prompts in headless).
                "extra_dirs": _granted_fs_dirs(caps, workspace),
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

    def _load_active_skills(self, channel_id: str) -> list[tuple[str, str]]:
        """Resolve the channel agent's *active* skills to (name, instructions).

        Honors explicit per-agent activation (PRD F-SK-5): only skills whose
        activation status is "active" for this channel's agent are injected — a
        pending or revoked activation has no effect on the reply.
        """
        if self._storage_root is None:
            return []
        try:
            channel = json.loads(
                (self._storage_root / "channels" / f"{channel_id}.json").read_text()
            )
            agent_id = channel.get("default_agent_id")
            if not agent_id:
                return []
            activations_dir = self._storage_root / "skill_activations"
            if not activations_dir.exists():
                return []
            skills: list[tuple[str, str]] = []
            for path in sorted(activations_dir.glob("*.json")):
                try:
                    act = json.loads(path.read_text())
                except Exception:  # noqa: BLE001 - skip an unreadable activation record
                    continue
                if act.get("agent_id") != agent_id or act.get("status") != "active":
                    continue
                resolved = self._resolve_skill_instructions(
                    act.get("skill_id"), act.get("skill_version_id")
                )
                if resolved is not None:
                    skills.append(resolved)
            return skills
        except Exception:  # noqa: BLE001 - no skills -> reply without them
            return []

    def _resolve_skill_instructions(
        self, skill_id: str | None, version_id: str | None
    ) -> tuple[str, str] | None:
        """Load a skill's (name, instructions), preferring the pinned version."""
        if self._storage_root is None or not skill_id:
            return None
        name = skill_id
        instructions = ""
        if version_id:
            vpath = self._storage_root / "skill_versions" / f"{version_id}.json"
            if vpath.exists():
                instructions = json.loads(vpath.read_text()).get("instructions", "") or ""
        spath = self._storage_root / "skills" / f"{skill_id}.json"
        if spath.exists():
            rec = json.loads(spath.read_text())
            name = rec.get("name") or name
            if not instructions:
                instructions = (rec.get("version") or {}).get("instructions", "") or ""
        instructions = instructions.strip()
        if not instructions:
            return None
        return (name, instructions)

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
        route_tier: str | None = None,
    ) -> str:
        metadata: dict[str, Any] = {}
        if tool_context is not None:
            metadata["meridian_tools"] = tool_context
            # Flag so routing rules can keep tool-bearing replies on a
            # tool-capable provider: only the claude_code_oauth provider builds
            # the MCP tool bridge, so the GLM/llama tiers would silently drop
            # tool access.
            metadata["agent_has_tools"] = True
        if route_tier is not None:
            metadata["route_tier"] = route_tier
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

    def _clear_history(self, channel_id: str, sender_id: str) -> None:
        """Drop this sender's short-term conversation context (for /new)."""
        if self._storage_root is None:
            return
        with contextlib.suppress(Exception):
            path = self._history_path(channel_id, sender_id)
            path.unlink(missing_ok=True)

    async def _handle_command(
        self,
        client: httpx.AsyncClient,
        channel_id: str,
        sender_id: str,
        content: str,
        store_id: str | None,
    ) -> str | None:
        """Handle an openclaw-style slash command locally.

        Returns the reply text for a recognized command, or None to let the
        message flow to the LLM. A bare unknown ``/command`` (no arguments) gets
        a help nudge; a slash followed by prose (e.g. "/etc is missing") is
        treated as a normal message so real questions are never swallowed.
        """
        stripped = content.strip()
        if not stripped.startswith("/"):
            return None
        head, _, rest = stripped.partition(" ")
        # Strip the leading slash and any @botname suffix Telegram appends in groups.
        cmd = head[1:].split("@", 1)[0].lower()
        args = rest.strip()

        if cmd == "start":
            return start_text()
        if cmd == "help":
            return "Commands:\n" + help_text()
        if cmd in ("new", "reset", "clear"):
            self._clear_history(channel_id, sender_id)
            return "Fresh start — I've cleared our short-term conversation context."
        if cmd == "whoami":
            return f"You're paired with me as sender {sender_id} on this channel."
        if cmd == "remember":
            if not args:
                return "Tell me what to remember, e.g. /remember I prefer concise answers."
            if store_id is None:
                return "I don't have a memory store configured, so I can't save that."
            await self._write_memory(client, channel_id, store_id, args, dialectic=True)
            return f"Got it — I'll remember that: {args}"
        # Unknown bare command -> nudge; slash-with-prose -> fall through to the LLM.
        if not args:
            return f"Unknown command /{cmd}. Send /help to see what I can do."
        return None

    async def _send_outbound(
        self,
        client: httpx.AsyncClient,
        channel_id: str,
        session_id: str,
        sender_id: str,
        reply: str,
    ) -> None:
        try:
            await client.post(
                f"/v1/channels/{channel_id}/outbound",
                json={"session_id": session_id, "recipient": sender_id, "content": reply},
            )
        except Exception as exc:  # noqa: BLE001 - outbound failure is audited, not fatal
            self._audit_failure(channel_id, "outbound", exc)

    async def deliver_text(
        self,
        channel_id: str,
        text: str,
        *,
        session_id: str = "",
        recipient: str = "cron",
    ) -> None:
        """Send a plain message to a channel without running the model.

        Used by system harnesses (e.g. the maintenance executor) that compose
        their own status text and need to deliver it through the channel's
        outbound route. No-op when the responder is unbound.
        """
        if self._app is None:
            return
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(transport=transport, base_url=self._base_url) as client:
            await self._send_outbound(client, channel_id, session_id, recipient, text)

    async def run_prompt(
        self,
        channel_id: str,
        prompt: str,
        *,
        session_id: str = "",
        recipient: str = "cron",
        deliver: bool = True,
    ) -> str | None:
        """Run a system-triggered agent turn (e.g. a cron fire) and deliver it.

        Unlike dispatch(), there is no inbound/pairing step — the trigger is
        system-authorized, not an external sender. Reuses the persona, skills,
        long-term memory, intelligent routing and channel delivery of a normal
        reply. Returns the reply text, or None if the responder is unbound.

        When ``deliver`` is False the reply is generated (and any file-editing
        tools run) but not sent to the channel — the caller takes responsibility
        for delivery. Used by the maintenance harness, which runs an edit-only
        turn and composes its own status message.
        """
        if self._app is None:
            return None
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(transport=transport, base_url=self._base_url) as client:
            persona = self._load_persona(channel_id)
            skills = self._load_active_skills(channel_id)
            if skills:
                persona += "\n\nActive skills — apply these when relevant:"
                for skill_name, skill_instructions in skills:
                    persona += f"\n\n## {skill_name}\n{skill_instructions}"
            store_id = self._load_memory_store_id(channel_id)
            if store_id is not None:
                memories = await self._retrieve_memories(client, channel_id, store_id, prompt)
                if memories:
                    persona += "\n\nThings you remember about the user (use when relevant):\n- "
                    persona += "\n- ".join(memories)

            route_tier = classify_tier(prompt) if self._intelligent_routing else None
            try:
                reply = await self._generate_reply(
                    [{"role": "user", "content": prompt}],
                    persona,
                    self._load_agent_context(channel_id),
                    route_tier,
                )
            except Exception as exc:  # noqa: BLE001 - model failure -> fallback reply
                self._audit_failure(channel_id, "cron_model", exc)
                reply = _FALLBACK_REPLY
            if not reply:
                reply = _FALLBACK_REPLY

            if deliver:
                await self._send_outbound(client, channel_id, session_id, recipient, reply)
            return reply

    async def dispatch(
        self,
        *,
        channel_id: str,
        sender_id: str,
        content: str,
        content_type: str,
        quote: str | None = None,
    ) -> None:
        if self._app is None:
            return
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(transport=transport, base_url=self._base_url) as client:
            try:
                inbound = await client.post(
                    f"/v1/channels/{channel_id}/inbound",
                    json={"sender_id": sender_id, "content": content, "content_type": content_type},
                )
            except Exception as exc:  # noqa: BLE001 - cannot verify sender -> fail closed
                self._audit_failure(channel_id, "inbound", exc)
                return

            # Allowlist enforcement: only reply when the channel's inbound policy
            # accepted this sender. paired_only returns 403 for non-allowlisted
            # senders; quarantine accepts but flags untrusted. In both cases the
            # agent must NOT run (no LLM, no tools, no memory) — fail closed.
            if not inbound.is_success:
                self._audit_failure(
                    channel_id,
                    "inbound_rejected",
                    RuntimeError(f"sender {sender_id!r} not allowed (policy rejected)"),
                )
                return
            inbound_data = inbound.json()
            if inbound_data.get("quarantined"):
                self._audit_failure(
                    channel_id,
                    "inbound_quarantined",
                    RuntimeError(f"sender {sender_id!r} quarantined"),
                )
                return
            session_id = inbound_data.get("session_id", "")
            store_id = self._load_memory_store_id(channel_id)

            # Slash commands (openclaw-style) are handled locally — no LLM, no
            # history pollution — and replied to directly.
            command_reply = await self._handle_command(
                client, channel_id, sender_id, content, store_id
            )
            if command_reply is not None:
                await self._send_outbound(client, channel_id, session_id, sender_id, command_reply)
                return

            # Fold any quoted/replied text in as the nearest context for this turn.
            user_content = content
            if quote:
                user_content = f"[Replying to: {quote}]\n\n{content}"

            history = self._load_history(channel_id, sender_id)
            history.append({"role": "user", "content": user_content})

            # Active skills: fold each granted skill's instructions into the prompt.
            persona = self._load_persona(channel_id)
            skills = self._load_active_skills(channel_id)
            if skills:
                persona += "\n\nActive skills — apply these when relevant:"
                for skill_name, skill_instructions in skills:
                    persona += f"\n\n## {skill_name}\n{skill_instructions}"

            # Long-term memory: retrieve durable facts relevant to this message and
            # fold them into the persona prompt.
            if store_id is not None:
                memories = await self._retrieve_memories(client, channel_id, store_id, user_content)
                if memories:
                    persona += "\n\nThings you remember about the user (use when relevant):\n- "
                    persona += "\n- ".join(memories)

            # Intelligent routing: deterministically score the task tier so config
            # rules size the reply model to the work (avoid over-/under-skilling).
            route_tier: str | None = None
            if self._intelligent_routing:
                route_tier = classify_tier(user_content)

            model_ok = True
            try:
                reply = await self._generate_reply(
                    history, persona, self._load_agent_context(channel_id), route_tier
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
                    for fact in await self._extract_facts(channel_id, user_content, reply):
                        await self._write_memory(client, channel_id, store_id, fact, dialectic=True)
                else:
                    await self._write_memory(client, channel_id, store_id, user_content)

            # Persist the user turn always; the assistant turn only on a real reply,
            # so a transient failure can be retried next turn with context intact.
            if model_ok:
                history.append({"role": "assistant", "content": reply})
            self._save_history(channel_id, sender_id, history)

            await self._send_outbound(client, channel_id, session_id, sender_id, reply)

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
