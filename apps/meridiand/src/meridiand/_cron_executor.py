"""Cron fire executor: turns a fired cron into an actual agent turn.

The scheduler (``_cron_scheduler``) only records that a cron fired. This executor
closes the loop: given the fired cron resource it runs the agent and delivers the
reply, returning an outcome the scheduler stamps onto the fire record.

A cron carries a ``session_id`` and ``capabilities``; the actionable instruction
and delivery target live in ``metadata``::

    {"prompt": "summarise today's PRs", "channel_id": "ch_telegram_main"}

``prompt`` is required to do anything; ``channel_id`` (or the top-level
``channel_id`` field) names the channel the reply is delivered to. A cron
missing either is reported ``skipped`` rather than executed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class _PromptRunner(Protocol):
    async def run_prompt(
        self, channel_id: str, prompt: str, *, session_id: str = ..., recipient: str = ...
    ) -> str | None: ...


# A specialised handler for a cron whose ``metadata.kind`` matches its key.
_KindHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

_MAX_RECORDED_OUTPUT = 1000


class CronExecutor:
    """Runs a fired cron as an agent turn via the responder, reporting the outcome.

    Crons whose ``metadata.kind`` matches a registered handler (e.g. the
    deterministic ``maintenance`` harness) are delegated to that handler instead
    of running a free-form prompt turn.
    """

    def __init__(
        self,
        *,
        responder: _PromptRunner,
        kind_handlers: dict[str, _KindHandler] | None = None,
    ) -> None:
        self._responder = responder
        self._kinds = kind_handlers or {}

    async def __call__(self, resource: dict[str, Any]) -> dict[str, Any]:
        """Execute *resource*; returns ``{"status": ...}`` (+ output/error/reason)."""
        meta = resource.get("metadata") or {}
        kind = meta.get("kind")
        handler = self._kinds.get(kind) if isinstance(kind, str) else None
        if handler is not None:
            try:
                return await handler(resource)
            except Exception as exc:  # noqa: BLE001 - any failure is reported, never raised
                return {"status": "error", "error": str(exc)}
        prompt = str(meta.get("prompt") or "").strip()
        channel_id = resource.get("channel_id") or meta.get("channel_id")
        if not prompt:
            return {"status": "skipped", "reason": "no 'prompt' in cron metadata"}
        if not channel_id:
            return {"status": "skipped", "reason": "no 'channel_id' for delivery"}
        try:
            reply = await self._responder.run_prompt(
                str(channel_id), prompt, session_id=str(resource.get("session_id") or "")
            )
        except Exception as exc:  # noqa: BLE001 - any failure is reported, never raised
            return {"status": "error", "error": str(exc)}
        return {"status": "completed", "output": (reply or "")[:_MAX_RECORDED_OUTPUT]}
