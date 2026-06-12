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

from typing import Any, Protocol


class _PromptRunner(Protocol):
    async def run_prompt(
        self, channel_id: str, prompt: str, *, session_id: str = ..., recipient: str = ...
    ) -> str | None: ...


_MAX_RECORDED_OUTPUT = 1000


class CronExecutor:
    """Runs a fired cron as an agent turn via the responder, reporting the outcome."""

    def __init__(self, *, responder: _PromptRunner) -> None:
        self._responder = responder

    async def __call__(self, resource: dict[str, Any]) -> dict[str, Any]:
        """Execute *resource*; returns ``{"status": ...}`` (+ output/error/reason)."""
        meta = resource.get("metadata") or {}
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
