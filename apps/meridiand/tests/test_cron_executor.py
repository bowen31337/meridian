"""Tests for CronExecutor — running a fired cron as an agent turn."""

from __future__ import annotations

from typing import Any

from meridiand._cron_executor import CronExecutor


class _FakeResponder:
    def __init__(self, *, reply: str | None = "ok", raise_exc: Exception | None = None) -> None:
        self._reply = reply
        self._raise = raise_exc
        self.calls: list[tuple[str, str, str]] = []

    async def run_prompt(
        self, channel_id: str, prompt: str, *, session_id: str = "", recipient: str = "cron"
    ) -> str | None:
        self.calls.append((channel_id, prompt, session_id))
        if self._raise is not None:
            raise self._raise
        return self._reply


def _resource(**meta: Any) -> dict[str, Any]:
    return {"id": "cron_1", "session_id": "sess_1", "metadata": meta or None}


async def test_completed_runs_the_turn() -> None:
    resp = _FakeResponder(reply="summary done")
    out = await CronExecutor(responder=resp)(_resource(prompt="summarize", channel_id="ch1"))
    assert out["status"] == "completed"
    assert out["output"] == "summary done"
    assert resp.calls == [("ch1", "summarize", "sess_1")]


async def test_channel_id_from_top_level_field() -> None:
    resp = _FakeResponder()
    resource = {"id": "c", "session_id": "s", "channel_id": "ch_top", "metadata": {"prompt": "go"}}
    out = await CronExecutor(responder=resp)(resource)
    assert out["status"] == "completed"
    assert resp.calls[0][0] == "ch_top"


async def test_skipped_without_prompt() -> None:
    resp = _FakeResponder()
    out = await CronExecutor(responder=resp)(_resource(channel_id="ch1"))
    assert out["status"] == "skipped"
    assert "prompt" in out["reason"]
    assert resp.calls == []  # never ran the turn


async def test_skipped_without_channel() -> None:
    resp = _FakeResponder()
    out = await CronExecutor(responder=resp)(_resource(prompt="do it"))
    assert out["status"] == "skipped"
    assert "channel_id" in out["reason"]
    assert resp.calls == []


async def test_error_is_reported_not_raised() -> None:
    resp = _FakeResponder(raise_exc=RuntimeError("model down"))
    out = await CronExecutor(responder=resp)(_resource(prompt="go", channel_id="ch1"))
    assert out["status"] == "error"
    assert "model down" in out["error"]


async def test_none_reply_yields_empty_output() -> None:
    resp = _FakeResponder(reply=None)
    out = await CronExecutor(responder=resp)(_resource(prompt="go", channel_id="ch1"))
    assert out["status"] == "completed"
    assert out["output"] == ""


async def test_kind_handler_is_delegated_to() -> None:
    resp = _FakeResponder()
    seen: list[dict[str, Any]] = []

    async def handler(resource: dict[str, Any]) -> dict[str, Any]:
        seen.append(resource)
        return {"status": "completed", "output": "maintained repo X"}

    out = await CronExecutor(responder=resp, kind_handlers={"maintenance": handler})(
        _resource(kind="maintenance")
    )
    assert out == {"status": "completed", "output": "maintained repo X"}
    assert seen and seen[0]["metadata"]["kind"] == "maintenance"
    assert resp.calls == []  # the prompt path was not taken


async def test_kind_handler_error_is_reported_not_raised() -> None:
    async def handler(resource: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("git exploded")

    out = await CronExecutor(responder=_FakeResponder(), kind_handlers={"maintenance": handler})(
        _resource(kind="maintenance")
    )
    assert out["status"] == "error"
    assert "git exploded" in out["error"]


async def test_unmatched_kind_falls_through_to_prompt() -> None:
    resp = _FakeResponder(reply="normal reply")
    out = await CronExecutor(responder=resp, kind_handlers={"maintenance": _unused_handler})(
        _resource(kind="other", prompt="hello", channel_id="ch1")
    )
    assert out["status"] == "completed"
    assert out["output"] == "normal reply"
    assert resp.calls == [("ch1", "hello", "sess_1")]


async def _unused_handler(resource: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
    raise AssertionError("handler should not be called for a non-matching kind")
