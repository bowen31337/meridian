from __future__ import annotations

from collections.abc import AsyncIterator
import contextlib
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
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
from meridian_sdk_provider import (
    Message,
    ModelCallOpts,
    ModelRouter,
    ToolDefinition,
)
from meridian_sdk_provider.errors import ProviderError
from meridian_sdk_provider.types import (
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    TextDeltaEvent,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)
from pydantic import BaseModel
from sdk_sandbox import ExecutionContext

from ._hook_dispatch import dispatch_hooks


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class MessagesInferError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        http_status_code: int = 502,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="inference_error", message=message, timestamp=timestamp, cause=cause)
        self._http_status_code = http_status_code

    def http_status(self) -> int:
        return self._http_status_code


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class MessagesRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int
    system: str | None = None
    temperature: float | None = None
    tools: list[ToolDefinition] | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Event stream collector
# ---------------------------------------------------------------------------


async def _collect(
    event_stream: AsyncIterator[Any],
    request_model: str,
) -> dict[str, Any]:
    """Drain the ModelRouter event stream and return an Anthropic-shaped response dict."""
    model = request_model
    stop_reason: str | None = None
    input_tokens = 0
    output_tokens = 0
    text_parts: list[str] = []
    tool_blocks: list[dict[str, Any]] = []
    tool_json: dict[str, list[str]] = {}

    async for event in event_stream:
        if isinstance(event, MessageStartEvent):
            if event.model:
                model = event.model
            if event.input_tokens is not None:
                input_tokens = event.input_tokens
        elif isinstance(event, TextDeltaEvent):
            text_parts.append(event.text)
        elif isinstance(event, ToolUseStartEvent):
            tool_blocks.append(
                {"type": "tool_use", "id": event.id, "name": event.name, "input": {}}
            )
            tool_json[event.id] = []
        elif isinstance(event, ToolInputDeltaEvent):
            if event.id in tool_json:
                tool_json[event.id].append(event.partial_json)
        elif isinstance(event, MessageDeltaEvent):
            if event.stop_reason is not None:
                stop_reason = event.stop_reason
        elif isinstance(event, MessageStopEvent):
            if event.stop_reason is not None:
                stop_reason = event.stop_reason
            if event.input_tokens is not None:
                input_tokens = event.input_tokens
            if event.output_tokens is not None:
                output_tokens = event.output_tokens

    content: list[dict[str, Any]] = []
    if text_parts:
        content.append({"type": "text", "text": "".join(text_parts)})
    for block in tool_blocks:
        joined = "".join(tool_json.get(block["id"], []))
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            block["input"] = json.loads(joined) if joined else {}
        content.append(block)

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_messages_router(
    *, audit_log: AuditLog, model_router: ModelRouter, hooks_dir: Path | None = None
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/messages")
    async def create_message(body: MessagesRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "messages.infer",
            attributes={"model": body.model, "max_tokens": body.max_tokens},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="messages.infer.invocation",
                    code="messages_infer",
                    timestamp=now,
                ),
            )

            try:
                session_id = (body.metadata or {}).get("session_id", "") if body.metadata else ""
                if hooks_dir is not None:
                    await dispatch_hooks(
                        "on_model_call",
                        {
                            "session_id": session_id,
                            "model": body.model,
                            "max_tokens": body.max_tokens,
                        },
                        ExecutionContext(session_id=session_id),
                        hooks_dir=hooks_dir,
                        audit_log=audit_log,
                    )
                opts = ModelCallOpts(
                    model=body.model,
                    messages=body.messages,
                    max_tokens=body.max_tokens,
                    system=body.system,
                    temperature=body.temperature,
                    tools=body.tools or [],
                    metadata=body.metadata or {},
                    stream=False,
                )
                response = await _collect(model_router.call(opts), body.model)
            except ProviderError as exc:
                err = MessagesInferError(
                    message=str(exc),
                    timestamp=_now(),
                    http_status_code=502,
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="messages.infer.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"model": body.model, "message": err.message},
                    )
                )
                raise err from exc
            except Exception as exc:
                err = MessagesInferError(
                    message=f"Unexpected inference error: {exc}",
                    timestamp=_now(),
                    http_status_code=500,
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="messages.infer.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"model": body.model, "message": err.message},
                    )
                )
                raise err from exc

        return JSONResponse(content=response, status_code=200)

    return router
