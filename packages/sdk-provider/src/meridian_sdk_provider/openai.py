"""OpenAIProvider — model adapter using the openai Python SDK.

The API key is resolved from Vault by the host application and passed to the
constructor; the provider itself never touches Vault (§13.5 isolation contract).
Supports streaming and tool-use calls; normalises responses to ModelEvent shape.

Emits an "openai.model.call" OTel span with a provider.invocation event on
every call(). On failure the span is marked ERROR, the audit log receives an
"openai.call.failed" entry, and the exception is surfaced to the caller as a
ProviderCallError subclass.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from .audit import AuditLog, AuditLogEntry, NoopAuditLog
from .errors import (
    ProviderCallError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from .protocol import ModelCapabilities, ModelEntry, ProviderCapabilities
from .telemetry import get_tracer, record_invocation_event, record_provider_failure
from .types import (
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    ModelCountReq,
    ModelEvent,
    TextDeltaEvent,
    TokenCount,
    ToolDefinition,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)

_LOG = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_REQUEST_TIMEOUT = 120.0
_LIST_TIMEOUT = 10.0
_DEFAULT_CONTEXT_WINDOW = 128000


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _convert_messages(messages: list[Any], system: str | None) -> list[dict[str, Any]]:
    """Convert Meridian messages + optional system prompt to OpenAI chat format."""
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})
    for msg in messages:
        result.append(_convert_message(msg))
    return result


def _convert_message(msg: Any) -> dict[str, Any]:
    content = msg.content
    if isinstance(content, str):
        return {"role": msg.role, "content": content}

    text_parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_result: tuple[str, str] | None = None

    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append({"type": "text", "text": block.text})
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
            )
        elif btype == "tool_result":
            raw = block.content
            tool_result = (
                block.tool_use_id,
                raw if isinstance(raw, str) else " ".join(str(c) for c in raw),
            )

    if tool_result is not None and not tool_calls:
        return {"role": "tool", "tool_call_id": tool_result[0], "content": tool_result[1]}

    if tool_calls:
        text_content: str | None = "".join(p["text"] for p in text_parts) if text_parts else None
        return {"role": "assistant", "content": text_content, "tool_calls": tool_calls}

    return {"role": msg.role, "content": "".join(p["text"] for p in text_parts)}


def _convert_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _api_status_to_provider_error(
    exc: openai.APIStatusError, provider_name: str
) -> ProviderCallError:
    status = exc.status_code
    msg = f"OpenAI HTTP {status}: {str(exc)[:200]}"
    if status == 429:
        return ProviderRateLimitError(msg, provider_name=provider_name, status_code=status)
    if status >= 500:
        return ProviderServerError(msg, provider_name=provider_name, status_code=status)
    return ProviderCallError(msg, provider_name=provider_name, status_code=status)


class OpenAIProvider:
    """ModelProvider adapter using the openai Python SDK.

    Parameters
    ----------
    api_key:
        OpenAI API key. Resolved from Vault by the host before construction.
    base_url:
        OpenAI API base URL. Defaults to ``https://api.openai.com/v1``.
    name:
        Provider instance identifier surfaced in OTel attributes and audit entries.
    audit_log:
        Audit log sink. Defaults to ``NoopAuditLog``.
    timeout:
        Per-request timeout in seconds for streaming calls.
    """

    kind: str = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        name: str = "openai",
        audit_log: AuditLog | None = None,
        timeout: float = _REQUEST_TIMEOUT,
        _client: AsyncOpenAI | None = None,
    ) -> None:
        self.name = name
        self.capabilities = ProviderCapabilities(streaming=True)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()
        self._timeout = timeout
        self._client = _client or AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "openai.model.call",
            attributes={"provider.name": self.name, "model": opts.model},
        ) as span:
            record_invocation_event(
                span,
                provider_name=self.name,
                provider_kind=self.kind,
                model=opts.model,
                session_id=opts.session_id,
                routing_rule=None,
            )
            try:
                async for event in self._do_stream(opts):
                    yield event
            except ProviderCallError as exc:
                record_provider_failure(span, exc, provider_name=self.name, model=opts.model)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="openai.call.failed",
                        provider_name=self.name,
                        provider_kind=self.kind,
                        model=opts.model,
                        session_id=opts.session_id,
                        timestamp=_now(),
                        detail={"error": str(exc), "error_type": type(exc).__name__},
                    )
                )
                raise
            except Exception as exc:
                err = ProviderCallError(str(exc), provider_name=self.name)
                record_provider_failure(span, err, provider_name=self.name, model=opts.model)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="openai.call.failed",
                        provider_name=self.name,
                        provider_kind=self.kind,
                        model=opts.model,
                        session_id=opts.session_id,
                        timestamp=_now(),
                        detail={"error": str(exc), "error_type": type(exc).__name__},
                    )
                )
                raise err from exc

    async def _do_stream(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        kwargs: dict[str, Any] = {
            "model": opts.model,
            "messages": _convert_messages(opts.messages, opts.system),
            "max_tokens": opts.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if opts.tools:
            kwargs["tools"] = _convert_tools(opts.tools)
            kwargs["tool_choice"] = "auto"
        if opts.temperature is not None:
            kwargs["temperature"] = opts.temperature

        try:
            stream = await self._client.chat.completions.create(**kwargs)

            first = True
            tool_id_by_index: dict[int, str] = {}
            input_tokens: int | None = None
            output_tokens: int | None = None
            stop_reason: str | None = None

            async for chunk in stream:
                if first:
                    yield MessageStartEvent(
                        type="message_start",
                        model=chunk.model or opts.model,
                        provider=self.name,
                    )
                    first = False

                for choice in chunk.choices or []:
                    delta = choice.delta

                    text = delta.content
                    if text:
                        yield TextDeltaEvent(type="text_delta", text=text)

                    for tc in delta.tool_calls or []:
                        idx = tc.index if tc.index is not None else 0
                        tc_id = tc.id
                        func = tc.function
                        tc_name = func.name if func else None
                        args_delta = func.arguments if func else ""

                        if tc_id and idx not in tool_id_by_index:
                            tool_id_by_index[idx] = tc_id
                            yield ToolUseStartEvent(
                                type="tool_use_start",
                                id=tc_id,
                                name=tc_name or "",
                            )

                        if args_delta and idx in tool_id_by_index:
                            yield ToolInputDeltaEvent(
                                type="tool_input_delta",
                                id=tool_id_by_index[idx],
                                partial_json=args_delta,
                            )

                    finish = choice.finish_reason
                    if finish:
                        stop_reason = finish

                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

            yield MessageStopEvent(
                type="message_stop",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason=stop_reason,
            )
        except ProviderCallError:
            raise
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc), provider_name=self.name) from exc
        except openai.APIConnectionError as exc:
            raise ProviderCallError(
                f"OpenAI connection error: {exc}", provider_name=self.name
            ) from exc
        except openai.APIStatusError as exc:
            raise _api_status_to_provider_error(exc, self.name) from exc

    def list_models(self) -> list[ModelEntry]:
        """Discover available models via GET /models."""
        try:
            response = httpx.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=_LIST_TIMEOUT,
            )
            if not response.is_success:
                _LOG.warning(
                    "OpenAIProvider(%s).list_models failed: HTTP %s",
                    self.name,
                    response.status_code,
                )
                return []
        except Exception as exc:
            _LOG.warning("OpenAIProvider(%s).list_models failed: %s", self.name, exc)
            return []

        entries: list[ModelEntry] = []
        for model in response.json().get("data", []):
            model_id: str = model.get("id", "")
            if not model_id:
                continue
            vision = any(x in model_id for x in ("gpt-4o", "gpt-4-turbo", "vision"))
            thinking = any(x in model_id for x in ("o1", "o3"))
            entries.append(
                ModelEntry(
                    provider=self.name,
                    model=model_id,
                    context_window=_DEFAULT_CONTEXT_WINDOW,
                    capabilities=ModelCapabilities(
                        streaming=True,
                        thinking=thinking,
                        vision=vision,
                        tools=True,
                        cache=False,
                    ),
                )
            )
        return entries

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        raise NotImplementedError("OpenAIProvider does not support token counting")

    async def close(self) -> None:
        await self._client.close()
