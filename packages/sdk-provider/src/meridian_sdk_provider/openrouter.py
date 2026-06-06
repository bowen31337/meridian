"""OpenRouterProvider — gateway adapter for the OpenRouter API.

Routes requests to many hosted models through a single OpenRouter endpoint
using the OpenAI-compatible chat completions API with SSE streaming.
Honors per-model feature flags: cache_control is passed through for models
that support it (e.g. Anthropic via OpenRouter), vision content blocks are
included when the model advertises image input, and tool definitions are
forwarded for models that support function calling.

Emits an "openrouter.model.call" OTel span with a provider.invocation event on
every call(). On failure the span is marked ERROR, the audit log receives an
"openrouter.call.failed" entry, and the exception is surfaced to the caller as a
ProviderCallError subclass.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx

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

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_CONTEXT_WINDOW = 128000
_REQUEST_TIMEOUT = 120.0
_LIST_TIMEOUT = 10.0


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
    has_cache_control = False

    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            block_dict: dict[str, Any] = {"type": "text", "text": block.text}
            if getattr(block, "cache_control", None) is not None:
                block_dict["cache_control"] = {"type": "ephemeral"}
                has_cache_control = True
            text_parts.append(block_dict)
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

    if has_cache_control:
        return {"role": msg.role, "content": text_parts}

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


def _http_to_provider_error(exc: httpx.HTTPStatusError, provider_name: str) -> ProviderCallError:
    status = exc.response.status_code
    msg = f"OpenRouter HTTP {status}: {exc.response.text[:200]}"
    if status == 429:
        return ProviderRateLimitError(msg, provider_name=provider_name, status_code=status)
    if status >= 500:
        return ProviderServerError(msg, provider_name=provider_name, status_code=status)
    return ProviderCallError(msg, provider_name=provider_name, status_code=status)


class OpenRouterProvider:
    """ModelProvider that routes to many models via the OpenRouter gateway.

    Parameters
    ----------
    api_key:
        OpenRouter API key. Required.
    base_url:
        OpenRouter API base URL. Defaults to ``https://openrouter.ai/api/v1``.
    name:
        Provider instance identifier surfaced in OTel attributes and audit entries.
    audit_log:
        Audit log sink. Defaults to ``NoopAuditLog``.
    timeout:
        Per-request timeout in seconds for streaming calls.
    http_referer:
        ``HTTP-Referer`` header value forwarded to OpenRouter for attribution.
    app_title:
        ``X-Title`` header value forwarded to OpenRouter for attribution.
    """

    kind: str = "openrouter"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        name: str = "openrouter",
        audit_log: AuditLog | None = None,
        timeout: float = _REQUEST_TIMEOUT,
        http_referer: str | None = None,
        app_title: str | None = None,
        _http: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.capabilities = ProviderCapabilities(streaming=True, cache_control=True)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()
        self._timeout = timeout
        self._http = _http or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "HTTP-Referer": http_referer or "https://meridian.ai",
                "X-Title": app_title or "Meridian",
            },
        )

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "openrouter.model.call",
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
                stream = self._do_stream(opts)
                while True:
                    try:
                        event = await stream.__anext__()
                    except StopAsyncIteration:
                        break
                    yield event
            except ProviderCallError as exc:
                record_provider_failure(span, exc, provider_name=self.name, model=opts.model)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="openrouter.call.failed",
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
                        event="openrouter.call.failed",
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
        payload: dict[str, Any] = {
            "model": opts.model,
            "messages": _convert_messages(opts.messages, opts.system),
            "max_tokens": opts.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if opts.tools:
            payload["tools"] = _convert_tools(opts.tools)
            payload["tool_choice"] = "auto"
        if opts.temperature is not None:
            payload["temperature"] = opts.temperature

        try:
            async with self._http.stream("POST", "/chat/completions", json=payload) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise _http_to_provider_error(exc, self.name) from exc

                first = True
                tool_id_by_index: dict[int, str] = {}
                input_tokens: int | None = None
                output_tokens: int | None = None
                stop_reason: str | None = None

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)

                    if first:
                        yield MessageStartEvent(
                            type="message_start",
                            model=chunk.get("model", opts.model),
                            provider=self.name,
                        )
                        first = False

                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}

                        text = delta.get("content")
                        if text:
                            yield TextDeltaEvent(type="text_delta", text=text)

                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            tc_id = tc.get("id")
                            func = tc.get("function") or {}
                            name = func.get("name")
                            args_delta = func.get("arguments") or ""

                            if tc_id and idx not in tool_id_by_index:
                                tool_id_by_index[idx] = tc_id
                                yield ToolUseStartEvent(
                                    type="tool_use_start",
                                    id=tc_id,
                                    name=name or "",
                                )

                            if args_delta and idx in tool_id_by_index:
                                yield ToolInputDeltaEvent(
                                    type="tool_input_delta",
                                    id=tool_id_by_index[idx],
                                    partial_json=args_delta,
                                )

                        finish = choice.get("finish_reason")
                        if finish:
                            stop_reason = finish

                    usage = chunk.get("usage")
                    if usage:
                        input_tokens = usage.get("prompt_tokens")
                        output_tokens = usage.get("completion_tokens")

                yield MessageStopEvent(
                    type="message_stop",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    stop_reason=stop_reason,
                )
        except ProviderCallError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider_name=self.name) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderCallError(
                f"OpenRouter connection error: {exc}", provider_name=self.name
            ) from exc

    def list_models(self) -> list[ModelEntry]:
        """Discover available models from OpenRouter via GET /models."""
        try:
            response = httpx.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=_LIST_TIMEOUT,
            )
            if not response.is_success:
                _LOG.warning(
                    "OpenRouterProvider(%s).list_models failed: HTTP %s",
                    self.name,
                    response.status_code,
                )
                return []
        except Exception as exc:
            _LOG.warning("OpenRouterProvider(%s).list_models failed: %s", self.name, exc)
            return []

        entries: list[ModelEntry] = []
        for model in response.json().get("data", []):
            model_id: str = model.get("id", "")
            if not model_id:
                continue
            arch = model.get("architecture") or {}
            input_modalities: list[str] = arch.get("input_modalities") or []
            context_window = model.get("context_length") or _DEFAULT_CONTEXT_WINDOW
            vision = "image" in input_modalities
            cache = model_id.startswith("anthropic/")
            thinking = "thinking" in model_id
            entries.append(
                ModelEntry(
                    provider=self.name,
                    model=model_id,
                    context_window=context_window,
                    capabilities=ModelCapabilities(
                        streaming=True,
                        thinking=thinking,
                        vision=vision,
                        tools=True,
                        cache=cache,
                    ),
                )
            )
        return entries

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        raise NotImplementedError("OpenRouterProvider does not support token counting")

    async def close(self) -> None:
        await self._http.aclose()
