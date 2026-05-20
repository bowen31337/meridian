"""AnthropicApiKeyProvider — model adapter using the raw anthropic Python SDK.

The API key is resolved from Vault by the host application and passed to the
constructor; the provider itself never touches Vault (§13.5 isolation contract).
Connects directly to api.anthropic.com; supports streaming, tool-use, vision,
extended thinking, and prompt caching (cache_control pass-through).

Emits an "anthropic.model.call" OTel span with a provider.invocation event on
every call(). On failure the span is marked ERROR, the audit log receives an
"anthropic.call.failed" entry, and the exception is surfaced to the caller as a
ProviderCallError subclass.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from meridian_sdk_provider.audit import AuditLog, AuditLogEntry, NoopAuditLog
from meridian_sdk_provider.errors import (
    ProviderCallError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from meridian_sdk_provider.protocol import ModelCapabilities, ModelEntry, ProviderCapabilities
from meridian_sdk_provider.telemetry import get_tracer, record_cache_metrics, record_invocation_event, record_provider_failure
from meridian_sdk_provider.types import (
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    ModelCountReq,
    ModelEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    TokenCount,
    ToolDefinition,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)

_LOG = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_REQUEST_TIMEOUT = 120.0

# Anthropic does not expose a list-models endpoint; known models are enumerated here.
_KNOWN_MODELS: list[tuple[str, int, bool, bool]] = [
    # (model_id, context_window, thinking, vision)
    ("claude-opus-4-7", 200_000, True, True),
    ("claude-sonnet-4-6", 200_000, True, True),
    ("claude-haiku-4-5-20251001", 200_000, False, True),
    ("claude-3-7-sonnet-20250219", 200_000, True, True),
    ("claude-3-5-sonnet-20241022", 200_000, False, True),
    ("claude-3-5-haiku-20241022", 200_000, False, True),
    ("claude-3-opus-20240229", 200_000, False, True),
    ("claude-3-haiku-20240307", 200_000, False, True),
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _convert_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [_convert_message(msg) for msg in messages]


def _convert_message(msg: Any) -> dict[str, Any]:
    content = msg.content
    if isinstance(content, str):
        return {"role": msg.role, "content": content}

    blocks: list[dict[str, Any]] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            b: dict[str, Any] = {"type": "text", "text": block.text}
            if getattr(block, "cache_control", None) is not None:
                b["cache_control"] = {"type": "ephemeral"}
            blocks.append(b)
        elif btype == "tool_use":
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        elif btype == "tool_result":
            raw = block.content
            b = {"type": "tool_result", "tool_use_id": block.tool_use_id}
            if isinstance(raw, str):
                b["content"] = raw
            else:
                b["content"] = [{"type": "text", "text": str(c)} for c in raw]
            blocks.append(b)
        elif btype == "thinking":
            blocks.append({
                "type": "thinking",
                "thinking": block.thinking,
                "signature": block.signature,
            })

    return {"role": msg.role, "content": blocks}


def _convert_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in tools
    ]


def _api_status_to_provider_error(
    exc: anthropic.APIStatusError, provider_name: str
) -> ProviderCallError:
    status = exc.status_code
    msg = f"Anthropic HTTP {status}: {str(exc)[:200]}"
    if status == 429:
        return ProviderRateLimitError(msg, provider_name=provider_name, status_code=status)
    if status >= 500:
        return ProviderServerError(msg, provider_name=provider_name, status_code=status)
    return ProviderCallError(msg, provider_name=provider_name, status_code=status)


class AnthropicApiKeyProvider:
    """ModelProvider adapter using the raw anthropic Python SDK with API key auth.

    Parameters
    ----------
    api_key:
        Anthropic API key (sk-ant-...). Resolved from Vault by the host before
        construction; never stored in config files in plaintext.
    base_url:
        Anthropic API base URL. Defaults to ``https://api.anthropic.com``.
    name:
        Provider instance identifier surfaced in OTel attributes and audit entries.
    audit_log:
        Audit log sink. Defaults to ``NoopAuditLog``.
    timeout:
        Per-request timeout in seconds for streaming calls.
    """

    kind: str = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        name: str = "anthropic",
        audit_log: AuditLog | None = None,
        timeout: float = _REQUEST_TIMEOUT,
        _client: AsyncAnthropic | None = None,
    ) -> None:
        self.name = name
        self.capabilities = ProviderCapabilities(
            streaming=True,
            thinking=True,
            cache_control=True,
            count_tokens=True,
        )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()
        self._timeout = timeout
        self._client = _client or AsyncAnthropic(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "anthropic.model.call",
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
                async for event in self._do_stream(opts, span):
                    yield event
            except ProviderCallError as exc:
                record_provider_failure(span, exc, provider_name=self.name, model=opts.model)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="anthropic.call.failed",
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
                        event="anthropic.call.failed",
                        provider_name=self.name,
                        provider_kind=self.kind,
                        model=opts.model,
                        session_id=opts.session_id,
                        timestamp=_now(),
                        detail={"error": str(exc), "error_type": type(exc).__name__},
                    )
                )
                raise err from exc

    async def _do_stream(self, opts: ModelCallOpts, span: Any) -> AsyncIterator[ModelEvent]:
        messages = _convert_messages(opts.messages)

        kwargs: dict[str, Any] = {
            "model": opts.model,
            "messages": messages,
            "max_tokens": opts.max_tokens,
            "stream": True,
        }
        if opts.system:
            # Per-call prompt-cache header injection: wrap system prompt with ephemeral
            # cache_control so repeated calls share the cached prefix.
            kwargs["system"] = [
                {"type": "text", "text": opts.system, "cache_control": {"type": "ephemeral"}}
            ]
        if opts.tools:
            kwargs["tools"] = _convert_tools(opts.tools)
            kwargs["tool_choice"] = {"type": "auto"}
        if opts.temperature is not None:
            kwargs["temperature"] = opts.temperature
        if opts.enable_thinking and opts.thinking_budget_tokens:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": opts.thinking_budget_tokens,
            }

        try:
            stream = await self._client.messages.create(**kwargs)

            tool_id_by_index: dict[int, str] = {}
            input_tokens: int | None = None
            output_tokens: int | None = None
            stop_reason: str | None = None
            cache_creation_input_tokens: int = 0
            cache_read_input_tokens: int = 0

            async for raw in stream:
                etype = raw.type

                if etype == "message_start":
                    usage = raw.message.usage
                    input_tokens = usage.input_tokens
                    cache_creation_input_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
                    cache_read_input_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                    yield MessageStartEvent(
                        type="message_start",
                        model=raw.message.model,
                        provider=self.name,
                        input_tokens=input_tokens,
                    )

                elif etype == "content_block_start":
                    block = raw.content_block
                    if block.type == "tool_use":
                        tool_id_by_index[raw.index] = block.id
                        yield ToolUseStartEvent(
                            type="tool_use_start",
                            id=block.id,
                            name=block.name,
                        )

                elif etype == "content_block_delta":
                    delta = raw.delta
                    dtype = delta.type
                    if dtype == "text_delta":
                        yield TextDeltaEvent(type="text_delta", text=delta.text)
                    elif dtype == "input_json_delta":
                        tool_id = tool_id_by_index.get(raw.index)
                        if tool_id:
                            yield ToolInputDeltaEvent(
                                type="tool_input_delta",
                                id=tool_id,
                                partial_json=delta.partial_json,
                            )
                    elif dtype == "thinking_delta":
                        yield ThinkingDeltaEvent(
                            type="thinking_delta",
                            thinking=delta.thinking,
                        )

                elif etype == "message_delta":
                    stop_reason = raw.delta.stop_reason
                    output_tokens = raw.usage.output_tokens

            yield MessageStopEvent(
                type="message_stop",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason=stop_reason,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            )
            record_cache_metrics(
                span,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            )

        except ProviderCallError:
            raise
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc), provider_name=self.name) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderCallError(
                f"Anthropic connection error: {exc}", provider_name=self.name
            ) from exc
        except anthropic.APIStatusError as exc:
            raise _api_status_to_provider_error(exc, self.name) from exc

    def list_models(self) -> list[ModelEntry]:
        """Return known Anthropic models.

        Anthropic does not provide a list-models endpoint; the set is enumerated
        from the known model catalogue.
        """
        return [
            ModelEntry(
                provider=self.name,
                model=model_id,
                context_window=ctx,
                capabilities=ModelCapabilities(
                    streaming=True,
                    thinking=thinking,
                    vision=vision,
                    tools=True,
                    cache=True,
                ),
            )
            for model_id, ctx, thinking, vision in _KNOWN_MODELS
        ]

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        """Count tokens for a request without executing the model call."""
        messages = _convert_messages(req.messages)
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
        }
        if req.system:
            kwargs["system"] = req.system
        if req.tools:
            kwargs["tools"] = _convert_tools(req.tools)

        try:
            response = await self._client.messages.count_tokens(**kwargs)
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc), provider_name=self.name) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderCallError(
                f"Anthropic connection error: {exc}", provider_name=self.name
            ) from exc
        except anthropic.APIStatusError as exc:
            raise _api_status_to_provider_error(exc, self.name) from exc

        return TokenCount(input_tokens=response.input_tokens)

    async def close(self) -> None:
        await self._client.close()
