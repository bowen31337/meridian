"""OllamaProvider — HTTP client for the local Ollama daemon.

Connects to a running Ollama instance (default http://localhost:11434),
discovers models via GET /api/tags, and streams responses via POST /api/chat.
Covers the local/offline inference path.

Emits an "ollama.model.call" OTel span with a provider.invocation event on
every call(). On failure the span is marked ERROR, the audit log receives an
"ollama.call.failed" entry, and the exception is surfaced to the caller as a
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

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_CONTEXT_WINDOW = 131072
_REQUEST_TIMEOUT = 120.0
_LIST_TIMEOUT = 5.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _convert_messages(messages: list[Any], system: str | None) -> list[dict[str, Any]]:
    """Convert Meridian messages + optional system prompt to Ollama /api/chat format."""
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

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_result: str | None = None

    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            tool_calls.append({"function": {"name": block.name, "arguments": block.input}})
        elif btype == "tool_result":
            raw = block.content
            tool_result = raw if isinstance(raw, str) else " ".join(str(c) for c in raw)

    if msg.role == "tool" or (tool_result is not None and not tool_calls):
        return {"role": "tool", "content": tool_result or "".join(text_parts)}

    entry: dict[str, Any] = {"role": msg.role, "content": "".join(text_parts)}
    if tool_calls:
        entry["tool_calls"] = tool_calls
    return entry


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
    msg = f"Ollama HTTP {status}: {exc.response.text[:200]}"
    if status == 429:
        return ProviderRateLimitError(msg, provider_name=provider_name, status_code=status)
    if status >= 500:
        return ProviderServerError(msg, provider_name=provider_name, status_code=status)
    return ProviderCallError(msg, provider_name=provider_name, status_code=status)


class OllamaProvider:
    """ModelProvider that streams from a local Ollama daemon.

    Parameters
    ----------
    base_url:
        Ollama daemon base URL. Defaults to ``http://localhost:11434``.
    name:
        Provider instance identifier surfaced in OTel attributes and audit entries.
    audit_log:
        Audit log sink. Defaults to ``NoopAuditLog``.
    timeout:
        Per-request timeout in seconds for streaming calls.
    """

    kind: str = "ollama"

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        *,
        name: str = "ollama",
        audit_log: AuditLog | None = None,
        timeout: float = _REQUEST_TIMEOUT,
        _http: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.capabilities = ProviderCapabilities(streaming=True)
        self._base_url = base_url.rstrip("/")
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()
        self._timeout = timeout
        self._http = _http or httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "ollama.model.call",
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
                        event="ollama.call.failed",
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
                        event="ollama.call.failed",
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
            "stream": True,
        }
        if opts.tools:
            payload["tools"] = _convert_tools(opts.tools)
        if opts.temperature is not None:
            payload["options"] = {"temperature": opts.temperature}

        try:
            async with self._http.stream("POST", "/api/chat", json=payload) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise _http_to_provider_error(exc, self.name) from exc

                first = True
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)

                    if first:
                        yield MessageStartEvent(
                            type="message_start",
                            model=opts.model,
                            provider=self.name,
                        )
                        first = False

                    message = chunk.get("message", {})
                    content = message.get("content", "")
                    if content:
                        yield TextDeltaEvent(type="text_delta", text=content)

                    for i, tc in enumerate(message.get("tool_calls") or []):
                        func = tc.get("function", {})
                        tool_id = f"call_{i}"
                        yield ToolUseStartEvent(
                            type="tool_use_start",
                            id=tool_id,
                            name=func.get("name", ""),
                        )
                        yield ToolInputDeltaEvent(
                            type="tool_input_delta",
                            id=tool_id,
                            partial_json=json.dumps(func.get("arguments", {})),
                        )

                    if chunk.get("done"):
                        yield MessageStopEvent(
                            type="message_stop",
                            input_tokens=chunk.get("prompt_eval_count"),
                            output_tokens=chunk.get("eval_count"),
                            stop_reason=chunk.get("done_reason", "stop"),
                        )
        except ProviderCallError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider_name=self.name) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderCallError(
                f"Ollama connection error: {exc}", provider_name=self.name
            ) from exc

    def list_models(self) -> list[ModelEntry]:
        """Discover available models from the Ollama daemon via GET /api/tags."""
        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=_LIST_TIMEOUT)
            if not response.is_success:
                _LOG.warning(
                    "OllamaProvider(%s).list_models failed: HTTP %s",
                    self.name,
                    response.status_code,
                )
                return []
        except Exception as exc:
            _LOG.warning("OllamaProvider(%s).list_models failed: %s", self.name, exc)
            return []
        entries: list[ModelEntry] = []
        for model in response.json().get("models", []):
            model_name: str = model.get("name") or model.get("model", "")
            if not model_name:
                continue
            entries.append(
                ModelEntry(
                    provider=self.name,
                    model=model_name,
                    context_window=_DEFAULT_CONTEXT_WINDOW,
                    capabilities=ModelCapabilities(
                        streaming=True,
                        thinking=False,
                        vision=False,
                        tools=True,
                        cache=False,
                    ),
                )
            )
        return entries

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        raise NotImplementedError("OllamaProvider does not support token counting")

    async def close(self) -> None:
        await self._http.aclose()
