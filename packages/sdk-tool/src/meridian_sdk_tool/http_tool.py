"""HTTP tool helper.

Wraps a remote HTTP endpoint as a Meridian tool.  The endpoint must accept
a POST request with the JSON body::

    {"args": ..., "context": {"workspace": ..., "session_id": ..., ...}}

and respond with::

    {"result": ...}       # on success
    {"error": {"code": "...", "message": "..."}}  # on failure

``httpx`` is an optional dependency; install with
``pip install meridian-sdk-tool[http]`` or ``pip install httpx``.
"""

from __future__ import annotations

from typing import Any

from ._execution import execute_tool
from ._types import Capability, HttpHandler, ToolContext, ToolDefinition, ToolResult

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False


async def _call_http(
    url: str,
    auth: dict[str, Any] | None,
    args: Any,
    ctx: ToolContext,
    timeout_ms: int,
) -> Any:
    if not _HTTPX_AVAILABLE:
        raise RuntimeError(
            "httpx is required for HTTP tools. Install with: pip install 'meridian-sdk-tool[http]'"
        )

    payload = {
        "args": args,
        "context": {
            "workspace": ctx.workspace,
            "session_id": ctx.session_id,
            "thread_id": ctx.thread_id,
        },
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if auth:
        if "bearer" in auth:
            headers["Authorization"] = f"Bearer {auth['bearer']}"
        elif "api_key" in auth:
            headers["X-Api-Key"] = str(auth["api_key"])

    timeout_s = timeout_ms / 1000.0

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    if "error" in data:
        err = data["error"]
        code = err.get("code", "http_tool_error")
        message = err.get("message", "unknown error from HTTP tool")
        raise RuntimeError(f"{code}: {message}")

    return data.get("result")


class HttpTool:
    """A Meridian tool that delegates to an HTTP endpoint."""

    def __init__(
        self,
        definition: ToolDefinition,
        audit_log_path: str | None = None,
    ) -> None:
        self.definition = definition
        self._audit_log_path = audit_log_path

    async def execute(self, args: Any, ctx: ToolContext) -> ToolResult:
        handler_def = self.definition.handler
        assert isinstance(handler_def, HttpHandler)

        async def _handler(a: Any, c: ToolContext) -> Any:
            return await _call_http(
                handler_def.url, handler_def.auth, a, c, self.definition.timeout_ms
            )

        return await execute_tool(
            self.definition,
            args,
            ctx,
            _handler,
            audit_log_path=self._audit_log_path,
        )

    def __repr__(self) -> str:
        return f"<HttpTool name={self.definition.name!r}>"


def http_tool(
    name: str,
    description: str,
    url: str,
    input_schema: dict[str, Any],
    output_schema: dict[str, Any] | None = None,
    auth: dict[str, Any] | None = None,
    capabilities: list[Capability] | None = None,
    required_environment: str | None = None,
    timeout_ms: int = 30_000,
    memory_cap_mb: int | None = None,
    audit_log_path: str | None = None,
) -> HttpTool:
    """Build an :class:`HttpTool` from a URL and a schema declaration.

    *auth* accepts a dict with one of:

    * ``{"bearer": "<token>"}`` — adds ``Authorization: Bearer <token>``
    * ``{"api_key": "<key>"}`` — adds ``X-Api-Key: <key>``

    Requires ``httpx``; install with ``pip install 'meridian-sdk-tool[http]'``.
    """
    definition = ToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        capabilities=capabilities or [],
        required_environment=required_environment,
        timeout_ms=timeout_ms,
        memory_cap_mb=memory_cap_mb,
        handler=HttpHandler(url=url, auth=auth),
    )
    return HttpTool(definition, audit_log_path)
