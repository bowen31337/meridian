from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from ._execution import execute_tool
from ._types import Capability, InProcessHandler, ToolContext, ToolDefinition, ToolResult


class MeridianTool:
    """An in-process Python tool produced by the @meridian_tool decorator.

    Wraps the handler with the full SDK pipeline (input/output validation,
    OTel span, structured logging, audit log on failure).

    Usage::

        @meridian_tool(
            description="Count words in a string",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            output_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
            capabilities=["fs.read[/workspace/**]"],
        )
        async def word_count(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            return {"count": len(args["text"].split())}

    Execute via::

        result = await word_count.execute({"text": "hello world"}, ctx)
    """

    def __init__(
        self,
        definition: ToolDefinition,
        fn: Callable[..., Awaitable[Any]],
        audit_log_path: str | None = None,
    ) -> None:
        self.definition = definition
        self._fn = fn
        self._audit_log_path = audit_log_path

    async def execute(self, args: Any, ctx: ToolContext) -> ToolResult:
        return await execute_tool(
            self.definition,
            args,
            ctx,
            self._fn,
            audit_log_path=self._audit_log_path,
        )

    # Forward calls so the wrapped function can still be called directly
    # (useful in tests that bypass the SDK pipeline).
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<MeridianTool name={self.definition.name!r}>"


@overload
def meridian_tool(fn: Callable[..., Awaitable[Any]]) -> MeridianTool: ...


@overload
def meridian_tool(
    *,
    name: str | None = None,
    description: str = "",
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    capabilities: list[Capability] | None = None,
    required_environment: str | None = None,
    timeout_ms: int = 30_000,
    memory_cap_mb: int | None = None,
    audit_log_path: str | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], MeridianTool]: ...


def meridian_tool(  # type: ignore[misc]
    fn: Callable[..., Awaitable[Any]] | None = None,
    *,
    name: str | None = None,
    description: str = "",
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    capabilities: list[Capability] | None = None,
    required_environment: str | None = None,
    timeout_ms: int = 30_000,
    memory_cap_mb: int | None = None,
    audit_log_path: str | None = None,
) -> MeridianTool | Callable[[Callable[..., Awaitable[Any]]], MeridianTool]:
    """Decorator that registers an async Python function as a Meridian tool.

    Can be used bare (``@meridian_tool``) or with arguments
    (``@meridian_tool(description="...", capabilities=[...])``)::

        # Bare usage — name inferred from function, schema inferred from
        # Pydantic model annotation on the first parameter.
        @meridian_tool
        async def my_tool(args: MyArgsModel, ctx: ToolContext) -> dict[str, Any]:
            ...

        # With arguments
        @meridian_tool(
            name="my_tool",
            description="Does X",
            input_schema={"type": "object", ...},
            capabilities=["fs.read[/workspace/**]"],
        )
        async def my_tool(args: dict[str, Any], ctx: ToolContext) -> Any:
            ...
    """

    def _make(f: Callable[..., Awaitable[Any]]) -> MeridianTool:
        tool_name = name or f.__name__
        resolved_schema = input_schema if input_schema is not None else _infer_input_schema(f)
        resolved_description = description or (inspect.getdoc(f) or "")
        pydantic_type = _get_pydantic_arg_type(f)

        # Wrap the user function to coerce a raw-dict args payload into the
        # Pydantic model declared as the first parameter, if any.
        async def _wrapped(args: Any, ctx: ToolContext) -> Any:
            coerced = (
                pydantic_type.model_validate(args)
                if (pydantic_type and isinstance(args, dict))
                else args
            )
            return await f(coerced, ctx)

        definition = ToolDefinition(
            name=tool_name,
            description=resolved_description,
            input_schema=resolved_schema if resolved_schema is not None else {},
            output_schema=output_schema,
            capabilities=capabilities or [],
            required_environment=required_environment,
            timeout_ms=timeout_ms,
            memory_cap_mb=memory_cap_mb,
            handler=InProcessHandler(module=f.__module__),
        )
        return MeridianTool(definition, _wrapped, audit_log_path)

    if fn is not None:
        # Bare usage: @meridian_tool
        return _make(fn)

    # Called with arguments: @meridian_tool(...)
    return _make


# ---------------------------------------------------------------------------
# Schema inference helper
# ---------------------------------------------------------------------------


def _get_pydantic_arg_type(fn: Callable[..., Any]) -> type[BaseModel] | None:
    """Return the Pydantic model class of the first parameter, or None."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        return None
    annotation = params[0].annotation
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        import sys

        ns = getattr(sys.modules.get(fn.__module__), "__dict__", {})
        try:
            annotation = eval(annotation, ns)  # noqa: S307
        except Exception:
            return None
    try:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
    except Exception:
        pass
    return None


def _infer_input_schema(fn: Callable[..., Any]) -> dict[str, Any] | None:
    """Try to derive an input JSON Schema from the function's first parameter.

    If the first parameter is annotated with a Pydantic BaseModel subclass,
    we generate the schema from it.  Otherwise we return None and let the
    caller fall back to an empty schema ``{}``.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        return None

    annotation = params[0].annotation
    if annotation is inspect.Parameter.empty:
        return None

    # Resolve string annotations (PEP 563 / from __future__ import annotations)
    if isinstance(annotation, str):
        import sys

        ns = getattr(sys.modules.get(fn.__module__), "__dict__", {})
        try:
            annotation = eval(annotation, ns)  # noqa: S307
        except Exception:
            return None

    try:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.model_json_schema()
    except Exception:
        pass

    return None
