"""Unit tests for the @meridian_tool decorator (in-process Python tools)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from meridian_sdk_tool import (
    MeridianTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
    meridian_tool,
)

_CTX = ToolContext(workspace="/workspace", session_id="sess_test")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class AddArgs(BaseModel):
    x: int
    y: int


@meridian_tool(
    description="Add two integers",
    capabilities=["fs.read[/workspace/**]"],
    timeout_ms=5_000,
)
async def add_tool(args: AddArgs, ctx: ToolContext) -> dict[str, Any]:
    return {"sum": args.x + args.y}


@meridian_tool(
    name="explicit_name",
    description="Explicit name tool",
    input_schema={"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]},
    output_schema={
        "type": "object",
        "properties": {"doubled": {"type": "integer"}},
        "required": ["doubled"],
    },
)
async def some_fn(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return {"doubled": args["v"] * 2}


# ---------------------------------------------------------------------------
# Tests — decorator metadata
# ---------------------------------------------------------------------------


def test_bare_decorator_returns_meridian_tool() -> None:
    assert isinstance(add_tool, MeridianTool)


def test_name_inferred_from_function() -> None:
    assert add_tool.definition.name == "add_tool"


def test_explicit_name_respected() -> None:
    assert some_fn.definition.name == "explicit_name"


def test_description_stored() -> None:
    assert "Add two integers" in add_tool.definition.description


def test_capabilities_stored() -> None:
    assert "fs.read[/workspace/**]" in add_tool.definition.capabilities


def test_timeout_stored() -> None:
    assert add_tool.definition.timeout_ms == 5_000


def test_pydantic_schema_inferred() -> None:
    schema = add_tool.definition.input_schema
    assert schema.get("type") == "object" or "properties" in schema
    props = schema.get("properties", {})
    assert "x" in props
    assert "y" in props


def test_explicit_schemas_stored() -> None:
    assert some_fn.definition.input_schema["properties"]["v"]["type"] == "integer"
    assert some_fn.definition.output_schema is not None


def test_definition_is_tool_definition() -> None:
    assert isinstance(add_tool.definition, ToolDefinition)


# ---------------------------------------------------------------------------
# Tests — execution
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_successful_execution() -> None:
    result = await add_tool.execute({"x": 3, "y": 4}, _CTX)
    assert not result.is_error
    assert result.result == {"sum": 7}


@pytest.mark.anyio
async def test_input_validation_failure_returns_is_error() -> None:
    # "x" should be int, "abc" is a string — validation fails
    result = await add_tool.execute({"x": "abc", "y": 4}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "validation" in result.error.code


@pytest.mark.anyio
async def test_missing_required_field_returns_is_error() -> None:
    result = await add_tool.execute({"x": 1}, _CTX)
    assert result.is_error


@pytest.mark.anyio
async def test_output_schema_validation_triggers_is_error() -> None:
    @meridian_tool(
        input_schema={"type": "object"},
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    )
    async def bad_output(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        return {"wrong_key": True}  # missing "ok"

    result = await bad_output.execute({}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "output" in result.error.code


@pytest.mark.anyio
async def test_handler_exception_returns_is_error() -> None:
    @meridian_tool(input_schema={"type": "object"})
    async def boom(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        raise ValueError("something went wrong")

    result = await boom.execute({}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert result.error.code == "execution_failed"
    assert "something went wrong" in result.error.message


@pytest.mark.anyio
async def test_result_ok_factory() -> None:
    result = ToolResult.ok({"a": 1})
    assert not result.is_error
    assert result.result == {"a": 1}


@pytest.mark.anyio
async def test_result_err_factory() -> None:
    result = ToolResult.err("my_code", "msg", extra="data")
    assert result.is_error
    assert result.error is not None
    assert result.error.code == "my_code"
    assert result.error.details["extra"] == "data"


# ---------------------------------------------------------------------------
# Tests — bare decorator usage
# ---------------------------------------------------------------------------


def test_bare_decorator_usage() -> None:
    @meridian_tool
    async def no_args_tool(args: dict[str, Any], ctx: ToolContext) -> None:
        pass

    assert isinstance(no_args_tool, MeridianTool)
    assert no_args_tool.definition.name == "no_args_tool"
