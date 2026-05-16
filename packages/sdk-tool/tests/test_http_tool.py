"""Tests for http_tool helper."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meridian_sdk_tool import ToolContext, http_tool
from meridian_sdk_tool.http_tool import HttpTool

_CTX = ToolContext(workspace="/workspace", session_id="sess_http")

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"results": {"type": "array"}},
    "required": ["results"],
}


def test_http_tool_returns_http_tool_instance() -> None:
    tool = http_tool(
        name="search",
        description="Web search",
        url="http://localhost:8080/search",
        input_schema=_INPUT_SCHEMA,
    )
    assert isinstance(tool, HttpTool)
    assert tool.definition.name == "search"


def test_http_tool_definition_stores_url() -> None:
    tool = http_tool(
        name="t",
        description="d",
        url="http://example.com/tool",
        input_schema={},
    )
    from meridian_sdk_tool._types import HttpHandler

    assert isinstance(tool.definition.handler, HttpHandler)
    assert tool.definition.handler.url == "http://example.com/tool"


def test_http_tool_auth_stored() -> None:
    from meridian_sdk_tool._types import HttpHandler

    tool = http_tool(
        name="t",
        description="d",
        url="http://example.com",
        input_schema={},
        auth={"bearer": "tok-123"},
    )
    assert isinstance(tool.definition.handler, HttpHandler)
    assert tool.definition.handler.auth == {"bearer": "tok-123"}


def test_http_tool_capabilities_stored() -> None:
    tool = http_tool(
        name="t",
        description="d",
        url="http://example.com",
        input_schema={},
        capabilities=["net.fetch[example.com]"],
    )
    assert "net.fetch[example.com]" in tool.definition.capabilities


@pytest.mark.anyio
async def test_http_tool_successful_call() -> None:
    tool = http_tool(
        name="search",
        description="d",
        url="http://localhost:8080/search",
        input_schema=_INPUT_SCHEMA,
        output_schema=_OUTPUT_SCHEMA,
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"results": ["item1", "item2"]}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("meridian_sdk_tool.http_tool.httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = mock_client
        result = await tool.execute({"query": "hello"}, _CTX)

    assert not result.is_error, result.error
    assert result.result == {"results": ["item1", "item2"]}


@pytest.mark.anyio
async def test_http_tool_input_validation_failure() -> None:
    tool = http_tool(
        name="t",
        description="d",
        url="http://example.com",
        input_schema=_INPUT_SCHEMA,
    )
    # "query" must be a string, not an int
    result = await tool.execute({"query": 99}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "validation" in result.error.code


@pytest.mark.anyio
async def test_http_tool_server_error_returned_as_is_error() -> None:
    tool = http_tool(
        name="t",
        description="d",
        url="http://example.com",
        input_schema={"type": "object"},
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {"error": {"code": "not_found", "message": "resource not found"}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("meridian_sdk_tool.http_tool.httpx") as mock_httpx:
        mock_httpx.AsyncClient.return_value = mock_client
        result = await tool.execute({}, _CTX)

    assert result.is_error
    assert result.error is not None
    assert "not_found" in result.error.message or "resource not found" in result.error.message
