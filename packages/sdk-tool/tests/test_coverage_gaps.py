"""Unit coverage for sdk-tool leaf branches: decorator schema-inference fall-throughs,
the execution timeout path, OTel-absent and span-error branches, HTTP auth headers /
no-httpx guard, the MCP builder, and subprocess timeout/grace-kill + repr paths."""

from __future__ import annotations

import asyncio
import importlib
import sys
import textwrap
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from meridian_sdk_tool import (
    InProcessHandler,
    MeridianTool,
    SubprocessTool,
    ToolContext,
    ToolDefinition,
    _decorator,
    _otel,
    http_tool,
    mcp_tool,
    subprocess_tool,
)
from meridian_sdk_tool._decorator import _get_pydantic_arg_type, _infer_input_schema
from meridian_sdk_tool._execution import execute_tool
from meridian_sdk_tool._otel import (
    record_tool_call_error,
    record_tool_call_result,
    tool_span,
)
from meridian_sdk_tool.http_tool import _call_http

http_mod = importlib.import_module("meridian_sdk_tool.http_tool")
subprocess_mod = importlib.import_module("meridian_sdk_tool.subprocess_tool")

_CTX = ToolContext(workspace="/workspace", session_id="sess_gap")


# ---------------------------------------------------------------------------
# _decorator: MeridianTool.__call__ / __repr__
# ---------------------------------------------------------------------------


async def test_meridian_tool_call_forwards_to_fn() -> None:
    async def _fn(*args: Any, **kwargs: Any) -> str:
        return "called"

    tool = MeridianTool(
        ToolDefinition(name="t", description="d", input_schema={}, handler=InProcessHandler()),
        _fn,
    )
    assert await tool("a", k=1) == "called"
    assert repr(tool) == "<MeridianTool name='t'>"


# ---------------------------------------------------------------------------
# _decorator: schema-inference fall-through branches
# ---------------------------------------------------------------------------


class _RaisingMeta(type):
    def __subclasscheck__(cls, sub: type) -> bool:
        raise RuntimeError("subclasscheck boom")


class _FakeBase(metaclass=_RaisingMeta):
    pass


def _no_params() -> None:
    return None


def _unannotated(args, ctx) -> None:  # type: ignore[no-untyped-def]
    return None


def _bad_annotation(args: NoSuchName, ctx: Any) -> None:  # type: ignore[name-defined]  # noqa: F821
    return None


def _int_annotation(args: int, ctx: Any) -> None:
    return None


def test_get_pydantic_arg_type_no_params() -> None:
    assert _get_pydantic_arg_type(_no_params) is None


def test_get_pydantic_arg_type_empty_annotation() -> None:
    assert _get_pydantic_arg_type(_unannotated) is None


def test_get_pydantic_arg_type_eval_failure() -> None:
    assert _get_pydantic_arg_type(_bad_annotation) is None


def test_get_pydantic_arg_type_subclasscheck_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_decorator, "BaseModel", _FakeBase)
    assert _get_pydantic_arg_type(_int_annotation) is None


def test_infer_input_schema_no_params() -> None:
    assert _infer_input_schema(_no_params) is None


def test_infer_input_schema_empty_annotation() -> None:
    assert _infer_input_schema(_unannotated) is None


def test_infer_input_schema_eval_failure() -> None:
    assert _infer_input_schema(_bad_annotation) is None


def test_infer_input_schema_subclasscheck_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_decorator, "BaseModel", _FakeBase)
    assert _infer_input_schema(_int_annotation) is None


class _RealArgs(_decorator.BaseModel):
    x: int


def _with_real_model_annotation() -> Any:
    # __future__ annotations stringifies all source annotations, so build a
    # function whose first-param annotation is a real class object to exercise
    # the non-string (isinstance False) branch in both inference helpers.
    def _fn(args, ctx):  # type: ignore[no-untyped-def]
        return None

    _fn.__annotations__ = {"args": _RealArgs, "ctx": Any}
    return _fn


def test_get_pydantic_arg_type_real_model() -> None:
    assert _get_pydantic_arg_type(_with_real_model_annotation()) is _RealArgs


def test_infer_input_schema_real_model() -> None:
    schema = _infer_input_schema(_with_real_model_annotation())
    assert schema == _RealArgs.model_json_schema()


# ---------------------------------------------------------------------------
# _execution: timeout classification branch (99-100)
# ---------------------------------------------------------------------------


async def test_execute_tool_timeout_classified() -> None:
    async def _handler(args: Any, ctx: ToolContext) -> Any:
        raise TimeoutError("too slow")

    definition = ToolDefinition(
        name="slow", description="d", input_schema={}, handler=InProcessHandler()
    )
    result = await execute_tool(definition, {}, _CTX, _handler)
    assert result.is_error
    assert result.error is not None
    assert result.error.code == "execution_timeout"
    assert result.error.details["timeout_reason"] == "too slow"


# ---------------------------------------------------------------------------
# _otel: OTEL-absent returns, timeout_reason attr, span error re-raise
# ---------------------------------------------------------------------------


def test_record_tool_call_error_otel_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_otel, "_OTEL_AVAILABLE", False)
    record_tool_call_error("c", "m")


def test_record_tool_call_error_with_timeout_reason() -> None:
    record_tool_call_error("c", "m", stderr_tail="boom", timeout_reason="slow")


def test_record_tool_call_result_otel_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_otel, "_OTEL_AVAILABLE", False)
    record_tool_call_result(stderr_tail="x")


async def test_tool_span_otel_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_otel, "_OTEL_AVAILABLE", False)
    async with tool_span("t") as span:
        assert span is None


async def test_tool_span_error_sets_status_and_reraises() -> None:
    with pytest.raises(ValueError, match="inner"):
        async with tool_span("t", session_id="s", extra_attrs={"k": "v"}):
            raise ValueError("inner")


async def test_tool_span_without_session_or_attrs() -> None:
    async with tool_span("t") as span:
        assert span is not None


# ---------------------------------------------------------------------------
# http_tool: no-httpx guard, auth header branches, repr
# ---------------------------------------------------------------------------


async def test_call_http_without_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_mod, "_HTTPX_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="httpx is required"):
        await _call_http("http://x", None, {}, _CTX, 1000)


class _FakeResp:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"result": "ok"}


class _FakeClient:
    captured: dict[str, Any] = {}

    def __init__(self, timeout: float | None = None) -> None:
        self._timeout = timeout

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, url: str, json: Any, headers: dict[str, str]) -> _FakeResp:
        _FakeClient.captured = dict(headers)
        return _FakeResp()


async def test_call_http_bearer_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_mod.httpx, "AsyncClient", _FakeClient)
    await _call_http("http://x", {"bearer": "tok"}, {}, _CTX, 1000)
    assert _FakeClient.captured["Authorization"] == "Bearer tok"


async def test_call_http_api_key_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_mod.httpx, "AsyncClient", _FakeClient)
    await _call_http("http://x", {"api_key": "k"}, {}, _CTX, 1000)
    assert _FakeClient.captured["X-Api-Key"] == "k"


async def test_call_http_auth_without_known_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_mod.httpx, "AsyncClient", _FakeClient)
    await _call_http("http://x", {"other": "v"}, {}, _CTX, 1000)
    assert "Authorization" not in _FakeClient.captured
    assert "X-Api-Key" not in _FakeClient.captured


def test_http_tool_repr() -> None:
    tool = http_tool(name="h", description="d", url="http://x", input_schema={"type": "object"})
    assert repr(tool) == "<HttpTool name='h'>"


# ---------------------------------------------------------------------------
# mcp_tool: builder returns a ToolDefinition (line 60)
# ---------------------------------------------------------------------------


def test_mcp_tool_builder() -> None:
    definition = mcp_tool(name="m", description="d", server_url="http://x")
    assert isinstance(definition, ToolDefinition)
    assert definition.handler.tool_name == "m"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# subprocess_tool: timeout + grace-kill path, repr
# ---------------------------------------------------------------------------


def test_subprocess_tool_repr() -> None:
    tool = subprocess_tool(
        name="s", description="d", path="/bin/true", input_schema={"type": "object"}
    )
    assert isinstance(tool, SubprocessTool)
    assert repr(tool) == "<SubprocessTool name='s'>"


async def test_subprocess_tool_timeout_grace_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Script ignores SIGTERM so terminate() does not stop it; the grace timeout
    # then forces a SIGKILL — exercising subprocess_tool.py:68-75 in full.
    monkeypatch.setattr(subprocess_mod, "_SIGKILL_GRACE_S", 0.2)
    script = tmp_path / "hang_tool.py"
    script.write_text(
        textwrap.dedent("""\
            import signal, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(30)
        """)
    )
    tool = subprocess_tool(
        name="hang",
        description="hangs and ignores SIGTERM",
        path=sys.executable,
        input_schema={"type": "object"},
        timeout_ms=100,
    )
    tool.definition.handler.path = sys.executable  # type: ignore[union-attr]

    # Run python <script> by swapping create_subprocess_exec to prepend the script.
    real_exec = asyncio.create_subprocess_exec

    async def _exec(program: str, *args: object, **kwargs: object) -> Any:
        return await real_exec(program, str(script), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)

    result = await tool.execute({}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert result.error.code == "execution_timeout"
