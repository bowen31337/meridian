# Meridian Tool Author Guide

This guide covers everything you need to write, test, and ship a Meridian
tool.  For the formal contract specification see [Architecture §11.5](./ARCHITECTURE.md).

---

## 1. Quick start — in-process Python tool

```python
from meridian_sdk_tool import meridian_tool, ToolContext
from pydantic import BaseModel

class SearchArgs(BaseModel):
    query: str
    max_results: int = 10

@meridian_tool(
    description="Search the knowledge base",
    capabilities=["kb.read[project-docs]"],
)
async def kb_search(args: SearchArgs, ctx: ToolContext) -> dict:
    hits = await _do_search(args.query, args.max_results, ctx.workspace)
    return {"hits": hits}

# Invoke from the harness or tests:
result = await kb_search.execute({"query": "retry policy"}, ctx)
assert not result.is_error
print(result.result["hits"])
```

The `@meridian_tool` decorator:
- infers `name` from the function name (override with `name=`).
- infers `input_schema` from the Pydantic model on the first parameter.
- wraps the handler with input/output validation, an OTel span, structured
  logging, and audit-log writes on failure.

---

## 2. Handler kinds

| Kind | When to use | Builder |
|------|-------------|---------|
| `in_process` | Fast Python logic with no isolation requirement | `@meridian_tool` |
| `subprocess` | Any language; strong isolation; stdin/stdout JSON | `subprocess_tool(...)` |
| `http` | Remote service or sidecar; POST JSON | `http_tool(...)` |
| `mcp` | Existing MCP server | `mcp_tool(...)` |
| `container` | Full OCI image isolation | `ToolDefinition(handler=ContainerHandler(...))` |

### 2.1 Subprocess tools

```python
from meridian_sdk_tool import subprocess_tool

grep = subprocess_tool(
    name="grep",
    description="Search files for a pattern",
    path="/usr/local/bin/grep_wrapper.py",
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["pattern", "path"],
    },
    capabilities=["fs.read[/workspace/**]"],
)
```

The subprocess receives on **stdin**:

```json
{
  "args": {"pattern": "retry", "path": "/workspace/src"},
  "context": {
    "workspace": "/workspace",
    "session_id": "sess_abc",
    "idempotency_key": "run-42"
  }
}
```

It must write to **stdout** exactly one JSON line then exit:

```json
{ "result": { "matches": [...] } }
```

or on failure:

```json
{ "error": { "code": "grep_failed", "message": "pattern invalid" } }
```

Anything written to **stderr** is captured (up to 64 KB) and attached to
the `tool_call.result` audit event.

### 2.2 HTTP tools

```python
from meridian_sdk_tool import http_tool

translator = http_tool(
    name="translate",
    description="Translate text",
    url="http://localhost:8080/translate",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}},
        "required": ["text", "target_lang"],
    },
    capabilities=["net.fetch[translate-service]"],
)
```

The Sandbox POSTs the JSON body `{"args": ..., "context": {...}}` and
expects the same `{"result": ...}` / `{"error": {...}}` shape as
subprocess tools.

---

## 3. Input and output schemas

Schemas are [JSON Schema draft-07](https://json-schema.org/) objects.

For **in-process** tools, declare a Pydantic model as the first parameter
and the schema is inferred automatically.  You can also pass an explicit
`input_schema=` dict to `@meridian_tool` or any builder.

```python
@meridian_tool(
    output_schema={
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    },
)
async def word_count(args: WordCountArgs, ctx: ToolContext) -> dict:
    return {"count": len(args.text.split())}
```

- **Input validation** runs before the handler.  A failure returns
  `ToolResult(is_error=True, error.code="input_validation_failed")`.
- **Output validation** runs after the handler.  A failure returns
  `ToolResult(is_error=True, error.code="output_validation_failed")`.
- Both failures are written to the audit log.

---

## 4. Capabilities

Declare every resource the tool will access:

```python
@meridian_tool(
    capabilities=[
        "fs.read[/workspace/**]",
        "net.fetch[api.example.com]",
    ],
)
async def my_tool(args: MyArgs, ctx: ToolContext) -> dict: ...
```

The Sandbox enforces the **intersection** of the tool's declared
capabilities and the capabilities granted to the agent session.  The
tool never executes if the intersection is empty.  Capability strings
are dotted names with an optional bracket parameter, e.g.
`kb.read[scope]`, `exec.shell`, `agent.spawn[agent_id]`.

---

## 5. Idempotency contract

**Tools must be safe to retry.**  If the caller supplies an
`idempotency_key` on the `ToolContext`, the SDK returns the cached
first result without re-running the handler:

```python
ctx = ToolContext(
    workspace="/workspace",
    session_id="sess_abc",
    idempotency_key="charge-order-99",  # stable caller-chosen key
)

result = await charge.execute(args, ctx)   # handler runs
retry  = await charge.execute(args, ctx)   # handler does NOT run again
assert result == retry
```

Rules:
- Both **success and failure** results are cached per `(tool_name,
  idempotency_key)` pair.
- **Input-validation failures are not cached** — fix the payload, then
  retry with the same key.
- The cache lives in the Sandbox worker process.  Cross-restart
  idempotency uses the session event-log replay path (Architecture §5.4).
- Tool implementations should still be intrinsically idempotent where
  possible (e.g. upsert rather than insert) so they behave correctly
  when called without a key.

---

## 6. Error handling

Return `ToolResult.err(...)` for every failure — do **not** raise:

```python
async def my_tool(args: MyArgs, ctx: ToolContext) -> dict:
    try:
        return await _do_work(args)
    except SomeExpectedError as exc:
        return {"error": {"code": "work_failed", "message": str(exc)}}
    # Don't catch everything here — unexpected exceptions bubble up to
    # the SDK pipeline which wraps them into execution_failed results.
```

For **in-process** tools the SDK catches any unhandled exception and
turns it into `ToolResult(is_error=True, error.code="execution_failed")`.
The session phase never transitions to terminated on a tool failure —
the agent decides whether to retry, escalate, or abandon.

Every failure path writes a record to the audit log
(`~/.meridian/audit.ndjson`) with fields:

| Field | Description |
|-------|-------------|
| `ts` | ISO 8601 timestamp |
| `type` | `tool.<code>` e.g. `tool.execution_failed` |
| `tool_name` | Registered tool name |
| `session_id` | Session that triggered the call |
| `idempotency_key` | Present when the caller supplied one |
| `error` | `{code, message, details}` |

---

## 7. OTel instrumentation

The SDK opens a span for every tool invocation automatically.  For
additional span attributes or child spans inside your handler:

```python
from opentelemetry import trace

tracer = trace.get_tracer("my_tool")

async def my_tool(args: MyArgs, ctx: ToolContext) -> dict:
    with tracer.start_as_current_span("my_tool.fetch") as span:
        span.set_attribute("meridian.tool.query", args.query)
        result = await _fetch(args)
    return result
```

---

## 8. Testing

```python
import pytest
from meridian_sdk_tool import ToolContext
from meridian_sdk_tool._idempotency import clear as clear_idempotency_cache

_CTX = ToolContext(workspace="/workspace", session_id="sess_test")


@pytest.mark.anyio
async def test_success() -> None:
    result = await my_tool.execute({"query": "hello"}, _CTX)
    assert not result.is_error
    assert "hits" in result.result


@pytest.mark.anyio
async def test_idempotent_retry() -> None:
    clear_idempotency_cache()  # isolate from other tests
    ctx = ToolContext(
        workspace="/workspace",
        session_id="sess_test",
        idempotency_key="test-key-1",
    )
    r1 = await my_tool.execute({"query": "hello"}, ctx)
    r2 = await my_tool.execute({"query": "hello"}, ctx)
    assert r1 == r2


@pytest.mark.anyio
async def test_bad_input_returns_error() -> None:
    result = await my_tool.execute({"wrong_field": 1}, _CTX)
    assert result.is_error
    assert result.error is not None
    assert "validation" in result.error.code
```

Use `meridian_sdk_tool._idempotency.clear()` in test fixtures to prevent
cached results from leaking between test cases.

---

## 9. Checklist

- [ ] Declare all required capabilities.
- [ ] Validate inputs via Pydantic model or explicit `input_schema`.
- [ ] Declare `output_schema` if the result shape matters.
- [ ] Handle expected errors with `ToolResult.err(...)` rather than raising.
- [ ] Make the implementation intrinsically idempotent where possible.
- [ ] Respect `ctx.idempotency_key` semantics — the SDK handles caching,
      but your handler should not perform side-effects that cannot be
      safely replayed.
- [ ] Test both happy path and error paths.
- [ ] Call `_idempotency.clear()` between test cases that use a key.
