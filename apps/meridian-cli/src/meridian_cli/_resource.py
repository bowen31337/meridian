"""Factory that produces a Click group with CRUD subcommands for a daemon resource.

Every command emits an OTel span, records an invocation event, and on failure
writes the error to the audit log and surfaces a message to stderr.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from ._audit import write_audit
from ._client import DaemonClient, DaemonError
from ._telemetry import get_tracer, record_failure, record_invocation_event

_API_PREFIX = "/v1/x"


def _client(ctx: click.Context) -> DaemonClient:
    return ctx.find_root().obj  # type: ignore[return-value]


def _run(
    resource: str,
    operation: str,
    method: str,
    path: str,
    *,
    ctx: click.Context,
    event_attrs: dict[str, object],
    json_body: Any = None,
) -> None:
    span_name = f"{resource}.{operation}"
    tracer = get_tracer()
    with tracer.start_as_current_span(span_name, attributes={"resource": resource, "operation": operation}) as span:
        record_invocation_event(span, {"event.name": f"{span_name}.invocation", "resource": resource, "operation": operation, **event_attrs})
        write_audit("info", f"{span_name}.invoked", {"resource": resource, "operation": operation, **{k: v for k, v in event_attrs.items() if isinstance(v, (str, int, float, bool))}})

        try:
            result = _client(ctx).request(method, path, json_body=json_body)
        except DaemonError as exc:
            record_failure(span, exc.code, exc.message)
            write_audit("error", f"{span_name}.failed", {"code": exc.code, "message": exc.message})
            click.echo(f"error: [{exc.code}] {exc.message}", err=True)
            sys.exit(1)

        span.add_event(f"{span_name}.completed")
        if result is not None:
            click.echo(json.dumps(result, indent=2))


def make_crud_group(resource: str) -> click.Group:
    """Return a Click group named *resource* with list/get/create/update/delete commands."""

    path_base = f"{_API_PREFIX}/{resource}"

    @click.group(name=resource)
    def grp() -> None:
        pass

    grp.__doc__ = f"Manage {resource}."

    @grp.command("list")
    @click.pass_context
    def _list(ctx: click.Context) -> None:
        f"""List all {resource}."""
        _run(resource, "list", "GET", path_base, ctx=ctx, event_attrs={})

    @grp.command("get")
    @click.argument("id")
    @click.pass_context
    def _get(ctx: click.Context, id: str) -> None:
        f"""Get a {resource[:-1] if resource.endswith('s') else resource} by ID."""
        _run(resource, "get", "GET", f"{path_base}/{id}", ctx=ctx, event_attrs={"id": id})

    @grp.command("create")
    @click.option("--data", required=True, metavar="JSON", help="JSON body for the new resource.")
    @click.pass_context
    def _create(ctx: click.Context, data: str) -> None:
        f"""Create a new {resource[:-1] if resource.endswith('s') else resource}."""
        try:
            body = json.loads(data)
        except json.JSONDecodeError as exc:
            click.echo(f"error: invalid JSON for --data: {exc}", err=True)
            sys.exit(1)
        _run(resource, "create", "POST", path_base, ctx=ctx, event_attrs={}, json_body=body)

    @grp.command("update")
    @click.argument("id")
    @click.option("--data", required=True, metavar="JSON", help="JSON patch body.")
    @click.pass_context
    def _update(ctx: click.Context, id: str, data: str) -> None:
        f"""Update an existing {resource[:-1] if resource.endswith('s') else resource}."""
        try:
            body = json.loads(data)
        except json.JSONDecodeError as exc:
            click.echo(f"error: invalid JSON for --data: {exc}", err=True)
            sys.exit(1)
        _run(resource, "update", "PATCH", f"{path_base}/{id}", ctx=ctx, event_attrs={"id": id}, json_body=body)

    @grp.command("delete")
    @click.argument("id")
    @click.pass_context
    def _delete(ctx: click.Context, id: str) -> None:
        f"""Delete a {resource[:-1] if resource.endswith('s') else resource} by ID."""
        _run(resource, "delete", "DELETE", f"{path_base}/{id}", ctx=ctx, event_attrs={"id": id})

    return grp
